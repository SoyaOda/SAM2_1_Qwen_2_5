# model/moe_integration_adapter.py
"""
MoE統合アダプター - Phase 2実装
統一トークンをLlama-4のMoE層に効率的にルーティング
視覚トークン専用のexpertを活用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class MoEIntegrationAdapter(nn.Module):
    """
    統一トークンをLlama-4のMoE層に効率的にルーティング
    Llama-4-Scout-17B-16E-Instructの構造に最適化
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 設定
        self.llama_dim = config.llama_hidden_size  # 5120
        self.num_experts = 16  # Llama-4-Scout-17B-16E
        self.num_active_experts = 2  # 各トークンが使用するエキスパート数
        
        # トークンタイプ埋め込み（vision, text, seg）
        self.token_type_embeddings = nn.Embedding(3, self.llama_dim)
        self.token_type_map = {'vision': 0, 'text': 1, 'seg': 2}
        
        # ルーティング最適化層
        # モダリティに応じて適切なエキスパートを選択
        self.routing_optimizer = nn.Sequential(
            nn.Linear(self.llama_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_experts)
        )
        
        # エキスパート専門性予測器
        # 各エキスパートがどのモダリティに特化しているかを学習
        self.expert_specialization = nn.Parameter(
            torch.randn(self.num_experts, 3) * 0.1  # 3モダリティ
        )
        
        # モダリティ別ルーティングバイアス
        self.modality_routing_bias = nn.Parameter(
            torch.zeros(3, self.num_experts)
        )
        
        # ゲート温度制御（動的調整）
        self.temperature_controller = nn.Sequential(
            nn.Linear(self.llama_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.base_temperature = 1.0
        self.temperature_range = 2.0  # 0.5 ~ 2.5の範囲
        
        # ロードバランシング損失の係数
        self.load_balance_alpha = 0.01
        
        # トークン重要度予測器
        self.token_importance_predictor = nn.Sequential(
            nn.Linear(self.llama_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 初期化
        self._init_weights()
        self._init_expert_specialization()
        
    def _init_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _init_expert_specialization(self):
        """エキスパート専門性の初期化"""
        # 一部のエキスパートを視覚専門に初期化
        vision_experts = [0, 1, 2, 3]  # 最初の4つを視覚専門
        text_experts = [4, 5, 6, 7, 8, 9]  # 次の6つをテキスト専門
        seg_experts = [10, 11]  # 2つをSEG専門
        # 残りは汎用
        
        with torch.no_grad():
            # 視覚専門エキスパート
            for idx in vision_experts:
                self.expert_specialization[idx, 0] = 1.0  # vision
                self.expert_specialization[idx, 1] = 0.2  # text
                self.expert_specialization[idx, 2] = 0.5  # seg
            
            # テキスト専門エキスパート
            for idx in text_experts:
                self.expert_specialization[idx, 0] = 0.2  # vision
                self.expert_specialization[idx, 1] = 1.0  # text
                self.expert_specialization[idx, 2] = 0.3  # seg
            
            # SEG専門エキスパート
            for idx in seg_experts:
                self.expert_specialization[idx, 0] = 0.5  # vision
                self.expert_specialization[idx, 1] = 0.3  # text
                self.expert_specialization[idx, 2] = 1.0  # seg
            
            # Softmax正規化
            self.expert_specialization.data = F.softmax(self.expert_specialization, dim=1)
    
    def compute_routing_scores(
        self,
        tokens: torch.Tensor,
        modality_masks: Dict[str, torch.Tensor],
        temperature: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ルーティングスコアの計算
        
        Args:
            tokens: 統一トークン (B, N, D)
            modality_masks: 各モダリティのマスク
            temperature: ゲート温度
        
        Returns:
            routing_weights: ルーティング重み (B, N, num_experts)
            expert_indices: 選択されたエキスパートのインデックス (B, N, num_active_experts)
        """
        batch_size, seq_len, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        
        # dtype統一を確実にする
        self.routing_optimizer = self.routing_optimizer.to(device=device, dtype=dtype)
        
        # 基本ルーティングスコア
        routing_logits = self.routing_optimizer(tokens)  # (B, N, num_experts)
        
        # モダリティ別バイアスの適用
        modality_bias = torch.zeros_like(routing_logits)
        for modality, mask in modality_masks.items():
            if modality in self.token_type_map:
                modality_id = self.token_type_map[modality]
                bias = self.modality_routing_bias[modality_id].unsqueeze(0).unsqueeze(0)
                modality_bias += mask.unsqueeze(-1).to(dtype=routing_logits.dtype) * bias
        
        routing_logits = routing_logits + modality_bias
        
        # 温度制御
        if temperature is None:
            # dtype統一
            self.temperature_controller = self.temperature_controller.to(device=device, dtype=dtype)
            
            # 動的温度の計算
            token_complexity = tokens.norm(dim=-1, keepdim=True)  # (B, N, 1)
            temperature_scale = self.temperature_controller(tokens)  # (B, N, 1)
            temperature = self.base_temperature + temperature_scale * self.temperature_range
        
        # Softmax with temperature
        routing_weights = F.softmax(routing_logits / temperature, dim=-1)
        
        # Top-k選択（各トークンに対してk個のエキスパートを選択）
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, 
            self.num_active_experts, 
            dim=-1
        )
        
        # 重みの再正規化
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, top_k_indices
    
    def compute_load_balance_loss(
        self,
        routing_weights: torch.Tensor,
        token_importance: torch.Tensor
    ) -> torch.Tensor:
        """
        ロードバランシング損失の計算
        各エキスパートへの負荷を均等化
        
        Args:
            routing_weights: ルーティング重み (B, N, num_experts)
            token_importance: トークン重要度 (B, N, 1)
        
        Returns:
            load_balance_loss: ロードバランシング損失
        """
        # 重要度加重平均負荷
        weighted_routing = routing_weights * token_importance
        expert_load = weighted_routing.sum(dim=(0, 1)) / token_importance.sum()
        
        # 理想的な均等負荷
        ideal_load = 1.0 / self.num_experts
        
        # 損失計算（分散を最小化）
        load_balance_loss = ((expert_load - ideal_load) ** 2).sum()
        
        return load_balance_loss * self.load_balance_alpha
    
    def forward(
        self,
        unified_tokens: torch.Tensor,
        modality_masks: Dict[str, torch.Tensor],
        return_routing_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        MoE統合アダプターの適用
        
        Args:
            unified_tokens: 統一されたトークン (B, N, D)
            modality_masks: 各モダリティのマスク
            return_routing_info: ルーティング情報を返すか
        
        Returns:
            outputs: アダプター出力
                - 'adapted_tokens': MoE最適化されたトークン
                - 'routing_weights': ルーティング重み（オプション）
                - 'load_balance_loss': ロードバランシング損失
        """
        batch_size, seq_len, hidden_dim = unified_tokens.shape
        device = unified_tokens.device
        
        # 1. トークンタイプ埋め込みの追加
        token_type_embeds = torch.zeros_like(unified_tokens)
        for modality, mask in modality_masks.items():
            if modality in self.token_type_map:
                type_id = self.token_type_map[modality]
                type_embed = self.token_type_embeddings(
                    torch.tensor(type_id, device=device, dtype=torch.long)
                )
                # dtypeを統一
                token_type_embeds += mask.unsqueeze(-1).to(dtype=unified_tokens.dtype) * type_embed
        
        # 残差接続
        enhanced_tokens = unified_tokens + 0.1 * token_type_embeds
        
        # 2. トークン重要度の計算
        self.token_importance_predictor = self.token_importance_predictor.to(device=device, dtype=enhanced_tokens.dtype)
        token_importance = self.token_importance_predictor(enhanced_tokens)
        
        # 3. ルーティングスコアの計算
        routing_weights, expert_indices = self.compute_routing_scores(
            enhanced_tokens,
            modality_masks
        )
        
        # 4. エキスパート専門性に基づく調整
        # 各トークンのモダリティタイプを取得
        token_modalities = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        for modality, mask in modality_masks.items():
            if modality in self.token_type_map:
                token_modalities[mask] = self.token_type_map[modality]
        
        # エキスパート専門性スコアの取得
        modality_one_hot = F.one_hot(token_modalities, num_classes=3).to(dtype=enhanced_tokens.dtype)  # (B, N, 3)
        expert_affinity = torch.matmul(
            modality_one_hot, 
            self.expert_specialization.T.to(dtype=enhanced_tokens.dtype)
        )  # (B, N, num_experts)
        
        # ルーティング重みを専門性で調整
        adjusted_routing_weights = routing_weights * (1 + 0.5 * expert_affinity)
        adjusted_routing_weights = adjusted_routing_weights / adjusted_routing_weights.sum(dim=-1, keepdim=True)
        
        # 5. ロードバランシング損失の計算
        load_balance_loss = self.compute_load_balance_loss(
            adjusted_routing_weights,
            token_importance
        )
        
        # 6. 適応的トークンの生成（ルーティング情報を埋め込む）
        # 選択されたエキスパートの情報をトークンに埋め込む
        routing_embedding = torch.zeros(
            batch_size, seq_len, self.num_experts, 
            device=device, dtype=unified_tokens.dtype
        )
        routing_embedding.scatter_(
            2, 
            expert_indices, 
            torch.ones_like(expert_indices, dtype=unified_tokens.dtype)
        )
        routing_features = routing_embedding.view(batch_size, seq_len, -1)
        
        # 線形変換で次元を合わせる
        if not hasattr(self, 'routing_projector'):
            self.routing_projector = nn.Linear(
                self.num_experts, 
                hidden_dim
            ).to(device=device, dtype=unified_tokens.dtype)
            nn.init.xavier_uniform_(self.routing_projector.weight)
        
        routing_info = self.routing_projector(routing_features)
        
        # 最終的な適応トークン
        adapted_tokens = enhanced_tokens + 0.1 * routing_info
        
        # 7. 出力の構築
        outputs = {
            'adapted_tokens': adapted_tokens,
            'load_balance_loss': load_balance_loss,
            'token_importance': token_importance,
        }
        
        if return_routing_info:
            outputs.update({
                'routing_weights': adjusted_routing_weights,
                'expert_indices': expert_indices,
                'expert_affinity': expert_affinity,
                'modality_distribution': modality_one_hot.mean(dim=1),  # (B, 3)
            })
        
        # ログ出力
        logger.debug(f"MoE統合完了: "
                    f"アクティブエキスパート={self.num_active_experts}, "
                    f"ロードバランス損失={load_balance_loss.item():.4f}")
        
        return outputs


def create_moe_integration_adapter(config) -> MoEIntegrationAdapter:
    """
    MoE統合アダプターのファクトリ関数
    
    Args:
        config: モデル設定
    
    Returns:
        MoEIntegrationAdapter インスタンス
    """
    logger.info("Creating MoEIntegrationAdapter for Phase 2 implementation")
    return MoEIntegrationAdapter(config)