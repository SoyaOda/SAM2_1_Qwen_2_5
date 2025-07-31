# model/cross_modal_unifier.py
"""
クロスモーダル統一層 - Phase 2実装
視覚・テキスト・[SEG]トークンを統一空間で処理
Llama-4のearly fusionアプローチを採用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CrossModalUnifier(nn.Module):
    """
    視覚・テキスト・[SEG]トークンを統一空間で処理
    Llama-4のearly fusionアプローチに基づく実装
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 設定
        self.qformer_dim = config.qformer_config['hidden_size']  # 768
        self.llama_dim = config.llama_hidden_size  # 5120
        self.sam_prompt_dim = config.qformer_config['sam_prompt_dim']  # 256
        
        # モダリティ別投影層
        self.projectors = nn.ModuleDict({
            'vision': nn.Sequential(
                nn.Linear(self.llama_dim, self.llama_dim),  # 既に圧縮器でLlama次元
                nn.LayerNorm(self.llama_dim),
                nn.GELU()
            ),
            'seg': nn.Identity(),  # 既にLlama次元
            'text': nn.Identity(),  # 既にLlama次元
        })
        
        # モダリティ位置埋め込み（3種類: vision, seg, text）
        self.modality_embeddings = nn.Embedding(3, self.llama_dim)
        self.modality_type_map = {'vision': 0, 'seg': 1, 'text': 2}
        
        # クロスモーダル融合層（Transformer Encoder）
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.llama_dim,
                nhead=16,  # Llama-4のアテンションヘッド数に合わせる
                dim_feedforward=self.llama_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN for stability
            ) for _ in range(2)  # 2層で軽量化
        ])
        
        # トークン位置埋め込み（最大512トークン）
        self.max_position_embeddings = 512
        self.position_embeddings = nn.Embedding(
            self.max_position_embeddings, 
            self.llama_dim
        )
        
        # ゲート機構（モダリティ間の相互作用を制御）
        self.modality_gates = nn.ModuleDict({
            'vision_to_text': nn.Sequential(
                nn.Linear(self.llama_dim * 2, self.llama_dim),
                nn.Sigmoid()
            ),
            'text_to_vision': nn.Sequential(
                nn.Linear(self.llama_dim * 2, self.llama_dim),
                nn.Sigmoid()
            ),
            'seg_to_all': nn.Sequential(
                nn.Linear(self.llama_dim, self.llama_dim),
                nn.Sigmoid()
            )
        })
        
        # 初期化
        self._init_weights()
        
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
    
    def apply_modality_embeddings(
        self,
        tokens: torch.Tensor,
        modality_type: str
    ) -> torch.Tensor:
        """
        モダリティ埋め込みの適用
        
        Args:
            tokens: 入力トークン (B, N, D)
            modality_type: 'vision', 'seg', or 'text'
        
        Returns:
            モダリティ埋め込み付きトークン
        """
        batch_size, seq_len = tokens.shape[:2]
        device = tokens.device
        
        # モダリティIDの取得
        modality_id = self.modality_type_map[modality_type]
        
        # モダリティ埋め込みの生成
        modality_ids = torch.full(
            (batch_size, seq_len), 
            modality_id, 
            dtype=torch.long, 
            device=device
        )
        modality_embeds = self.modality_embeddings(modality_ids)
        
        return tokens + modality_embeds
    
    def apply_position_embeddings(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        位置埋め込みの適用
        
        Args:
            tokens: 入力トークン (B, N, D)
            start_pos: 開始位置
        
        Returns:
            位置埋め込み付きトークン
        """
        batch_size, seq_len = tokens.shape[:2]
        device = tokens.device
        
        # 位置IDの生成
        position_ids = torch.arange(
            start_pos, 
            start_pos + seq_len, 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # 最大位置を超えないようにクリップ
        position_ids = torch.clamp(position_ids, max=self.max_position_embeddings - 1)
        
        position_embeds = self.position_embeddings(position_ids)
        
        return tokens + position_embeds
    
    def apply_modality_gates(
        self,
        unified_tokens: torch.Tensor,
        modality_masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        モダリティ間のゲート機構を適用
        
        Args:
            unified_tokens: 統一されたトークン (B, N, D)
            modality_masks: 各モダリティのマスク
        
        Returns:
            ゲート適用後のトークン
        """
        batch_size, seq_len, hidden_dim = unified_tokens.shape
        device = unified_tokens.device
        
        # モダリティ別の平均表現を計算
        modality_means = {}
        for modality, mask in modality_masks.items():
            if mask.any():
                masked_tokens = unified_tokens * mask.unsqueeze(-1)
                modality_means[modality] = masked_tokens.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                modality_means[modality] = torch.zeros(batch_size, hidden_dim, device=device)
        
        # ゲート適用
        gated_tokens = unified_tokens.clone()
        
        # Vision → Text ゲート
        if 'vision' in modality_means and 'text' in modality_masks:
            vision_text_concat = torch.cat([
                modality_means['vision'].unsqueeze(1).expand(-1, seq_len, -1),
                unified_tokens
            ], dim=-1)
            vision_to_text_gate = self.modality_gates['vision_to_text'](vision_text_concat)
            gated_tokens = gated_tokens + vision_to_text_gate * modality_masks['text'].unsqueeze(-1)
        
        # Text → Vision ゲート
        if 'text' in modality_means and 'vision' in modality_masks:
            text_vision_concat = torch.cat([
                modality_means['text'].unsqueeze(1).expand(-1, seq_len, -1),
                unified_tokens
            ], dim=-1)
            text_to_vision_gate = self.modality_gates['text_to_vision'](text_vision_concat)
            gated_tokens = gated_tokens + text_to_vision_gate * modality_masks['vision'].unsqueeze(-1)
        
        # SEG → All ゲート
        if 'seg' in modality_means:
            seg_gate = self.modality_gates['seg_to_all'](
                modality_means['seg'].unsqueeze(1).expand(-1, seq_len, -1)
            )
            gated_tokens = gated_tokens * (1 + seg_gate)  # 乗算的ゲート
        
        return gated_tokens
    
    def forward(
        self,
        vision_tokens: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        seg_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_separated: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        クロスモーダル統一処理
        
        Args:
            vision_tokens: 視覚トークン (B, N_v, D) - 圧縮済み
            text_tokens: テキストトークン (B, N_t, D)
            seg_tokens: [SEG]トークン (B, N_s, D) - 通常1個
            attention_mask: アテンションマスク
            return_separated: モダリティ別に分離して返すか
        
        Returns:
            outputs: 統一処理結果
                - 'unified_tokens': 統一されたトークン
                - 'modality_masks': 各モダリティのマスク
                - その他の情報
        """
        tokens_list = []
        modality_types = []
        modality_masks = {}
        position_offset = 0
        
        # 1. 各モダリティのトークンを収集・投影
        if vision_tokens is not None:
            vision_projected = self.projectors['vision'](vision_tokens)
            vision_embedded = self.apply_modality_embeddings(vision_projected, 'vision')
            vision_positioned = self.apply_position_embeddings(vision_embedded, position_offset)
            tokens_list.append(vision_positioned)
            modality_types.extend(['vision'] * vision_tokens.size(1))
            position_offset += vision_tokens.size(1)
            
            # ビジョンマスク
            vision_mask = torch.zeros(len(tokens_list), dtype=torch.bool)
            vision_mask[0] = True
            modality_masks['vision'] = vision_mask
        
        if seg_tokens is not None:
            seg_projected = self.projectors['seg'](seg_tokens)
            seg_embedded = self.apply_modality_embeddings(seg_projected, 'seg')
            seg_positioned = self.apply_position_embeddings(seg_embedded, position_offset)
            tokens_list.append(seg_positioned)
            modality_types.extend(['seg'] * seg_tokens.size(1))
            position_offset += seg_tokens.size(1)
            
            # SEGマスク
            if vision_tokens is not None:
                seg_mask = torch.zeros(len(tokens_list), dtype=torch.bool)
                seg_mask[1] = True
                modality_masks['seg'] = seg_mask
            else:
                seg_mask = torch.zeros(len(tokens_list), dtype=torch.bool)
                seg_mask[0] = True
                modality_masks['seg'] = seg_mask
        
        if text_tokens is not None:
            text_projected = self.projectors['text'](text_tokens)
            text_embedded = self.apply_modality_embeddings(text_projected, 'text')
            text_positioned = self.apply_position_embeddings(text_embedded, position_offset)
            tokens_list.append(text_positioned)
            modality_types.extend(['text'] * text_tokens.size(1))
            
            # テキストマスク
            text_mask = torch.zeros(len(tokens_list), dtype=torch.bool)
            text_mask[-1] = True
            modality_masks['text'] = text_mask
        
        # 2. トークンの結合
        if not tokens_list:
            raise ValueError("At least one modality must be provided")
        
        unified_tokens = torch.cat(tokens_list, dim=1)  # (B, N_total, D)
        batch_size, total_seq_len, _ = unified_tokens.shape
        
        # モダリティマスクの作成（統一トークン用）
        unified_modality_masks = {}
        current_pos = 0
        for i, tokens in enumerate(tokens_list):
            seq_len = tokens.size(1)
            mask = torch.zeros(batch_size, total_seq_len, dtype=torch.bool, device=tokens.device)
            mask[:, current_pos:current_pos + seq_len] = True
            
            if i == 0 and vision_tokens is not None:
                unified_modality_masks['vision'] = mask
            elif seg_tokens is not None and ((i == 1 and vision_tokens is not None) or (i == 0 and vision_tokens is None)):
                unified_modality_masks['seg'] = mask
            else:
                unified_modality_masks['text'] = mask
            
            current_pos += seq_len
        
        # 3. クロスモーダル融合
        fused_tokens = unified_tokens
        for fusion_layer in self.fusion_layers:
            # アテンションマスクの適用
            if attention_mask is not None:
                # パディングマスクをTransformer形式に変換
                src_key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
            else:
                src_key_padding_mask = None
            
            fused_tokens = fusion_layer(
                fused_tokens,
                src_key_padding_mask=src_key_padding_mask
            )
        
        # 4. モダリティゲートの適用
        gated_tokens = self.apply_modality_gates(fused_tokens, unified_modality_masks)
        
        # 5. 出力の構築
        outputs = {
            'unified_tokens': gated_tokens,
            'modality_masks': unified_modality_masks,
            'fusion_complete': True,
            'total_seq_len': total_seq_len,
            'modality_types': modality_types,
        }
        
        # 6. モダリティ別に分離（オプション）
        if return_separated:
            separated_tokens = {}
            for modality, mask in unified_modality_masks.items():
                separated_tokens[modality] = gated_tokens[mask].reshape(
                    batch_size, -1, self.llama_dim
                )
            outputs['separated_tokens'] = separated_tokens
        
        # ログ出力
        logger.debug(f"クロスモーダル統一完了: "
                    f"総トークン数={total_seq_len}, "
                    f"モダリティ={list(unified_modality_masks.keys())}")
        
        return outputs


def create_cross_modal_unifier(config) -> CrossModalUnifier:
    """
    クロスモーダル統一層のファクトリ関数
    
    Args:
        config: モデル設定
    
    Returns:
        CrossModalUnifier インスタンス
    """
    logger.info("Creating CrossModalUnifier for Phase 2 implementation")
    return CrossModalUnifier(config)