# model/seg_token_generator.py
"""
Sa2VA風[SEG]トークン生成器
Q-Former出力を活用した軽量実装でSa2VAの核心アイデアを実現

参考: Sa2VA: Marrying SAM2 with LLaVA (arXiv:2501.04001)
- [SEG]トークンの隠れ状態をSAM2の空間-時間プロンプトとして使用
- 2層の線形変換でLLM空間からSAMプロンプト空間へ投影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class LightweightSEGTokenGenerator(nn.Module):
    """
    Q-Former出力を活用した軽量[SEG]トークン生成器
    Sa2VAの核心アイデアを最小限の変更で実現
    """
    
    def __init__(self, config):
        super().__init__()
        # 次元設定
        self.qformer_dim = config.qformer_config['hidden_size']  # 768
        self.llama_dim = config.llama_hidden_size  # 5120
        self.sam_prompt_dim = config.qformer_config['sam_prompt_dim']  # 256
        
        # [SEG]トークン生成パイプライン
        self.seg_token_projector = nn.Sequential(
            nn.Linear(self.qformer_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, self.llama_dim),
            nn.LayerNorm(self.llama_dim)
        )
        
        # SAM2プロンプト変換（Sa2VA準拠: 2層の線形変換）
        self.to_sam_prompt = nn.Sequential(
            nn.Linear(self.llama_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.sam_prompt_dim)
        )
        
        # 学習可能な[SEG]トークン埋め込み
        self.seg_token_embedding = nn.Parameter(
            torch.randn(1, 1, self.qformer_dim) * 0.02
        )
        
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
    
    def forward(
        self, 
        qformer_outputs: Dict[str, torch.Tensor],
        llama_hidden_states: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Q-Former出力から[SEG]トークンとSAM2プロンプトを生成
        
        Args:
            qformer_outputs: Q-Formerの出力辞書
                - 'query_embeds': (B, num_queries, qformer_dim)
            llama_hidden_states: Llama-4の隠れ状態（オプション）
                - (B, seq_len, llama_dim)
            return_attention: 注意重みを返すかどうか
        
        Returns:
            seg_outputs: [SEG]トークンとSAM2制御信号
                - 'seg_token': Llama空間の[SEG]トークン (B, llama_dim)
                - 'sam_prompt': SAM2制御用プロンプト (B, sam_prompt_dim)
                - 'attention_weights': Q-Formerクエリへの注意重み (B, num_queries)
        """
        # デバッグ: 入力デバイス確認
        logger.info("🔍 SEGトークン生成器内部デバイス詳細:")
        for key, value in qformer_outputs.items():
            if hasattr(value, 'device'):
                logger.info(f"  - 入力 {key}: device={value.device}, dtype={value.dtype}, shape={value.shape}")
        
        if llama_hidden_states is not None:
            logger.info(f"  - llama_hidden_states: device={llama_hidden_states.device}, dtype={llama_hidden_states.dtype}")
        
        # 生成器パラメータのデバイス確認
        logger.info(f"  - self.seg_token_projector[0].weight: device={self.seg_token_projector[0].weight.device}")
        logger.info(f"  - self.to_sam_prompt[0].weight: device={self.to_sam_prompt[0].weight.device}")
        logger.info(f"  - self.seg_token_embedding: device={self.seg_token_embedding.device}")
        
        # 入力検証
        if 'query_embeds' not in qformer_outputs:
            raise ValueError("qformer_outputs must contain 'query_embeds'")
        
        query_embeds = qformer_outputs['query_embeds']
        
        # 空テンソルチェック
        if query_embeds.numel() == 0:
            raise ValueError("query_embeds is an empty tensor")
        
        # 次元チェック
        if query_embeds.dim() < 3:
            raise ValueError(f"query_embeds must be at least 3D, got {query_embeds.dim()}D tensor with shape {query_embeds.shape}")
            
        batch_size = query_embeds.size(0)
        
        # 提案A: SEGトークン生成器のデバイスに統一（最小侵襲修正）
        target_device = next(self.parameters()).device  # SEGトークン生成器のデバイス（cuda:0）
        target_dtype = query_embeds.dtype
        
        # 入力テンソルをSEGトークン生成器のデバイスに移動
        logger.info(f"  - デバイス統一: {query_embeds.device} → {target_device}")
        query_embeds = query_embeds.to(device=target_device, dtype=target_dtype)
        
        if llama_hidden_states is not None:
            logger.info(f"  - llama_hidden_states移動: {llama_hidden_states.device} → {target_device}")
            llama_hidden_states = llama_hidden_states.to(device=target_device, dtype=target_dtype)
        
        device = target_device
        dtype = target_dtype
        logger.info(f"  - 統一後デバイス: {device}, dtype: {dtype}")
        
        # 2. 学習可能な[SEG]トークン埋め込みとの注意機構
        logger.info(f"  - seg_token_embedding元デバイス: {self.seg_token_embedding.device}")
        logger.info(f"  - 目標デバイス: {device}, 目標dtype: {dtype}")
        seg_embedding = self.seg_token_embedding.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        logger.info(f"  - seg_embedding移動後: device={seg_embedding.device}, dtype={seg_embedding.dtype}")
        
        # クエリとの類似度計算（スケーリング付きドット積注意）
        attention_scores = torch.matmul(
            seg_embedding, 
            query_embeds.transpose(-1, -2)
        ) / (self.qformer_dim ** 0.5)  # (B, 1, 32)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 重み付き集約
        seg_token_qformer = torch.matmul(
            attention_weights, 
            query_embeds
        ).squeeze(1)  # (B, 768)
        
        # 3. Llama次元への投影
        logger.info(f"  - seg_token_qformer: device={seg_token_qformer.device}, shape={seg_token_qformer.shape}")
        logger.info(f"  - seg_token_projector device: {next(self.seg_token_projector.parameters()).device}")
        seg_token_hidden = self.seg_token_projector(seg_token_qformer)  # (B, 5120)
        logger.info(f"  - seg_token_hidden: device={seg_token_hidden.device}, shape={seg_token_hidden.shape}")
        
        # 4. Llama隠れ状態との融合（利用可能な場合）
        if llama_hidden_states is not None:
            # Sa2VA準拠：最終層の隠れ状態を使用
            if llama_hidden_states.dim() == 3:
                # シーケンス次元がある場合は平均プーリング
                llama_pooled = llama_hidden_states.mean(dim=1)  # (B, 5120)
            else:
                llama_pooled = llama_hidden_states  # (B, 5120)
            
            # 残差接続で融合（学習可能な重み付き）
            logger.info(f"  - llama_pooled: device={llama_pooled.device}, shape={llama_pooled.shape}")
            logger.info(f"  - 融合前 seg_token_hidden: device={seg_token_hidden.device}")
            seg_token_hidden = seg_token_hidden + 0.5 * llama_pooled
            logger.info(f"  - 融合後 seg_token_hidden: device={seg_token_hidden.device}")
        
        # 5. SAM2プロンプト生成（Sa2VA準拠）
        logger.info(f"  - to_sam_prompt入力: device={seg_token_hidden.device}, shape={seg_token_hidden.shape}")
        logger.info(f"  - to_sam_prompt device: {next(self.to_sam_prompt.parameters()).device}")
        sam_prompt = self.to_sam_prompt(seg_token_hidden)  # (B, 256)
        logger.info(f"  - sam_prompt出力: device={sam_prompt.device}, shape={sam_prompt.shape}")
        
        # 6. 出力辞書の構築
        outputs = {
            'seg_token': seg_token_hidden,  # Llama空間の[SEG]トークン
            'sam_prompt': sam_prompt,  # SAM2制御用プロンプト
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights.squeeze(1)  # (B, 32)
        
        # デバッグ情報
        logger.debug(f"[SEG] token shape: {seg_token_hidden.shape}")
        logger.debug(f"SAM prompt shape: {sam_prompt.shape}")
        if return_attention:
            # 最も注目されているクエリのインデックス
            top_queries = attention_weights.squeeze(1).argmax(dim=-1)
            logger.debug(f"Top attended queries: {top_queries.tolist()}")
        
        return outputs


class MultiFrameSEGTokenGenerator(LightweightSEGTokenGenerator):
    """
    動画処理用の拡張版[SEG]トークン生成器
    複数フレームに対応し、時間的一貫性を保持
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 時間的集約のための追加層
        self.temporal_fusion = nn.LSTM(
            input_size=self.llama_dim,
            hidden_size=self.llama_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # フレーム数予測器（Sa2VAのrepeat戦略改善版）
        self.frame_predictor = nn.Sequential(
            nn.Linear(self.llama_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # 連続値として予測
            nn.Sigmoid()
        )
        
    def forward(
        self,
        qformer_outputs: Dict[str, torch.Tensor],
        llama_hidden_states: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        動画用の[SEG]トークン生成
        
        Args:
            num_frames: 処理するフレーム数（Noneの場合は自動推定）
        """
        # 基本的な[SEG]トークン生成
        base_outputs = super().forward(
            qformer_outputs, 
            llama_hidden_states, 
            return_attention
        )
        
        seg_token = base_outputs['seg_token']  # (B, llama_dim)
        batch_size = seg_token.size(0)
        
        # フレーム数の決定
        if num_frames is None:
            # 自動推定（1-16の範囲）
            frame_score = self.frame_predictor(seg_token)  # (B, 1)
            num_frames = torch.clamp(
                (frame_score * 15 + 1).round().long(), 
                min=1, 
                max=16
            ).item()
        
        # 複数フレーム用の[SEG]トークン生成
        if num_frames > 1:
            # 時間的拡張
            seg_tokens_expanded = seg_token.unsqueeze(1).expand(
                -1, num_frames, -1
            )  # (B, T, llama_dim)
            
            # 時間的融合（各フレームに微小な変化を加える）
            seg_tokens_fused, _ = self.temporal_fusion(seg_tokens_expanded)
            
            # SAMプロンプトも同様に拡張
            sam_prompts = []
            for t in range(num_frames):
                frame_seg_token = seg_tokens_fused[:, t, :]
                frame_sam_prompt = self.to_sam_prompt(frame_seg_token)
                sam_prompts.append(frame_sam_prompt)
            
            sam_prompts = torch.stack(sam_prompts, dim=1)  # (B, T, sam_prompt_dim)
            
            base_outputs['seg_tokens_temporal'] = seg_tokens_fused
            base_outputs['sam_prompts_temporal'] = sam_prompts
            base_outputs['num_frames'] = num_frames
        
        return base_outputs


def create_seg_token_generator(config, multi_frame: bool = False):
    """
    [SEG]トークン生成器のファクトリ関数
    
    Args:
        config: モデル設定
        multi_frame: 動画処理用の拡張版を使用するか
    
    Returns:
        SEGトークン生成器のインスタンス
    """
    if multi_frame:
        logger.info("Creating multi-frame SEG token generator")
        return MultiFrameSEGTokenGenerator(config)
    else:
        logger.info("Creating lightweight SEG token generator")
        return LightweightSEGTokenGenerator(config)