# model/adaptive_token_compressor.py
"""
適応的トークン圧縮器 - Phase 2実装
Q-Formerクエリを1-8個の統一トークンに動的圧縮
LLaVA-Miniの実証結果に基づく効率的な実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D正弦波位置エンコーディング
    空間情報を保持しながらトークンを圧縮
    """
    
    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        
        # 位置エンコーディングの事前計算
        pe = self._create_pe(max_h, max_w, d_model)
        self.register_buffer('pe', pe)
        
    def _create_pe(self, h: int, w: int, d_model: int) -> torch.Tensor:
        """2D位置エンコーディングの生成"""
        pe = torch.zeros(h, w, d_model)
        
        # 各次元に対する周波数を計算
        d_h = d_model // 2
        d_w = d_model - d_h
        
        # 高さ方向のエンコーディング
        pos_h = torch.arange(h).unsqueeze(1).float()
        div_term_h = torch.exp(torch.arange(0, d_h, 2).float() * 
                               -(math.log(10000.0) / d_h))
        
        pe[:, :, 0:d_h:2] = torch.sin(pos_h * div_term_h).unsqueeze(1).expand(-1, w, -1)
        pe[:, :, 1:d_h:2] = torch.cos(pos_h * div_term_h).unsqueeze(1).expand(-1, w, -1)
        
        # 幅方向のエンコーディング
        pos_w = torch.arange(w).unsqueeze(0).float()
        # d_wが奇数の場合の処理を追加
        div_term_w_sin = torch.exp(torch.arange(0, d_w, 2).float() * 
                                   -(math.log(10000.0) / d_w))
        div_term_w_cos = torch.exp(torch.arange(0, d_w - (d_w % 2), 2).float() * 
                                   -(math.log(10000.0) / d_w))
        
        # sin成分
        sin_values = torch.sin(pos_w.unsqueeze(-1) * div_term_w_sin)  # (1, w, num_freqs)
        pe[:, :, d_h:d_h+len(div_term_w_sin)*2:2] = sin_values.expand(h, -1, -1)
        
        # cos成分（奇数次元の場合は最後の要素を除く）
        if len(div_term_w_cos) > 0:
            cos_values = torch.cos(pos_w.unsqueeze(-1) * div_term_w_cos)  # (1, w, num_freqs)
            pe[:, :, d_h+1:d_h+1+len(div_term_w_cos)*2:2] = cos_values.expand(h, -1, -1)
        
        return pe
        
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        位置エンコーディングの適用
        
        Args:
            x: 入力テンソル (B, N, D)
            h, w: 空間次元
        
        Returns:
            位置エンコーディング付きテンソル
        """
        if h > self.max_h or w > self.max_w:
            # 動的に大きなサイズに対応
            pe = self._create_pe(h, w, self.d_model).to(x.device)
        else:
            pe = self.pe[:h, :w]
        
        # (H, W, D) -> (H*W, D) -> (1, H*W, D)
        pe_flat = pe.reshape(-1, self.d_model).unsqueeze(0)
        
        # バッチサイズに合わせて拡張
        pe_flat = pe_flat.expand(x.size(0), -1, -1)
        
        return x + pe_flat[:, :x.size(1)]


class AdaptiveTokenCompressor(nn.Module):
    """
    Q-Formerクエリを1-8個の統一トークンに適応的に圧縮
    タスク複雑度に応じて動的にトークン数を調整
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 設定
        self.qformer_dim = config.qformer_config['hidden_size']  # 768
        self.llama_dim = config.llama_hidden_size  # 5120
        self.max_compressed_tokens = 8
        self.min_compressed_tokens = 1
        
        # 学習可能な圧縮クエリ（最大8個）
        self.compression_queries = nn.Parameter(
            torch.randn(self.max_compressed_tokens, self.qformer_dim) * 0.02
        )
        
        # 複雑度予測器
        self.complexity_predictor = nn.Sequential(
            nn.Linear(self.qformer_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # クロスアテンション層（圧縮用）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.qformer_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 投影層（Q-Former -> Llama次元）
        self.to_llama_dim = nn.Sequential(
            nn.Linear(self.qformer_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, self.llama_dim),
            nn.LayerNorm(self.llama_dim)
        )
        
        # 2D位置エンコーディング
        self.spatial_pos_encoder = SinusoidalPositionalEncoding2D(
            d_model=self.qformer_dim,
            max_h=8,  # Q-Formerは通常32クエリ = 8x4グリッドと仮定
            max_w=4
        )
        
        # トークンタイプ埋め込み（圧縮トークン用）
        self.token_type_embedding = nn.Parameter(
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
    
    def predict_num_tokens(self, query_embeds: torch.Tensor) -> int:
        """
        クエリ埋め込みから必要なトークン数を予測
        
        Args:
            query_embeds: Q-Formerクエリ埋め込み (B, N, D)
        
        Returns:
            予測されたトークン数 (1-8)
        """
        # クエリの平均表現から複雑度を推定
        query_pooled = query_embeds.mean(dim=1)  # (B, D)
        complexity_score = self.complexity_predictor(query_pooled)  # (B, 1)
        
        # スコアをトークン数にマッピング（1-8）
        num_tokens = torch.clamp(
            (complexity_score * (self.max_compressed_tokens - 1) + 1).round().long(),
            min=self.min_compressed_tokens,
            max=self.max_compressed_tokens
        )
        
        # バッチ内で最大値を取る（統一処理のため）
        num_tokens = num_tokens.max().item()
        
        logger.debug(f"予測トークン数: {num_tokens} (複雑度スコア: {complexity_score.mean().item():.3f})")
        
        return num_tokens
    
    def compress_tokens(
        self,
        query_embeds: torch.Tensor,
        num_tokens: int,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        クロスアテンションによるトークン圧縮
        
        Args:
            query_embeds: Q-Formerクエリ埋め込み (B, N, D)
            num_tokens: 圧縮後のトークン数
            return_attention: アテンション重みを返すか
        
        Returns:
            compressed_tokens: 圧縮されたトークン (B, num_tokens, D)
            attention_weights: アテンション重み（オプション）
        """
        batch_size = query_embeds.size(0)
        device = query_embeds.device
        dtype = query_embeds.dtype
        
        # 使用する圧縮クエリを選択
        selected_queries = self.compression_queries[:num_tokens]  # (num_tokens, D)
        selected_queries = selected_queries.unsqueeze(0).expand(batch_size, -1, -1)
        selected_queries = selected_queries.to(device=device, dtype=dtype)
        
        # トークンタイプ埋め込みを追加
        selected_queries = selected_queries + self.token_type_embedding.to(device=device, dtype=dtype)
        
        # 位置エンコーディングを追加（空間情報保持）
        # Q-Formerの32クエリを8x4グリッドと仮定
        h, w = 8, 4
        query_embeds_with_pos = self.spatial_pos_encoder(query_embeds, h, w)
        
        # クロスアテンションで圧縮
        compressed_tokens, attention_weights = self.cross_attention(
            query=selected_queries,
            key=query_embeds_with_pos,
            value=query_embeds_with_pos,
            need_weights=return_attention
        )
        
        return compressed_tokens, attention_weights
    
    def forward(
        self,
        qformer_outputs: Dict[str, torch.Tensor],
        force_num_tokens: Optional[int] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Q-Former出力を適応的に圧縮
        
        Args:
            qformer_outputs: Q-Formerの出力辞書
                - 'query_embeds': (B, num_queries, qformer_dim)
            force_num_tokens: 強制的に使用するトークン数（デバッグ用）
            return_details: 詳細情報を返すか
        
        Returns:
            outputs: 圧縮結果
                - 'compressed_tokens': Llama次元の圧縮トークン (B, num_tokens, llama_dim)
                - 'num_tokens': 使用されたトークン数
                - 'compression_ratio': 圧縮率
                - 'attention_weights': アテンション重み（return_details=Trueの場合）
        """
        if 'query_embeds' not in qformer_outputs:
            raise ValueError("qformer_outputs must contain 'query_embeds'")
        
        query_embeds = qformer_outputs['query_embeds']
        batch_size, num_queries, _ = query_embeds.shape
        
        # トークン数の決定
        if force_num_tokens is not None:
            num_tokens = force_num_tokens
        else:
            num_tokens = self.predict_num_tokens(query_embeds)
        
        # トークン圧縮
        compressed_qformer, attention_weights = self.compress_tokens(
            query_embeds,
            num_tokens,
            return_attention=return_details
        )
        
        # Llama次元への投影
        compressed_llama = self.to_llama_dim(compressed_qformer)
        
        # 圧縮率の計算
        compression_ratio = num_queries / num_tokens
        
        # 出力辞書の構築
        outputs = {
            'compressed_tokens': compressed_llama,  # (B, num_tokens, llama_dim)
            'num_tokens': num_tokens,
            'compression_ratio': compression_ratio,
            'original_num_queries': num_queries,
        }
        
        if return_details:
            outputs.update({
                'compressed_qformer': compressed_qformer,  # 投影前のトークン
                'attention_weights': attention_weights,
                'complexity_score': self.complexity_predictor(query_embeds.mean(dim=1))
            })
        
        # ログ出力
        logger.info(f"トークン圧縮完了: {num_queries} → {num_tokens} "
                   f"(圧縮率: {compression_ratio:.1f}x)")
        
        return outputs


def create_adaptive_compressor(config) -> AdaptiveTokenCompressor:
    """
    適応的トークン圧縮器のファクトリ関数
    
    Args:
        config: モデル設定
    
    Returns:
        AdaptiveTokenCompressor インスタンス
    """
    logger.info("Creating AdaptiveTokenCompressor for Phase 2 implementation")
    return AdaptiveTokenCompressor(config)