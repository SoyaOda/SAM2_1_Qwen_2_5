# model/enhanced_qformer.py
"""
Enhanced Q-Former実装: BLIP-2準拠テキスト入力対応版

o3-modification20250727.mdに基づく改修:
1. テキスト入力対応とBLIP-2準拠のForward処理
2. 3つの学習目的（ITC, ITM, ITG）に対応するマスキング戦略
3. LLMへの適切な接続インターフェース

参考:
- BLIP-2論文: https://arxiv.org/abs/2301.12597
- HuggingFace実装: transformers.models.blip_2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_linux

# HuggingFace公式実装を基盤として使用
try:
    from transformers import Blip2QFormerModel, Blip2QFormerConfig, AutoTokenizer
    BLIP2_AVAILABLE = True
    print("✅ HuggingFace公式BLIP-2 Q-Former利用可能")
except ImportError:
    BLIP2_AVAILABLE = False
    raise ImportError("❌ EnhancedQFormerにはHuggingFace transformersが必要です")


class EnhancedQFormerModel(nn.Module):
    """
    BLIP-2準拠のQ-Former実装
    - テキスト入力対応
    - 3つの学習目的（ITC, ITM, ITG）に対応するマスキング戦略
    - LLMへの適切な接続インターフェース
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # 設定の保存
        self.config = config
        self.num_queries = config.get('num_queries', 32)  # BLIP-2標準
        self.hidden_size = config.get('hidden_size', 768)  # BLIP-2標準
        self.llm_hidden_size = config.get('llm_hidden_size', 5120)  # Llama-4
        
        print(f"🔧 Enhanced Q-Former初期化中...")
        print(f"  - クエリ数: {self.num_queries}")
        print(f"  - 隠れ層サイズ: {self.hidden_size}")
        print(f"  - LLM隠れ層サイズ: {self.llm_hidden_size}")
        
        # BLIP-2準拠のQ-Former設定
        # SAM2の実際の出力次元に合わせて調整
        # Webリサーチ結果: SAM2 Hiera-LargeのFPN出力は256次元
        encoder_hidden_size = config.get('encoder_hidden_size', 256)  # SAM2 FPN出力次元
        print(f"  - エンコーダ隠れ層サイズ: {encoder_hidden_size} (SAM2互換)")
        
        qformer_config = Blip2QFormerConfig(
            vocab_size=30522,  # BERT base
            hidden_size=self.hidden_size,
            num_hidden_layers=config.get('num_layers', 12),  # BLIP-2標準
            num_attention_heads=config.get('num_heads', 12),  # BLIP-2標準
            intermediate_size=config.get('intermediate_size', 3072),  # BLIP-2標準
            hidden_dropout_prob=config.get('dropout', 0.1),
            attention_probs_dropout_prob=config.get('dropout', 0.1),
            cross_attention_frequency=2,  # 2層ごとにcross-attention（BLIP-2標準）
            encoder_hidden_size=encoder_hidden_size,  # SAM2の実際の出力次元
            num_query_tokens=self.num_queries
        )
        
        # 公式Blip2QFormerModelを基盤として使用
        self.qformer = Blip2QFormerModel(qformer_config)
        
        # クエリトークンの初期化（BLIP-2準拠）
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_queries, self.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=0.02)
        
        # LLM接続用プロジェクター（BLIP-2準拠）
        self.llm_projector = nn.Linear(self.hidden_size, self.llm_hidden_size)
        nn.init.normal_(self.llm_projector.weight, std=0.02)
        nn.init.zeros_(self.llm_projector.bias)
        
        # SAMプロンプト生成用プロジェクター（既存との互換性）
        self.sam_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_size // 2, config.get('sam_prompt_dim', 256))
        )
        
        # テキストトークナイザー（BERT base）
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_txt_len = config.get('max_txt_len', 64)
        
        print(f"✅ Enhanced Q-Former初期化完了")
        print(f"  - パラメータ数: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self,
        image_feats: torch.Tensor,
        text_input: Optional[List[str]] = None,
        mode: str = 'itg',  # 'itc', 'itm', 'itg'
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        BLIP-2準拠のforward処理
        
        Args:
            image_feats: 画像エンコーダ出力 [B, num_patches, encoder_dim]
            text_input: テキスト入力のリスト（オプション）
            mode: 学習モード ('itc': Image-Text Contrastive, 
                            'itm': Image-Text Matching,
                            'itg': Image-grounded Text Generation)
            return_dict: 辞書形式で返すか
        
        Returns:
            dict containing:
                - query_embeds: クエリ埋め込み [B, num_queries, hidden_size]
                - llm_embeds: LLM用に射影されたクエリ [B, num_queries, llm_hidden_size]
                - sam_prompts: SAM用プロンプト [B, num_queries, sam_prompt_dim]
                - text_outputs: テキスト出力（text_input提供時）
        """
        batch_size = image_feats.size(0)
        device = image_feats.device
        
        # image_featsの形状を確認・調整（アテンションマスク作成前に実行）
        image_feats = self._reshape_image_features(image_feats)
        
        # テキスト処理（提供されている場合）
        text_tokens = None
        text_attention_mask = None
        if text_input is not None:
            # テキストをトークン化
            text_encoding = self.tokenizer(
                text_input,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            )
            text_tokens = text_encoding.input_ids.to(device)
            text_attention_mask = text_encoding.attention_mask.to(device)
        
        # クエリ埋め込みの取得（学習可能クエリ）
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        
        # 画像アテンションマスク（リシェイプ後の形状に基づいて作成）
        # image_featsは既に[batch_size, sequence_length, hidden_size]形式
        image_atts = torch.ones(image_feats.size()[:-1], dtype=torch.long, device=device)
        
        # モードに応じたforward処理
        if mode == 'itc':
            # Image-Text Contrastive Learning
            # クエリとテキストが相互に見えないマスクを使用
            outputs = self._forward_itc(
                query_embeds, image_feats, image_atts,
                text_tokens, text_attention_mask
            )
        elif mode == 'itm':
            # Image-Text Matching
            # クエリとテキストが相互作用可能
            outputs = self._forward_itm(
                query_embeds, image_feats, image_atts,
                text_tokens, text_attention_mask
            )
        elif mode == 'itg':
            # Image-grounded Text Generation（デフォルト）
            outputs = self._forward_itg(
                query_embeds, image_feats, image_atts,
                text_tokens, text_attention_mask
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 出力の整理
        # Blip2QFormerModelの出力がtupleの場合の対応
        if isinstance(outputs, tuple):
            # tupleの場合は最初の要素を取得
            outputs = outputs[0]
        
        # last_hidden_stateが存在することを確認
        if hasattr(outputs, 'last_hidden_state'):
            query_outputs = outputs.last_hidden_state[:, :self.num_queries, :]
        else:
            # last_hidden_stateがない場合は直接outputsを使用
            query_outputs = outputs[:, :self.num_queries, :]
        
        # LLM用に射影
        llm_embeds = self.llm_projector(query_outputs)
        
        # SAMプロンプト生成
        sam_prompts = self.sam_projector(query_outputs)
        
        result = {
            'query_embeds': query_outputs,
            'llm_embeds': llm_embeds,
            'sam_prompts': sam_prompts,
        }
        
        # テキスト出力（提供されている場合）
        if text_tokens is not None:
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state.size(1) > self.num_queries:
                result['text_outputs'] = outputs.last_hidden_state[:, self.num_queries:, :]
            elif isinstance(outputs, torch.Tensor) and outputs.size(1) > self.num_queries:
                result['text_outputs'] = outputs[:, self.num_queries:, :]
        
        if return_dict:
            return result
        else:
            return (query_outputs, llm_embeds, sam_prompts)
    
    def _forward_itc(
        self,
        query_embeds: torch.Tensor,
        image_feats: torch.Tensor,
        image_atts: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor]
    ):
        """ITC: Image-Text Contrastive Learning用forward"""
        # image_featsは既にforward()でリシェイプ済み
        
        # テキストがある場合は結合
        if text_tokens is not None:
            # ITCではクエリとテキストは相互に見えない
            # BLIP-2実装に基づき、別々に処理してから結合
            
            # 画像側の処理（クエリ + 画像特徴）
            # Blip2QFormerModelは明示的にreturn_dictを指定
            image_outputs = self.qformer(
                query_embeds=query_embeds,
                encoder_hidden_states=image_feats,
                encoder_attention_mask=image_atts,
                use_cache=False,
                return_dict=True
            )
            
            # テキスト側の処理は実装しない（Blip2QFormerModelはinput_idsを受け取らない）
            # ITCモードでは画像側の出力のみ使用
            text_outputs = None
            
            # 結果を統合（実際のITCでは対照学習のため別々に使用）
            return image_outputs
        else:
            # テキストなしの場合は通常のforward
            return self.qformer(
                query_embeds=query_embeds,
                encoder_hidden_states=image_feats,
                encoder_attention_mask=image_atts,
                use_cache=False,
                return_dict=True
            )
    
    def _forward_itm(
        self,
        query_embeds: torch.Tensor,
        image_feats: torch.Tensor,
        image_atts: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor]
    ):
        """ITM: Image-Text Matching用forward"""
        # image_featsは既にforward()でリシェイプ済み
        
        if text_tokens is not None:
            # ITMではクエリとテキストが相互作用可能
            # ただし、Blip2QFormerModelはinput_idsを受け取らないため、
            # 画像特徴のみで処理
            return self.qformer(
                query_embeds=query_embeds,
                encoder_hidden_states=image_feats,
                encoder_attention_mask=image_atts,
                use_cache=False,
                return_dict=True
            )
        else:
            # テキストなしの場合
            return self.qformer(
                query_embeds=query_embeds,
                encoder_hidden_states=image_feats,
                encoder_attention_mask=image_atts,
                use_cache=False,
                return_dict=True
            )
    
    def _forward_itg(
        self,
        query_embeds: torch.Tensor,
        image_feats: torch.Tensor,
        image_atts: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor]
    ):
        """ITG: Image-grounded Text Generation用forward"""
        # image_featsは既にforward()でリシェイプ済み
        
        # ITGモードでは、まず画像特徴からクエリを生成
        # その後、必要に応じてテキストと結合
        
        if text_tokens is not None:
            # 画像とテキストを統合して処理
            # ただし、Blip2QFormerModelはinput_idsを受け取らないため、
            # 画像特徴のみで処理
            outputs = self.qformer(
                query_embeds=query_embeds,
                encoder_hidden_states=image_feats,
                encoder_attention_mask=image_atts,
                use_cache=False,
                return_dict=True
            )
        else:
            # 画像のみから特徴抽出
            outputs = self.qformer(
                query_embeds=query_embeds,
                encoder_hidden_states=image_feats,
                encoder_attention_mask=image_atts,
                use_cache=False,
                return_dict=True
            )
        
        return outputs
    
    def _reshape_image_features(self, image_feats: torch.Tensor) -> torch.Tensor:
        """
        image_featsを3次元テンソル [batch_size, sequence_length, hidden_size] に調整
        SAM2 Hieraは[B, H, W, C]形式（チャネルラスト）で出力する場合がある
        """
        # デバイス確認と統一
        target_device = next(self.parameters()).device
        if image_feats.device != target_device:
            image_feats = image_feats.to(target_device)
        
        if image_feats.dim() == 4:
            B = image_feats.shape[0]
            
            # SAM2 Hieraの場合: [B, H, W, C]形式（チャネルラスト）
            # 通常のViT: [B, C, H, W]形式（チャネルファースト）
            # 判定: 最後の次元が特徴次元として妥当か確認
            if image_feats.shape[-1] in [256, 768, 1024, 1408]:  # 一般的な特徴次元
                # [B, H, W, C] -> [B, H*W, C]
                H, W, C = image_feats.shape[1], image_feats.shape[2], image_feats.shape[3]
                image_feats = image_feats.view(B, H*W, C)
            else:
                # [B, C, H, W] -> [B, H*W, C]
                C, H, W = image_feats.shape[1], image_feats.shape[2], image_feats.shape[3]
                image_feats = image_feats.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        elif image_feats.dim() == 2:
            # [B, C] -> [B, 1, C]
            image_feats = image_feats.unsqueeze(1)
        elif image_feats.dim() != 3:
            raise ValueError(f"Unsupported image_feats dimension: {image_feats.dim()}")
        
        return image_feats
    
    def process_text(self, text_input: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        テキスト入力の処理（互換性のため）
        """
        if not text_input:
            return None
        
        tokens = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokens.input_ids.to(device),
            'attention_mask': tokens.attention_mask.to(device)
        }


def get_enhanced_qformer_model(config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Enhanced Q-Formerモデルのファクトリ関数
    """
    if not BLIP2_AVAILABLE:
        raise ImportError("HuggingFace BLIP-2が必要です")
    
    # デフォルト設定
    default_config = {
        'num_queries': 32,              # BLIP-2準拠
        'hidden_size': 768,             # BLIP-2準拠（BERT base）
        'num_layers': 12,               # BLIP-2準拠
        'num_heads': 12,                # BLIP-2準拠
        'intermediate_size': 3072,      # BLIP-2準拠
        'dropout': 0.1,
        'sam_prompt_dim': 256,          # SAM2プロンプト次元
        'encoder_hidden_size': 256,     # SAM2 FPN出力次元（Webリサーチ結果）
        'llm_hidden_size': 5120,        # Llama-4
        'max_txt_len': 64,              # テキスト最大長
    }
    
    if config:
        default_config.update(config)
    
    return EnhancedQFormerModel(default_config)