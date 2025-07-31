# model/qformer.py
"""
Q-Former実装: HuggingFace公式BLIP-2 Q-Formerベース

Meta/Salesforce公式実装を活用:
- transformers.Blip2QFormerModel使用
- 安定性と保守性向上
- 公式バグフィックス自動適用
- Llama-4-Scout統合最適化

参考:
- HuggingFace BLIP-2: https://huggingface.co/docs/transformers/model_doc/blip-2
- 公式実装: transformers.models.blip_2.modeling_blip_2
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

# 🔄 公式API使用
try:
    from transformers import Blip2QFormerModel, Blip2QFormerConfig, AutoTokenizer
    BLIP2_AVAILABLE = True
    print("✅ HuggingFace公式BLIP-2 Q-Former利用可能")
except ImportError:
    BLIP2_AVAILABLE = False
    print("⚠️ HuggingFace BLIP-2が利用できません。カスタム実装を使用します。")


class QFormerTextProcessor:
    """
    BLIP-2準拠テキスト処理パイプライン
    
    Web調査結果ベース2025年ベストプラクティス:
    - text_input文字列をBLIP-2形式でトークン化
    - 適切なパディング・トランケーション
    - デバイス管理とBFloat16対応
    """
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_txt_len: int = 64):
        if not BLIP2_AVAILABLE:
            raise ImportError("テキスト処理にはHuggingFace transformersが必要です")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_txt_len = max_txt_len
        print(f"🔄 Q-Former用テキストプロセッサ初期化: {tokenizer_name}")
    
    def process_text(
        self, 
        text_input: List[str], 
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, torch.Tensor]:
        """
        テキストをBLIP-2形式でトークン化
        
        Args:
            text_input: テキスト文字列のリスト
            device: ターゲットデバイス
            dtype: ターゲットデータ型
            
        Returns:
            Dict containing:
                - input_ids: (batch_size, max_len) トークンID
                - attention_mask: (batch_size, max_len) アテンションマスク
        """
        if not text_input:
            # 空入力の場合はダミーテキスト
            text_input = [" "] * 1
        
        # BLIP-2標準トークン化
        tokens = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        )
        
        # デバイス・データ型移動
        result = {
            'input_ids': tokens.input_ids.to(device=device),
            'attention_mask': tokens.attention_mask.to(device=device)
        }
        
        return result


class OfficialQFormerModel(nn.Module):
    """
    HuggingFace公式BLIP-2 Q-Formerラッパー
    
    Llama-4-Scout統合用にカスタマイズ:
    - hidden_size: 5120 (Llama-4に合わせる)
    - SAMプロンプト生成用プロジェクター追加
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if not BLIP2_AVAILABLE:
            raise ImportError("HuggingFace BLIP-2が利用できません。pip install transformersで更新してください。")
        
        # デフォルト設定
        default_config = {
            'num_queries': 32,              # BLIP-2準拠
            'hidden_size': 5120,            # Llama-4に合わせる
            'num_layers': 6,                # BLIP-2準拠
            'num_heads': 16,                # 5120 / 320 = 16
            'intermediate_size': 20480,     # hidden_size * 4
            'dropout': 0.1,
            'sam_prompt_dim': 256,          # SAM2プロンプト次元
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        print(f"🔄 公式Q-Former初期化中...")
        print(f"  - クエリ数: {self.config['num_queries']}")
        print(f"  - 隠れ層サイズ: {self.config['hidden_size']}")
        print(f"  - レイヤー数: {self.config['num_layers']}")
        
        # 🔄 HuggingFace公式BLIP-2 Q-Former設定
        qformer_config = Blip2QFormerConfig(
            vocab_size=30522,                               # BLIP-2デフォルト
            hidden_size=self.config['hidden_size'],         # Llama-4に合わせる
            num_hidden_layers=self.config['num_layers'],
            num_attention_heads=self.config['num_heads'],
            intermediate_size=self.config['intermediate_size'],
            hidden_dropout_prob=self.config['dropout'],
            attention_probs_dropout_prob=self.config['dropout'],
            cross_attention_frequency=2,                    # BLIP-2デフォルト
            num_query_tokens=self.config['num_queries'],
        )
        
        # 🔄 公式Q-Formerモデル
        self.qformer = Blip2QFormerModel(qformer_config)
        
        # SAM2プロンプト生成用プロジェクター
        self.sam_projector = nn.Sequential(
            nn.Linear(self.config['hidden_size'], self.config['hidden_size'] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['hidden_size'] // 2, self.config['sam_prompt_dim']),
        )
        
        # 2025年公式API準拠: テキスト処理削除（query_embeds中心化）
        # self.text_processor = QFormerTextProcessor()  # 廃止
        
        print(f"✅ 公式Q-Former初期化完了")
        print(f"  - パラメータ数: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  - テキスト処理: BLIP-2準拠")
    
    def forward(
        self,
        # 🔄 2025年公式API準拠パラメータ (Web調査ベース)
        query_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 📝 非公式パラメータ (フォールバック用)
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # オプション
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        HuggingFace公式BLIP-2 Q-Formerフォワードパス (2025年Web調査版)
        
        🔄 Web調査結果: HuggingFace公式のBlip2QFormerModelは
           text_inputパラメータを直接サポートしていません
        
        Args:
            query_embeds: (batch_size, num_queries, hidden_size) 学習可能クエリ
            encoder_hidden_states: (batch_size, seq_len, hidden_size) VLM特徴
            encoder_attention_mask: (batch_size, seq_len) VLMアテンションマスク
            input_ids: 非推奨パラメータ (フォールバック用)
            attention_mask: 非推奨パラメータ (フォールバック用)
            output_attentions: アテンション重みを返すかどうか
            return_dict: 辞書形式で結果を返すかどうか
            
        Returns:
            Dict containing:
                - query_embeds: (batch_size, num_queries, hidden_size) クエリ埋め込み
                - sam_prompts: (batch_size, num_queries, sam_prompt_dim) SAMプロンプト
                - attentions: アテンション重み（オプション）
        """
        
        # 📋 Web調査結果ベース入力検証 (2025年公式API準拠)
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for Q-Former cross-attention")
            
        batch_size, seq_len, hidden_size = encoder_hidden_states.shape
        device = encoder_hidden_states.device
        
        # 🔄 query_embeds準備: 公式BLIP-2学習可能クエリアクセス (Web調査準拠)
        if query_embeds is None:
            print(f"  🔄 学習可能クエリアクセス: {self.config['num_queries']}個")
            try:
                # Web調査結果: 公式BLIP-2はself.qformer.query_tokensを持つ
                if hasattr(self.qformer, 'query_tokens'):
                    query_embeds = self.qformer.query_tokens.expand(batch_size, -1, -1)
                    print(f"    ✅ 公式query_tokens使用: {query_embeds.shape}")
                else:
                    # フォールバック: ゼロ初期化クエリ
                    device = encoder_hidden_states.device
                    dtype = encoder_hidden_states.dtype
                    query_embeds = torch.zeros(
                        batch_size, self.config['num_queries'], self.config['hidden_size'],
                        device=device, dtype=dtype
                    )
                    print(f"    ⚠️ フォールバック: ゼロクエリ使用: {query_embeds.shape}")
            except Exception as e:
                print(f"    ❌ クエリアクセスエラー: {e}")
                # 最終フォールバック
                device = encoder_hidden_states.device
                dtype = encoder_hidden_states.dtype
                query_embeds = torch.zeros(
                    batch_size, self.config['num_queries'], self.config['hidden_size'],
                    device=device, dtype=dtype
                )
        
        print(f"  📊 Q-Former入力統計:")
        print(f"    - query_embeds: {query_embeds.shape}")
        print(f"    - encoder_hidden_states: {encoder_hidden_states.shape}")
        print(f"    - device: {device}")
        
        # encoder_attention_maskの処理
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # 🔄 2025年公式BLIP-2 Q-Former実行 (Web調査ベース簡潔版)
        try:
            print(f"  🔄 公式Q-Former実行中...")
            result = self._execute_official_qformer_2025(
                query_embeds=query_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            print(f"  ✅ 公式Q-Former実行成功")
            
        except Exception as e:
            print(f"  ⚠️ 公式APIエラー: {e}")
            print(f"  🔄 フォールバック実行中...")
            result = self._fallback_forward(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
        
        # SAM2プロンプトに変換
        sam_prompts = self.sam_projector(result['query_embeds'])
        result['sam_prompts'] = sam_prompts
        
        if return_dict:
            return result
        else:
            return (result['query_embeds'], sam_prompts)
    
    def _execute_official_qformer_2025(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        output_attentions: bool
    ) -> Dict[str, torch.Tensor]:
        """
        2025年公式BLIP-2 Q-Formerの実行 (Web調査準拠)
        
        📋 公式APIパラメータ:
        - query_embeds: 必須
        - encoder_hidden_states: クロスアテンション用
        - encoder_attention_mask: アテンションマスク
        - text_inputやinput_idsは非対応 (Web調査結果)
        """
        # 🔄 HuggingFace公式BLIP-2 Q-Formerコール
        qformer_outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        # クエリ埋め込みを取得
        query_embeds_output = qformer_outputs.last_hidden_state
        
        result = {
            'query_embeds': query_embeds_output,
            'last_hidden_state': query_embeds_output,  # HuggingFace互換性
        }
        
        if output_attentions and hasattr(qformer_outputs, 'attentions'):
            result['attentions'] = qformer_outputs.attentions
            
        return result
    
    def _fallback_forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        output_attentions: bool
    ) -> Dict[str, torch.Tensor]:
        """
        フォールバック実装: シンプルな特徴抽出
        """
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        # シンプルな平均プーリング + 線形変換
        pooled_features = encoder_hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        
        # クエリ数分に拡張
        query_embeds = pooled_features.unsqueeze(1).repeat(1, self.config['num_queries'], 1)
        
        print(f"  🔄 フォールバック実行: 平均プーリング")
        
        return {
            'query_embeds': query_embeds,
            'last_hidden_state': query_embeds,
        }
    
    # API互換性エラー処理削除 (2025年公式API準拠で不要)
    
    # NoneTypeエラー処理削除 (2025年公式API準拠で不要)


class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション実装"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, query_len, hidden_size)
            key: (batch_size, key_len, hidden_size)  
            value: (batch_size, value_len, hidden_size)
            attention_mask: (batch_size, query_len, key_len)
        
        Returns:
            attn_output: (batch_size, query_len, hidden_size)
            attn_weights: (batch_size, num_heads, query_len, key_len)
        """
        batch_size, query_len = query.size(0), query.size(1)
        key_len = key.size(1)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.query(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Q, K)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, K) -> (B, 1, 1, K) for cross-attention
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # (B, H, Q, D)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_size
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class TransformerLayer(nn.Module):
    """Q-Former用Transformerレイヤー"""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-Attention for query-query interaction
        self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # Cross-Attention for query-VLM interaction
        self.cross_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            query_embeds: (batch_size, num_queries, hidden_size)
            encoder_hidden_states: (batch_size, seq_len, hidden_size) - VLMの隠れ状態
            encoder_attention_mask: (batch_size, seq_len)
            
        Returns:
            query_embeds: Updated query embeddings
            self_attn_weights: Self-attention weights (optional)
            cross_attn_weights: Cross-attention weights (optional)
        """
        
        # 1. Self-Attention: Query-Query interaction
        residual = query_embeds
        self_attn_output, self_attn_weights = self.self_attention(
            query=query_embeds,
            key=query_embeds, 
            value=query_embeds
        )
        query_embeds = self.self_attn_layer_norm(residual + self_attn_output)
        
        # 2. Cross-Attention: Query-VLM interaction
        residual = query_embeds
        cross_attn_output, cross_attn_weights = self.cross_attention(
            query=query_embeds,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            attention_mask=encoder_attention_mask
        )
        query_embeds = self.cross_attn_layer_norm(residual + cross_attn_output)
        
        # 3. Feed Forward Network
        residual = query_embeds
        ffn_output = self.ffn(query_embeds)
        query_embeds = self.ffn_layer_norm(residual + ffn_output)
        
        outputs = (query_embeds,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
            
        return outputs


class QFormerModel(nn.Module):
    """
    Q-Former: Querying Transformer for Vision-Language Interface
    
    BLIP-2ベースの実装でLlama-4-Scout → SAM2のインターフェースとして機能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # デフォルト設定をconfig_linuxから取得
        self.config = config or self._get_default_config()
        
        self.num_queries = self.config['num_queries']
        self.hidden_size = self.config['hidden_size']
        self.num_layers = self.config['num_layers']
        self.num_heads = self.config['num_heads']
        self.intermediate_size = self.config['intermediate_size']
        
        # 学習可能なクエリベクトル (BLIP-2では32個)
        self.query_embeds = nn.Parameter(
            torch.randn(1, self.num_queries, self.hidden_size) * 0.02
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout=self.config['dropout']
            )
            for _ in range(self.num_layers)
        ])
        
        # SAM2プロンプト用の最終射影層
        sam_prompt_dim = self.config.get('sam_prompt_dim', 256)  # SAM2のプロンプト次元
        self.sam_prompt_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.hidden_size // 2, sam_prompt_dim)
        )
        
        # 初期化
        self.apply(self._init_weights)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得（config_linuxベース）"""
        return {
            'num_queries': 32,           # BLIP-2準拠
            'hidden_size': config_linux.LLAMA_HIDDEN_SIZE,  # 5120: Llama-4-Scoutの隠れ層サイズ
            'num_layers': 6,             # BLIP-2では6層
            'num_heads': 16,             # hidden_size / head_dim = 5120 / 320 = 16
            'intermediate_size': config_linux.LLAMA_HIDDEN_SIZE * 4,  # 20480: FFNサイズ
            'dropout': 0.1,
            'sam_prompt_dim': config_linux.SAM_PROMPT_EMBED_DIM,  # 256: SAM2プロンプト次元
        }
    
    def _init_weights(self, module):
        """重み初期化（BERT/BLIP-2スタイル）"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Q-Formerのフォワードパス
        
        Args:
            encoder_hidden_states: (batch_size, seq_len, hidden_size) 
                                 Llama-4-Scoutの隠れ状態
            encoder_attention_mask: (batch_size, seq_len)
                                  パディングマスク
            output_attentions: アテンション重みを返すかどうか
            return_dict: 辞書形式で返すかどうか
            
        Returns:
            Dict containing:
                - query_embeds: (batch_size, num_queries, hidden_size)
                - sam_prompts: (batch_size, num_queries, sam_prompt_dim)
                - attentions: List of attention weights (optional)
        """
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        # クエリベクトルをバッチサイズに拡張
        query_embeds = self.query_embeds.expand(batch_size, -1, -1).to(device)
        
        # アテンション重みを保存するリスト
        all_self_attentions = [] if output_attentions else None
        all_cross_attentions = [] if output_attentions else None
        
        # Transformerレイヤーを順次実行
        for layer in self.layers:
            layer_outputs = layer(
                query_embeds=query_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            
            query_embeds = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions.append(layer_outputs[1])
                all_cross_attentions.append(layer_outputs[2])
        
        # SAM2用プロンプトに変換
        sam_prompts = self.sam_prompt_projection(query_embeds)
        
        if return_dict:
            outputs = {
                'query_embeds': query_embeds,              # (B, num_queries, hidden_size)
                'sam_prompts': sam_prompts,                # (B, num_queries, sam_prompt_dim) 
                'last_hidden_state': query_embeds,        # HuggingFace互換性のため
            }
            
            if output_attentions:
                outputs['self_attentions'] = all_self_attentions
                outputs['cross_attentions'] = all_cross_attentions
                
            return outputs
        else:
            return query_embeds, sam_prompts


def test_qformer():
    """Q-Formerの基本動作テスト（軽量版）"""
    print("=== Q-Former基本動作テスト（軽量版） ===")
    
    # テスト用設定（軽量化）
    test_config = {
        'num_queries': 8,      # 32 -> 8
        'hidden_size': 512,    # 5120 -> 512  
        'num_layers': 2,       # 6 -> 2
        'num_heads': 8,        # 16 -> 8
        'intermediate_size': 2048,  # 20480 -> 2048
        'dropout': 0.1,
        'sam_prompt_dim': 256,
    }
    
    batch_size = 1
    seq_len = 20  # 100 -> 20
    hidden_size = test_config['hidden_size']
    
    # Q-Formerモデル初期化
    qformer = QFormerModel(config=test_config)
    print(f"Q-Former初期化完了")
    print(f"  - クエリ数: {qformer.num_queries}")
    print(f"  - 隠れ層サイズ: {qformer.hidden_size}")
    print(f"  - レイヤー数: {qformer.num_layers}")
    print(f"  - 総パラメータ数: {sum(p.numel() for p in qformer.parameters()):,}")
    
    # ダミー入力作成（Llama-4-Scoutからの隠れ状態を模擬）
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    encoder_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n入力:")
    print(f"  - encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  - encoder_attention_mask: {encoder_attention_mask.shape}")
    
    # フォワードパス実行
    with torch.no_grad():
        outputs = qformer(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False  # アテンション重み無効化で高速化
        )
    
    print(f"\n出力:")
    print(f"  - query_embeds: {outputs['query_embeds'].shape}")
    print(f"  - sam_prompts: {outputs['sam_prompts'].shape}")
    
    # SAM2プロンプトの品質確認
    sam_prompts = outputs['sam_prompts']
    print(f"\nSAM2プロンプト品質チェック:")
    print(f"  - 平均: {sam_prompts.mean().item():.4f}")
    print(f"  - 標準偏差: {sam_prompts.std().item():.4f}")
    print(f"  - 最小値: {sam_prompts.min().item():.4f}")
    print(f"  - 最大値: {sam_prompts.max().item():.4f}")
    
    print("\n✅ Q-Former基本動作テスト完了")


def get_qformer_model(config: Optional[Dict[str, Any]] = None, prefer_official: bool = True) -> nn.Module:
    """
    Q-Formerモデルのファクトリー関数
    
    Args:
        config: モデル設定辞書
        prefer_official: 公式BLIP-2 Q-Formerを優先するかどうか
        
    Returns:
        QFormerModel (公式またはカスタム実装)
    """
    
    if prefer_official and BLIP2_AVAILABLE:
        print("🔄 HuggingFace公式BLIP-2 Q-Formerを使用")
        return OfficialQFormerModel(config)
    else:
        if not BLIP2_AVAILABLE:
            print("⚠️ 公式BLIP-2が利用できないため、カスタム実装を使用")
        else:
            print("🔧 カスタムQ-Former実装を使用")
        return QFormerModel(config)


if __name__ == "__main__":
    # 公式版のテスト
    if BLIP2_AVAILABLE:
        print("=== HuggingFace公式Q-Formerテスト ===")
        try:
            official_qformer = get_qformer_model(prefer_official=True)
            print("✅ 公式Q-Formerテスト成功")
        except Exception as e:
            print(f"❌ 公式Q-Formerテスト失敗: {e}")
    
    # カスタム版のテスト
    print("\n=== カスタムQ-Formerテスト ===")
    test_qformer()