# model/modality_separation.py
"""
視覚・言語特徴分離機構

o3-modification20250727.mdに基づく実装:
- Llama-4の早期融合出力から視覚・言語特徴を分離
- モダリティ別トークン管理
- デュアルヘッド出力構造
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import math


class ModalityAwareTokenizer:
    """
    モダリティ別トークン管理
    
    視覚トークンとテキストトークンを区別し、
    Llama-4の早期融合アーキテクチャで適切に処理できるよう管理
    """
    
    # 特殊トークン定義
    VISUAL_TOKEN_START = "<VISUAL_START>"
    VISUAL_TOKEN_END = "<VISUAL_END>"
    TEXT_TOKEN_START = "<TEXT_START>"
    TEXT_TOKEN_END = "<TEXT_END>"
    
    def __init__(self, llm_tokenizer=None):
        """
        Args:
            llm_tokenizer: LLM用のトークナイザー（特殊トークン追加用）
        """
        self.llm_tokenizer = llm_tokenizer
        
        # 特殊トークンをトークナイザーに追加
        if self.llm_tokenizer is not None:
            special_tokens = {
                'additional_special_tokens': [
                    self.VISUAL_TOKEN_START,
                    self.VISUAL_TOKEN_END,
                    self.TEXT_TOKEN_START,
                    self.TEXT_TOKEN_END
                ]
            }
            num_added = self.llm_tokenizer.add_special_tokens(special_tokens)
            print(f"✅ モダリティ特殊トークン追加: {num_added}個")
            
            # 特殊トークンIDの取得
            self.visual_start_id = self.llm_tokenizer.convert_tokens_to_ids(self.VISUAL_TOKEN_START)
            self.visual_end_id = self.llm_tokenizer.convert_tokens_to_ids(self.VISUAL_TOKEN_END)
            self.text_start_id = self.llm_tokenizer.convert_tokens_to_ids(self.TEXT_TOKEN_START)
            self.text_end_id = self.llm_tokenizer.convert_tokens_to_ids(self.TEXT_TOKEN_END)
    
    def prepare_multimodal_input(
        self,
        visual_embeds: torch.Tensor,      # [B, num_queries, llm_dim]
        text_tokens: Optional[torch.Tensor] = None,  # [B, text_len]
        text_embeds: Optional[torch.Tensor] = None,  # [B, text_len, llm_dim]
        llm_embed_layer: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        視覚・テキストトークンを統合し、種別情報を付与
        
        Args:
            visual_embeds: Q-Formerから得た視覚埋め込み
            text_tokens: テキストトークンID（text_embedsがない場合）
            text_embeds: テキスト埋め込み（直接提供される場合）
            llm_embed_layer: LLMの埋め込み層（text_tokensから埋め込みを生成用）
        
        Returns:
            Dict containing:
                - input_embeds: 結合された埋め込み [B, total_len, llm_dim]
                - attention_mask: アテンションマスク [B, total_len]
                - token_type_ids: トークン種別 (0: text, 1: visual)
                - visual_token_positions: 視覚トークンの位置リスト
                - text_token_positions: テキストトークンの位置リスト
        """
        batch_size = visual_embeds.size(0)
        device = visual_embeds.device
        dtype = visual_embeds.dtype
        
        # 視覚トークン数
        num_visual_tokens = visual_embeds.size(1)
        
        # 特殊トークンの埋め込みを作成
        if llm_embed_layer is not None and self.llm_tokenizer is not None:
            # 特殊トークンIDから埋め込みを生成
            special_token_ids = torch.tensor([
                self.visual_start_id,
                self.visual_end_id,
                self.text_start_id,
                self.text_end_id
            ], device=device)
            special_embeds = llm_embed_layer(special_token_ids)  # [4, llm_dim]
            
            visual_start_embed = special_embeds[0:1].expand(batch_size, 1, -1)
            visual_end_embed = special_embeds[1:2].expand(batch_size, 1, -1)
            text_start_embed = special_embeds[2:3].expand(batch_size, 1, -1)
            text_end_embed = special_embeds[3:4].expand(batch_size, 1, -1)
        else:
            # フォールバック: ゼロ埋め込み
            embed_dim = visual_embeds.size(-1)
            visual_start_embed = torch.zeros(batch_size, 1, embed_dim, device=device, dtype=dtype)
            visual_end_embed = torch.zeros(batch_size, 1, embed_dim, device=device, dtype=dtype)
            text_start_embed = torch.zeros(batch_size, 1, embed_dim, device=device, dtype=dtype)
            text_end_embed = torch.zeros(batch_size, 1, embed_dim, device=device, dtype=dtype)
        
        # 視覚部分の構築: <VISUAL_START> + visual_embeds + <VISUAL_END>
        visual_section = torch.cat([
            visual_start_embed,
            visual_embeds,
            visual_end_embed
        ], dim=1)  # [B, num_visual_tokens + 2, llm_dim]
        
        # 位置情報の記録
        visual_token_positions = list(range(1, num_visual_tokens + 1))  # 特殊トークンを除く
        
        # テキスト処理
        if text_embeds is not None:
            # テキスト埋め込みが直接提供されている場合
            num_text_tokens = text_embeds.size(1)
            text_section = torch.cat([
                text_start_embed,
                text_embeds,
                text_end_embed
            ], dim=1)
        elif text_tokens is not None and llm_embed_layer is not None:
            # トークンIDから埋め込みを生成
            text_embeds = llm_embed_layer(text_tokens)
            num_text_tokens = text_embeds.size(1)
            text_section = torch.cat([
                text_start_embed,
                text_embeds,
                text_end_embed
            ], dim=1)
        else:
            # テキストなし
            num_text_tokens = 0
            text_section = None
        
        # 全体の結合
        if text_section is not None:
            input_embeds = torch.cat([visual_section, text_section], dim=1)
            
            # テキストトークン位置（視覚セクションの後）
            text_start_pos = visual_section.size(1) + 1  # +1 for TEXT_START
            text_token_positions = list(range(text_start_pos, text_start_pos + num_text_tokens))
            
            # トークン種別ID
            visual_type_ids = torch.ones(batch_size, visual_section.size(1), device=device, dtype=torch.long)
            text_type_ids = torch.zeros(batch_size, text_section.size(1), device=device, dtype=torch.long)
            token_type_ids = torch.cat([visual_type_ids, text_type_ids], dim=1)
            
            # アテンションマスク（全て有効）
            attention_mask = torch.ones(batch_size, input_embeds.size(1), device=device, dtype=torch.long)
        else:
            # 視覚のみ
            input_embeds = visual_section
            text_token_positions = []
            token_type_ids = torch.ones(batch_size, visual_section.size(1), device=device, dtype=torch.long)
            attention_mask = torch.ones(batch_size, input_embeds.size(1), device=device, dtype=torch.long)
        
        return {
            'inputs_embeds': input_embeds,  # LLMへの入力用
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'visual_token_positions': visual_token_positions,
            'text_token_positions': text_token_positions,
            'num_visual_tokens': num_visual_tokens,
            'num_text_tokens': num_text_tokens
        }


class DualModalityOutput(nn.Module):
    """
    視覚・言語の分離出力機構
    
    Llama-4の統合出力から視覚・言語特徴を分離し、
    それぞれ専用のヘッドで処理
    """
    
    def __init__(
        self,
        llm_hidden_size: int,
        vocab_size: int,
        vision_output_dim: int = 1024,  # SAM2互換
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.llm_hidden_size = llm_hidden_size
        self.vocab_size = vocab_size
        self.vision_output_dim = vision_output_dim
        
        # 言語出力ヘッド（標準のLMヘッド）
        self.language_head = nn.Linear(llm_hidden_size, vocab_size, bias=False)
        
        # 視覚出力ヘッド（セグメンテーション用）
        self.vision_head = nn.Sequential(
            nn.Linear(llm_hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, vision_output_dim),
            nn.LayerNorm(vision_output_dim)
        )
        
        # 視覚特徴の集約方法
        self.vision_pooler = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.Tanh()
        )
        
        print(f"✅ DualModalityOutput初期化完了")
        print(f"  - LLM隠れ層: {llm_hidden_size}")
        print(f"  - 語彙サイズ: {vocab_size}")
        print(f"  - 視覚出力次元: {vision_output_dim}")
    
    def forward(
        self,
        llm_outputs: torch.Tensor,  # [B, seq_len, llm_hidden_size]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, seq_len]
        visual_positions: Optional[List[int]] = None,
        text_positions: Optional[List[int]] = None,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        LLM出力を視覚・言語に分離
        
        Args:
            llm_outputs: LLMの最終層出力
            token_type_ids: トークン種別（1: visual, 0: text）
            visual_positions: 視覚トークンの位置
            text_positions: テキストトークンの位置
            output_hidden_states: 隠れ状態も返すか
        
        Returns:
            Dict containing:
                - visual_features: 視覚特徴 [B, num_visual, vision_output_dim]
                - text_logits: テキストロジット [B, num_text, vocab_size]
                - visual_pooled: プールされた視覚特徴 [B, llm_hidden_size]
                - visual_hidden_states: 視覚隠れ状態（オプション）
                - text_hidden_states: テキスト隠れ状態（オプション）
        """
        batch_size = llm_outputs.size(0)
        device = llm_outputs.device
        
        # トークン種別による分離
        if token_type_ids is not None:
            # token_type_ids == 1の位置が視覚トークン
            visual_mask = (token_type_ids == 1)
            text_mask = (token_type_ids == 0)
            
            # 各モダリティの隠れ状態を抽出
            visual_hidden = llm_outputs[visual_mask.unsqueeze(-1).expand_as(llm_outputs)]
            text_hidden = llm_outputs[text_mask.unsqueeze(-1).expand_as(llm_outputs)]
            
            # バッチサイズと系列長を復元
            num_visual = visual_mask.sum(dim=1).max().item()
            num_text = text_mask.sum(dim=1).max().item()
            
            visual_hidden = visual_hidden.view(batch_size, -1, self.llm_hidden_size)[:, :num_visual]
            text_hidden = text_hidden.view(batch_size, -1, self.llm_hidden_size)[:, :num_text]
        
        # 位置による分離（フォールバック）
        elif visual_positions is not None and text_positions is not None:
            visual_hidden = self._extract_by_positions(llm_outputs, visual_positions)
            text_hidden = self._extract_by_positions(llm_outputs, text_positions)
        
        else:
            # 位置情報がない場合は前半を視覚、後半をテキストと仮定
            seq_len = llm_outputs.size(1)
            split_point = seq_len // 2
            visual_hidden = llm_outputs[:, :split_point]
            text_hidden = llm_outputs[:, split_point:]
        
        # 視覚特徴の処理
        visual_features = self.vision_head(visual_hidden)
        
        # 視覚特徴のプーリング（全体的な視覚表現）
        visual_pooled = self.vision_pooler(visual_hidden.mean(dim=1))
        
        # テキストロジットの生成
        if text_hidden.size(1) > 0:
            text_logits = self.language_head(text_hidden)
        else:
            # テキストがない場合はダミー
            text_logits = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        
        result = {
            'visual_features': visual_features,
            'text_logits': text_logits,
            'visual_pooled': visual_pooled,
        }
        
        if output_hidden_states:
            result['visual_hidden_states'] = visual_hidden
            result['text_hidden_states'] = text_hidden
        
        return result
    
    def _extract_by_positions(
        self,
        hidden_states: torch.Tensor,
        positions: List[int]
    ) -> torch.Tensor:
        """
        指定位置の隠れ状態を抽出
        """
        if not positions:
            return torch.zeros(hidden_states.size(0), 0, hidden_states.size(-1), 
                              device=hidden_states.device)
        
        # 位置インデックスでスライス
        extracted = hidden_states[:, positions, :]
        return extracted


class VisualLanguageSeparator(nn.Module):
    """
    視覚・言語分離機構の統合クラス
    
    ModalityAwareTokenizerとDualModalityOutputを統合し、
    エンドツーエンドの処理を提供
    """
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        llm_tokenizer=None,
        llm_embed_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        # コンポーネントの初期化
        self.tokenizer = ModalityAwareTokenizer(llm_tokenizer)
        self.output_separator = DualModalityOutput(
            llm_hidden_size=llm_config.get('hidden_size', 5120),
            vocab_size=llm_config.get('vocab_size', 128256),  # Llama-4デフォルト
            vision_output_dim=llm_config.get('vision_output_dim', 1024)
        )
        
        self.llm_embed_layer = llm_embed_layer
    
    def prepare_input(
        self,
        visual_embeds: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        マルチモーダル入力の準備
        """
        return self.tokenizer.prepare_multimodal_input(
            visual_embeds=visual_embeds,
            text_tokens=text_tokens,
            text_embeds=text_embeds,
            llm_embed_layer=self.llm_embed_layer
        )
    
    def separate_output(
        self,
        llm_outputs: torch.Tensor,
        modality_info: Dict[str, Any],
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        LLM出力の分離
        """
        return self.output_separator(
            llm_outputs=llm_outputs,
            token_type_ids=modality_info.get('token_type_ids'),
            visual_positions=modality_info.get('visual_token_positions'),
            text_positions=modality_info.get('text_token_positions'),
            output_hidden_states=output_hidden_states
        )