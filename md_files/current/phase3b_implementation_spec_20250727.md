# Phase3B実装仕様書 - o3-modification対応版

## 概要

本仕様書は、o3-modification20250727.mdに基づいて、Phase3Bモデルの根本的な設計改修を実装するための具体的な技術仕様を定めます。Webリサーチ結果（2025年最新情報）を反映し、実装可能な形で各コンポーネントの詳細設計を記述します。

## 1. Q-Former改修仕様（BLIP-2準拠テキスト入力対応）

### 1.1 現行実装の分析
- 現在の実装: HuggingFaceのBlip2QFormerModelを使用
- 問題点: テキスト入力との相互作用が未実装、LLMへの接続が不完全

### 1.2 改修内容

#### 1.2.1 Q-Formerクラスの拡張

```python
class EnhancedQFormerModel(nn.Module):
    """
    BLIP-2準拠のQ-Former実装
    - テキスト入力対応
    - 3つの学習目的（ITC, ITM, ITG）に対応するマスキング戦略
    - LLMへの適切な接続インターフェース
    """
    
    def __init__(self, config):
        super().__init__()
        
        # BLIP-2準拠のQ-Former設定
        self.config = config
        self.num_queries = config.get('num_queries', 32)  # BLIP-2標準
        
        # 公式Blip2QFormerModelを基盤として使用
        qformer_config = Blip2QFormerConfig(
            hidden_size=config['hidden_size'],  # 768 (BLIP-2標準)
            num_hidden_layers=config['num_layers'],  # 12
            num_attention_heads=config['num_heads'],  # 12
            cross_attention_frequency=2,  # 2層ごとにcross-attention
            num_query_tokens=self.num_queries,
            encoder_hidden_size=config.get('encoder_hidden_size', 1408)  # ViT-L/14
        )
        
        self.qformer = Blip2QFormerModel(qformer_config)
        
        # LLM接続用プロジェクター（BLIP-2準拠）
        self.llm_projector = nn.Linear(
            config['hidden_size'],  # 768
            config['llm_hidden_size']  # 5120 (Llama-4)
        )
        
        # 学習目的に応じたマスキング戦略
        self.masking_strategies = {
            'itc': self._create_itc_mask,  # Image-Text Contrastive
            'itm': self._create_itm_mask,  # Image-Text Matching
            'itg': self._create_itg_mask   # Image-grounded Text Generation
        }
```

#### 1.2.2 Forward処理の再設計

```python
def forward(
    self,
    image_feats: torch.Tensor,
    text_tokens: Optional[torch.Tensor] = None,
    text_attention_mask: Optional[torch.Tensor] = None,
    mode: str = 'itg',  # デフォルトは生成モード
    return_dict: bool = True
) -> Dict[str, torch.Tensor]:
    """
    BLIP-2準拠のforward処理
    
    Args:
        image_feats: 画像エンコーダ出力 [B, num_patches, encoder_dim]
        text_tokens: テキストトークン [B, text_len]
        text_attention_mask: テキストアテンションマスク [B, text_len]
        mode: 学習モード ('itc', 'itm', 'itg')
    
    Returns:
        dict containing:
            - query_embeds: クエリ埋め込み [B, num_queries, hidden_size]
            - llm_embeds: LLM用に射影されたクエリ [B, num_queries, llm_hidden_size]
            - text_outputs: テキスト出力（ITMモード時）
    """
    
    # アテンションマスクの生成（モードに応じて）
    attention_mask = self.masking_strategies[mode](
        batch_size=image_feats.size(0),
        has_text=text_tokens is not None
    )
    
    # Q-Former forward
    outputs = self.qformer(
        query_embeds=self.query_tokens.expand(image_feats.size(0), -1, -1),
        encoder_hidden_states=image_feats,
        encoder_attention_mask=torch.ones(image_feats.size()[:-1], device=image_feats.device),
        input_ids=text_tokens,
        attention_mask=text_attention_mask,
        use_cache=False,
        return_dict=True
    )
    
    # クエリ出力の取得
    query_outputs = outputs.last_hidden_state[:, :self.num_queries, :]
    
    # LLM用に射影
    llm_embeds = self.llm_projector(query_outputs)
    
    return {
        'query_embeds': query_outputs,
        'llm_embeds': llm_embeds,
        'text_outputs': outputs.last_hidden_state[:, self.num_queries:, :] if text_tokens is not None else None
    }
```

#### 1.2.3 マスキング戦略の実装

```python
def _create_itc_mask(self, batch_size: int, has_text: bool) -> torch.Tensor:
    """ITC: クエリとテキストが相互に見えないマスク"""
    # 実装詳細...
    
def _create_itm_mask(self, batch_size: int, has_text: bool) -> torch.Tensor:
    """ITM: クエリとテキストが相互作用可能なマスク"""
    # 実装詳細...
    
def _create_itg_mask(self, batch_size: int, has_text: bool) -> torch.Tensor:
    """ITG: 生成用の因果的マスク"""
    # 実装詳細...
```

## 2. 視覚・言語特徴分離機構

### 2.1 Llama-4統合部の設計

#### 2.1.1 トークン種別管理

```python
class ModalityAwareTokenizer:
    """モダリティ別トークン管理"""
    
    VISUAL_TOKEN_START = "<VISUAL_START>"
    VISUAL_TOKEN_END = "<VISUAL_END>"
    TEXT_TOKEN_START = "<TEXT_START>"
    TEXT_TOKEN_END = "<TEXT_END>"
    
    def prepare_multimodal_input(
        self,
        visual_embeds: torch.Tensor,  # [B, num_queries, llm_dim]
        text_tokens: torch.Tensor,    # [B, text_len]
        llm_tokenizer
    ) -> Dict[str, torch.Tensor]:
        """
        視覚・テキストトークンを統合し、種別情報を付与
        """
        # 特殊トークンの追加
        # 視覚埋め込みの前後にマーカートークンを配置
        # 位置情報の記録
        return {
            'input_embeds': combined_embeds,
            'token_type_ids': token_types,  # 0: text, 1: visual
            'visual_token_positions': visual_positions,
            'text_token_positions': text_positions
        }
```

#### 2.1.2 デュアルヘッド出力構造

```python
class DualModalityOutput(nn.Module):
    """視覚・言語の分離出力機構"""
    
    def __init__(self, llm_hidden_size: int, vocab_size: int):
        super().__init__()
        
        # 言語出力ヘッド（標準のLMヘッド）
        self.language_head = nn.Linear(llm_hidden_size, vocab_size)
        
        # 視覚出力ヘッド（セグメンテーション用）
        self.vision_head = nn.Sequential(
            nn.Linear(llm_hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),  # SAM2互換の次元
        )
        
    def forward(
        self,
        llm_outputs: torch.Tensor,
        token_type_ids: torch.Tensor,
        visual_positions: List[int],
        text_positions: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        LLM出力を視覚・言語に分離
        """
        # 視覚トークン部分の抽出
        visual_hidden = self._extract_by_positions(llm_outputs, visual_positions)
        visual_features = self.vision_head(visual_hidden)
        
        # テキストトークン部分の抽出
        text_hidden = self._extract_by_positions(llm_outputs, text_positions)
        text_logits = self.language_head(text_hidden)
        
        return {
            'visual_features': visual_features,  # セグメンテーション用
            'text_logits': text_logits,          # テキスト生成用
            'visual_hidden_states': visual_hidden,
            'text_hidden_states': text_hidden
        }
```

## 3. LoRAエキスパート統合設計

### 3.1 LoRAモジュール実装

```python
class LoRAExpert(nn.Module):
    """低ランク適応エキスパート"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        
        # 低ランク分解
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # スケーリング係数（PEFT準拠）
        self.scaling = 1.0 / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA補正の適用"""
        # x @ A^T @ B^T * scaling
        return x @ self.lora_A.T @ self.lora_B.T * self.scaling
```

### 3.2 MoEルーター設計

```python
class MultiModalityRouter(nn.Module):
    """マルチモーダルLoRAルーター"""
    
    def __init__(self, input_dim: int, num_experts: int = 4):
        super().__init__()
        
        # モダリティ判定ネットワーク
        self.router = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # エキスパートLoRA群
        self.experts = nn.ModuleList([
            LoRAExpert(input_dim, input_dim) for _ in range(num_experts)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        modality_hint: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        動的エキスパート選択と適用
        """
        # ルーティング重みの計算
        if modality_hint is not None:
            routing_weights = self.router(modality_hint)
        else:
            # 入力特徴から自動判定
            routing_weights = self.router(x.mean(dim=1))
        
        # エキスパート出力の重み付き結合
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=0)
        
        # [num_experts, B, ...] x [B, num_experts] -> [B, ...]
        output = torch.einsum('ebd,be->bd', expert_outputs, routing_weights)
        
        return output
```

## 4. マルチスケールセグメンテーションヘッド

### 4.1 マルチスケール特徴抽出

```python
class MultiScaleFeatureExtractor(nn.Module):
    """ViTからの多段特徴抽出"""
    
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        
        # 特徴抽出フック
        self.feature_maps = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """中間層へのフック登録"""
        # Stage 1 (高解像度): layer 3
        # Stage 2 (中解像度): layer 6  
        # Stage 3 (低解像度): layer 9
        # Stage 4 (最終): layer 12
        
        stages = [3, 6, 9, 12]
        for idx in stages:
            layer = self.vit.encoder.layer[idx-1]
            layer.register_forward_hook(
                lambda m, i, o, stage=idx: self._save_feature(o, f'stage_{stage}')
            )
```

### 4.2 拡張MaskDecoder

```python
class EnhancedMaskDecoder(nn.Module):
    """マルチスケール対応MaskDecoder"""
    
    def __init__(self, sam_decoder, hidden_dim: int = 256):
        super().__init__()
        
        # 既存のSAM2 MaskDecoder
        self.main_decoder = sam_decoder
        
        # 補助デコーダ（高解像度特徴用）
        self.aux_decoder = nn.ModuleDict({
            'high_res': nn.Sequential(
                nn.Conv2d(768, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.ReLU(),
            ),
            'mid_res': nn.Sequential(
                nn.Conv2d(768, 256, 3, padding=1),
                nn.ReLU(),
            )
        })
        
        # 特徴融合モジュール
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 3, 256, 1),  # main + high + mid
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,  # 低解像度（メイン）
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multiscale_features: Dict[str, torch.Tensor],  # 追加
        visual_context: Optional[torch.Tensor] = None  # LLMから
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        マルチスケール統合マスク生成
        """
        # メインデコーダ（低解像度特徴）
        masks_main, iou_pred = self.main_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=True
        )
        
        # 補助デコーダ（高・中解像度特徴）
        mask_high = self.aux_decoder['high_res'](multiscale_features['stage_1'])
        mask_mid = self.aux_decoder['mid_res'](multiscale_features['stage_2'])
        
        # アップサンプリングして解像度を合わせる
        mask_high_up = F.interpolate(mask_high, size=masks_main.shape[-2:], mode='bilinear')
        mask_mid_up = F.interpolate(mask_mid, size=masks_main.shape[-2:], mode='bilinear')
        
        # 特徴融合
        fused_features = torch.cat([masks_main, mask_high_up, mask_mid_up], dim=1)
        final_masks = self.fusion(fused_features)
        
        # visual_contextがある場合は追加の調整
        if visual_context is not None:
            # LLMからの視覚コンテキストで重み付け
            context_weight = self._compute_context_weight(visual_context)
            final_masks = final_masks * context_weight.unsqueeze(-1).unsqueeze(-1)
        
        return final_masks, iou_pred
```

## 5. 統合実装フロー

### 5.1 QFormerSegmentationBridgeクラスの改修

```python
class QFormerSegmentationBridge(nn.Module):
    """改修された統合モデル"""
    
    def __init__(self, config: LlamaQFormerSAM2Config):
        super().__init__()
        
        # 1. 改修Q-Former
        self.qformer = EnhancedQFormerModel(config.qformer_config)
        
        # 2. Llama-4（早期融合対応）
        self.llama = LLAMA4_MODEL_CLASS.from_pretrained(
            config.llama_model_id,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype
        )
        
        # 3. モダリティ管理
        self.modality_tokenizer = ModalityAwareTokenizer()
        self.dual_output = DualModalityOutput(
            config.llama_hidden_size,
            self.llama.config.vocab_size
        )
        
        # 4. LoRAエキスパート（SAMエンコーダ用）
        self.sam_lora_router = MultiModalityRouter(
            input_dim=1408,  # ViT-L/14
            num_experts=2    # RGB, Depth（将来拡張用）
        )
        
        # 5. マルチスケールSAM2
        self.sam_wrapper = get_sam2_wrapper(config.sam_config)
        self.multiscale_extractor = MultiScaleFeatureExtractor(
            self.sam_wrapper.image_encoder
        )
        self.enhanced_mask_decoder = EnhancedMaskDecoder(
            self.sam_wrapper.mask_decoder
        )
```

### 5.2 Forward処理の統合

```python
def forward(
    self,
    images: torch.Tensor,
    text_input: Optional[List[str]] = None,
    mode: str = 'itg'
) -> Dict[str, torch.Tensor]:
    """
    統合forward処理
    """
    # 1. マルチスケール画像特徴抽出
    with torch.no_grad():
        image_features = self.multiscale_extractor(images)
        main_features = image_features['stage_4']  # 最終層
        
    # 2. LoRAによる適応（必要に応じて）
    if self.training and hasattr(self, 'sam_lora_router'):
        main_features = main_features + self.sam_lora_router(main_features)
    
    # 3. Q-Formerでクエリ生成
    qformer_outputs = self.qformer(
        image_feats=main_features,
        text_tokens=text_tokens if text_input else None,
        mode=mode
    )
    
    # 4. Llama-4での統合処理
    multimodal_input = self.modality_tokenizer.prepare_multimodal_input(
        visual_embeds=qformer_outputs['llm_embeds'],
        text_tokens=text_tokens,
        llm_tokenizer=self.llama_tokenizer
    )
    
    llm_outputs = self.llama(**multimodal_input)
    
    # 5. 視覚・言語出力の分離
    separated_outputs = self.dual_output(
        llm_outputs.last_hidden_state,
        multimodal_input['token_type_ids'],
        multimodal_input['visual_token_positions'],
        multimodal_input['text_token_positions']
    )
    
    # 6. マルチスケールセグメンテーション
    masks, iou_scores = self.enhanced_mask_decoder(
        image_embeddings=main_features,
        image_pe=self.sam_wrapper.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=qformer_outputs['query_embeds'],
        dense_prompt_embeddings=torch.zeros_like(main_features[:, 0, :, :]),
        multiscale_features=image_features,
        visual_context=separated_outputs['visual_features']
    )
    
    return {
        'masks': masks,
        'iou_scores': iou_scores,
        'text_logits': separated_outputs['text_logits'],
        'visual_features': separated_outputs['visual_features']
    }
```

## 6. 段階的実装計画

### Phase 1: Q-Former改修（1-2日）
1. EnhancedQFormerModelの実装
2. マスキング戦略の実装
3. 単体テストの作成

### Phase 2: 視覚・言語分離機構（2-3日）
1. ModalityAwareTokenizerの実装
2. DualModalityOutputの実装
3. Llama-4統合テスト

### Phase 3: LoRAエキスパート（2日）
1. LoRAExpertモジュールの実装
2. MultiModalityRouterの実装
3. SAMエンコーダへの統合

### Phase 4: マルチスケール対応（2日）
1. MultiScaleFeatureExtractorの実装
2. EnhancedMaskDecoderの実装
3. 統合テスト

### Phase 5: 全体統合・最適化（2日）
1. QFormerSegmentationBridgeの統合
2. メモリ最適化
3. train_phase3b_qformer_bridge.pyでの検証

## 7. 評価指標

1. **機能検証**
   - テキスト入力対応の動作確認
   - 視覚・言語出力の正常分離
   - マルチスケールマスクの品質

2. **性能指標**
   - 推論速度: 44fps目標（SAM2準拠）
   - メモリ使用量: H100 4枚で動作
   - セグメンテーション精度: IoU改善

3. **拡張性確認**
   - 新モダリティ追加の容易さ
   - LoRAによる適応の効果
   - Phase4への移行準備