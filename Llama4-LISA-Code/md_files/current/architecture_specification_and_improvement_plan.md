# 現在のアーキテクチャ仕様と段階的改善実装計画

## 📅 作成日: 2025年1月19日

## 🎯 プロジェクト目標
料理画像の具材・分量予測を行うVLMの構築
- LISAの思想（VLM + SAM統合）を継承
- Llama-4のMoE構造とEarly Fusionを活用
- SAM2の高精度セグメンテーション能力を統合

## 📊 現在のアーキテクチャ仕様

### 1. コアコンポーネント構成

#### 1.1 モデル構成
```
┌─────────────────┐     ┌─────────────────┐
│  Llama-4-Scout  │     │      SAM2       │
│  (17B active)   │     │  (Hiera-large)  │
│  109B total     │     │                 │
│  16 experts     │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│            Q-Former (BLIP-2)            │
│         32 queries, 768 dim             │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│        Heterogeneous MoE Adapters       │
│    (Language, Vision, Fusion experts)   │
└─────────────────────────────────────────┘
```

#### 1.2 デュアルエンコーダー構成
- **Llama-4用**: 448×448解像度（pixel_values）
- **SAM2用**: 1024×1024解像度（sam_pixel_values）
- **メリット**: 各モデルに最適な解像度で処理

### 2. 現在の実装の特徴

#### 2.1 強み
- ✅ デュアルエンコーダーによる柔軟な画像処理
- ✅ Q-Formerによるクロスモーダル特徴抽出
- ✅ MixLoRAによるメモリ効率的な学習（41%削減）
- ✅ Phase 3B統合で28%の性能向上を達成

#### 2.2 改善可能な点
- ❌ Early Fusionが不完全（後期融合アプローチ）
- ❌ Sa2VA風統一トークン空間が未実装
- ❌ 食品特化のエキスパートが不在
- ❌ SAM2の制御が静的（動的制御なし）

### 3. 技術スタック
```python
# 主要ライブラリ
- PyTorch 2.x
- Transformers 4.45+
- PEFT (LoRA/MixLoRA)
- SAM2 (Meta公式)
- BLIP-2 Q-Former (HuggingFace)
```

## 🚀 段階的改善実装計画

### 優先度1: Sa2VA風統一トークン空間の実装

#### 概要
Sa2VAの最新アプローチ（2025年1月）を参考に、テキスト・画像・セグメンテーションを統一トークン空間で扱う。

#### 実装詳細

```python
# model/unified_token_space.py (新規作成)

class UnifiedTokenSpace(nn.Module):
    """Sa2VA風統一トークン空間モジュール"""
    
    def __init__(self, config):
        super().__init__()
        # 各モダリティの次元
        self.llama_dim = config.llama_hidden_size  # 5120
        self.sam_dim = config.sam_prompt_embed_dim  # 256
        self.qformer_dim = config.qformer_config['hidden_size']  # 768
        self.unified_dim = config.llama_hidden_size  # LLMトークン空間に統一
        
        # プロジェクション層
        self.sam_to_unified = nn.Linear(self.sam_dim, self.unified_dim)
        self.qformer_to_unified = nn.Linear(self.qformer_dim, self.unified_dim)
        
        # 指示トークン生成器
        self.instruction_generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.unified_dim,
                nhead=16,
                dim_feedforward=self.unified_dim * 4
            ),
            num_layers=4
        )
        
        # 学習可能な指示クエリ
        self.instruction_queries = nn.Parameter(
            torch.randn(1, 8, self.unified_dim)  # 8個の指示クエリ
        )
    
    def forward(self, llama_features, sam_features, qformer_features):
        """
        各モダリティの特徴を統一トークン空間に投影し、
        SAM2制御用の指示トークンを生成
        """
        # 1. 統一空間への投影
        sam_unified = self.sam_to_unified(sam_features)
        qformer_unified = self.qformer_to_unified(qformer_features)
        
        # 2. トークン結合（Llama特徴は既にunified_dim）
        unified_tokens = torch.cat([
            llama_features,      # (B, L1, unified_dim)
            sam_unified,         # (B, L2, unified_dim)
            qformer_unified      # (B, L3, unified_dim)
        ], dim=1)
        
        # 3. 指示トークン生成
        batch_size = llama_features.size(0)
        instruction_queries = self.instruction_queries.expand(batch_size, -1, -1)
        
        instruction_tokens = self.instruction_generator(
            tgt=instruction_queries,
            memory=unified_tokens
        )
        
        return instruction_tokens, unified_tokens
```

#### 統合方法
```python
# model/llama4_qformer_sam2.py に追加

def __init__(self, config, ...):
    # 既存の初期化コード
    ...
    # 統一トークン空間モジュール追加
    self.use_unified_token_space = config.use_unified_token_space
    if self.use_unified_token_space:
        self.unified_token_space = UnifiedTokenSpace(config)
    
def forward(self, images, input_ids, sam_images=None, ...):
    # 既存のデュアルエンコーダー処理
    llama_outputs = self.llama_model(images, ...)
    qformer_outputs = self.qformer(...)
    
    # 統一トークン空間処理（追加）
    if self.use_unified_token_space:
        instruction_tokens, unified_features = self.unified_token_space(
            llama_features=llama_outputs.hidden_states[-1],
            sam_features=sam_prompts,
            qformer_features=qformer_outputs['query_embeds']
        )
        
        # SAM2を指示トークンで制御
        sam_outputs = self.sam2.predict_with_instructions(
            image=sam_images if sam_images is not None else images,
            instruction_tokens=instruction_tokens
        )
```

### 優先度2: Early Fusionの真の実装

#### 概要
Llama-4の設計思想である「即座のトークン結合」を実装し、より深いクロスモーダル理解を実現。

#### 実装詳細

```python
# model/early_fusion.py (新規作成)

class EarlyFusionModule(nn.Module):
    """Llama-4スタイルのEarly Fusion実装"""
    
    def __init__(self, config):
        super().__init__()
        self.vision_dim = config.llama_hidden_size
        self.text_dim = config.llama_hidden_size
        
        # iRoPE（interpolated Rotary Position Embeddings）風の位置エンコーディング
        self.vision_pos_embed = nn.Parameter(
            torch.randn(1, 1024, self.vision_dim)  # 最大1024トークン
        )
        self.cross_modal_pos_embed = nn.Parameter(
            torch.randn(1, 2048, self.vision_dim)  # 結合後の最大長
        )
        
        # クロスモーダルアテンション層
        self.cross_modal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.vision_dim,
                nhead=32,  # Llama-4準拠
                dim_feedforward=self.vision_dim * 4,
                activation='silu',  # Llama-4準拠
                batch_first=True
            ) for _ in range(4)  # 4層の早期融合
        ])
        
        # モダリティタイプ埋め込み
        self.modality_embeddings = nn.Embedding(3, self.vision_dim)  # text, vision, fusion
    
    def forward(self, text_features, vision_features, attention_mask=None):
        """
        テキストとビジョン特徴を即座に結合し、
        クロスモーダル相互作用を早期に実現
        """
        batch_size = text_features.size(0)
        
        # 1. モダリティ埋め込みを追加
        text_modality = self.modality_embeddings(
            torch.zeros(batch_size, text_features.size(1), dtype=torch.long, device=text_features.device)
        )
        vision_modality = self.modality_embeddings(
            torch.ones(batch_size, vision_features.size(1), dtype=torch.long, device=vision_features.device)
        )
        
        text_features = text_features + text_modality
        vision_features = vision_features + vision_modality
        
        # 2. 位置エンコーディングを追加
        vision_features = vision_features + self.vision_pos_embed[:, :vision_features.size(1), :]
        
        # 3. 即座の結合（Llama-4の核心）
        fused_features = torch.cat([text_features, vision_features], dim=1)
        
        # 4. クロスモーダル位置エンコーディング
        fused_features = fused_features + self.cross_modal_pos_embed[:, :fused_features.size(1), :]
        
        # 5. 早期クロスモーダルアテンション
        for layer in self.cross_modal_layers:
            fused_features = layer(fused_features, src_key_padding_mask=attention_mask)
        
        # 6. モダリティ別に分離して返す（必要に応じて）
        text_len = text_features.size(1)
        enhanced_text = fused_features[:, :text_len, :]
        enhanced_vision = fused_features[:, text_len:, :]
        
        return {
            'fused_features': fused_features,
            'enhanced_text': enhanced_text,
            'enhanced_vision': enhanced_vision
        }
```

#### 統合方法
```python
# model/llama4_qformer_sam2.py に追加

def __init__(self, config, ...):
    # 既存の初期化
    ...
    # Early Fusionモジュール追加
    self.use_early_fusion = config.early_fusion
    if self.use_early_fusion:
        self.early_fusion_module = EarlyFusionModule(config)

def forward(self, images, input_ids, sam_images=None, ...):
    # Llama-4のマルチモーダル出力を取得
    llama_outputs = self.llama_model(...)
    
    # Early Fusion適用（Q-Former前に実行）
    if self.use_early_fusion:
        # テキストとビジョンの隠れ状態を取得
        text_hidden = llama_outputs.hidden_states[-1][:, :input_ids.size(1), :]
        vision_hidden = llama_outputs.hidden_states[-1][:, input_ids.size(1):, :]
        
        # Early Fusion
        fusion_outputs = self.early_fusion_module(
            text_features=text_hidden,
            vision_features=vision_hidden,
            attention_mask=attention_mask
        )
        
        # Q-Formerには融合済み特徴を渡す
        encoder_hidden_states = fusion_outputs['fused_features']
    else:
        encoder_hidden_states = llama_outputs.hidden_states[-1]
    
    # Q-Former処理（融合済み特徴を使用）
    qformer_outputs = self.qformer(
        query_embeds=query_embeds,
        encoder_hidden_states=encoder_hidden_states,
        ...
    )
```

### 優先度3: 食品特化MoEエキスパートの追加

#### 概要
RoDE（食品認識向けMoE）の知見を活用し、材料・分量・調理法に特化したエキスパートを追加。

#### 実装詳細

```python
# model/food_experts.py (新規作成)

class FoodSpecificExperts(nn.Module):
    """食品認識タスク特化のMoEエキスパート"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.llama_hidden_size
        
        # RoDE風の異なるランクのLoRAエキスパート
        self.experts = nn.ModuleDict({
            # 材料認識（シンプル）
            'ingredient': self._create_lora_expert(rank=8, alpha=16),
            
            # 分量予測（複雑）
            'quantity': self._create_lora_expert(rank=32, alpha=64),
            
            # 調理法認識（中程度）
            'cooking_method': self._create_lora_expert(rank=16, alpha=32),
            
            # 栄養成分推定（複雑）
            'nutrition': self._create_lora_expert(rank=32, alpha=64),
            
            # 料理カテゴリ分類（シンプル）
            'category': self._create_lora_expert(rank=4, alpha=8)
        })
        
        # 線形整流ルーター（RoDE準拠）
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),  # 線形整流
            nn.Linear(256, len(self.experts)),
            nn.Softmax(dim=-1)
        )
        
        # タスク認識器（入力からタスクを推定）
        self.task_recognizer = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(self.experts)),
            nn.Sigmoid()
        )
    
    def _create_lora_expert(self, rank, alpha):
        """LoRAエキスパートの作成"""
        return nn.ModuleDict({
            'lora_A': nn.Linear(self.hidden_size, rank, bias=False),
            'lora_B': nn.Linear(rank, self.hidden_size, bias=False),
            'scaling': nn.Parameter(torch.tensor(alpha / rank))
        })
    
    def forward(self, hidden_states, task_hints=None):
        """
        食品特化エキスパートによる処理
        
        Args:
            hidden_states: (B, L, D) 入力特徴
            task_hints: タスクのヒント（例: "ingredient", "quantity"）
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. タスク認識
        if task_hints is None:
            # 入力から自動的にタスクを推定
            pooled = hidden_states.mean(dim=1)  # (B, D)
            task_weights = self.task_recognizer(pooled)  # (B, num_experts)
        else:
            # 明示的なタスクヒントを使用
            task_weights = self._encode_task_hints(task_hints, batch_size)
        
        # 2. ルーティング重みの計算
        routing_weights = self.router(hidden_states.mean(dim=1))  # (B, num_experts)
        
        # 3. タスクとルーティングの組み合わせ
        combined_weights = routing_weights * task_weights  # (B, num_experts)
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. エキスパート処理
        expert_outputs = []
        for i, (name, expert) in enumerate(self.experts.items()):
            # LoRA変換
            lora_output = expert['lora_B'](
                expert['lora_A'](hidden_states)
            ) * expert['scaling']
            
            # 重み付け
            weight = combined_weights[:, i:i+1].unsqueeze(1)  # (B, 1, 1)
            expert_outputs.append(lora_output * weight)
        
        # 5. エキスパート出力の統合
        output = sum(expert_outputs) + hidden_states  # 残差接続
        
        # 6. 統計情報
        expert_stats = {
            'routing_weights': routing_weights,
            'task_weights': task_weights,
            'combined_weights': combined_weights,
            'active_experts': [
                self.experts.keys()[i] 
                for i in torch.topk(combined_weights.mean(0), k=2).indices
            ]
        }
        
        return output, expert_stats
```

#### 統合方法
```python
# model/moe_adapters.py を修正

def create_heterogeneous_moe_adapter(
    llama_model,
    sam_model=None,
    qformer_model=None,
    config=None,
    include_food_experts=True  # 新規パラメータ
):
    """Heterogeneous MoE Adapter作成（食品エキスパート対応）"""
    
    # 既存のエキスパート作成
    experts = {}
    expert_weights = {}
    
    # 既存のエキスパート（言語、ビジョン、融合）
    if llama_model is not None:
        experts['llama'] = LoRAExpert(...)
        expert_weights['llama'] = 0.3
    
    if sam_model is not None:
        experts['sam2'] = LoRAExpert(...)
        expert_weights['sam2'] = 0.3
    
    if qformer_model is not None:
        experts['qformer'] = LoRAExpert(...)
        expert_weights['qformer'] = 0.2
    
    # 食品特化エキスパートの追加
    if include_food_experts:
        food_experts = FoodSpecificExperts(config)
        experts['food'] = food_experts
        expert_weights['food'] = 0.2
    
    # 重みの正規化
    total_weight = sum(expert_weights.values())
    expert_weights = {k: v/total_weight for k, v in expert_weights.items()}
    
    return HeterogeneousMoEAdapter(
        experts=experts,
        expert_weights=expert_weights,
        config=config
    )
```

### 優先度4: SAM2の動的制御実装

#### 概要
Sa2VAのようにLLMが生成した指示トークンでSAM2を動的に制御し、より精密なセグメンテーションを実現。

#### 実装詳細

```python
# model/sam2_dynamic_control.py (新規作成)

class SAM2DynamicController(nn.Module):
    """LLM指示トークンによるSAM2動的制御"""
    
    def __init__(self, config):
        super().__init__()
        self.instruction_dim = config.llama_hidden_size
        self.sam_prompt_dim = config.sam_prompt_embed_dim
        
        # 指示トークンをSAMプロンプトに変換
        self.instruction_to_prompt = nn.Sequential(
            nn.Linear(self.instruction_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, self.sam_prompt_dim * 4),  # 4種類のプロンプト
            nn.Tanh()
        )
        
        # プロンプトタイプ分類器
        self.prompt_classifier = nn.Sequential(
            nn.Linear(self.instruction_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # point, box, mask, text
            nn.Softmax(dim=-1)
        )
        
        # 適応的しきい値予測
        self.threshold_predictor = nn.Sequential(
            nn.Linear(self.instruction_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # マルチマスク選択器
        self.mask_selector = nn.Sequential(
            nn.Linear(self.instruction_dim + 256, 512),  # 指示+SAM特徴
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),  # SAM2は3つのマスクを生成
            nn.Softmax(dim=-1)
        )
    
    def forward(self, instruction_tokens, sam_features=None):
        """
        指示トークンからSAM2制御信号を生成
        
        Returns:
            control_signals: SAM2制御用の各種信号
        """
        batch_size = instruction_tokens.size(0)
        
        # 1. 指示トークンのプーリング
        if instruction_tokens.dim() == 3:
            pooled_instruction = instruction_tokens.mean(dim=1)
        else:
            pooled_instruction = instruction_tokens
        
        # 2. SAMプロンプト生成
        sam_prompts = self.instruction_to_prompt(pooled_instruction)
        sam_prompts = sam_prompts.view(batch_size, 4, self.sam_prompt_dim)
        
        # 3. プロンプトタイプの決定
        prompt_types = self.prompt_classifier(pooled_instruction)
        
        # 4. セグメンテーションしきい値
        seg_threshold = self.threshold_predictor(pooled_instruction)
        
        # 5. マスク選択重み（SAM特徴が利用可能な場合）
        if sam_features is not None:
            combined = torch.cat([
                pooled_instruction,
                sam_features.mean(dim=(1, 2)) if sam_features.dim() > 2 else sam_features
            ], dim=-1)
            mask_weights = self.mask_selector(combined)
        else:
            mask_weights = torch.ones(batch_size, 3) / 3
        
        return {
            'prompts': sam_prompts,
            'prompt_types': prompt_types,
            'threshold': seg_threshold,
            'mask_weights': mask_weights,
            'control_mode': 'dynamic'
        }
    
    def generate_sam2_inputs(self, control_signals, image_embeddings):
        """制御信号をSAM2の入力形式に変換"""
        # 最も確率の高いプロンプトタイプを選択
        prompt_type_idx = control_signals['prompt_types'].argmax(dim=-1)
        
        sam_inputs = {
            'image_embeddings': image_embeddings,
            'prompt_embeddings': control_signals['prompts'],
            'prompt_type': prompt_type_idx,
            'mask_threshold': control_signals['threshold'],
            'multimask_output': True,
            'return_logits': True
        }
        
        return sam_inputs
```

#### 統合方法
```python
# model/llama4_qformer_sam2.py に追加

def __init__(self, config, ...):
    # 既存の初期化
    ...
    # 動的制御モジュール追加
    self.use_dynamic_sam_control = config.use_dynamic_sam_control
    if self.use_dynamic_sam_control:
        self.sam_controller = SAM2DynamicController(config)

def forward(self, images, input_ids, sam_images=None, ...):
    # 既存の処理...
    
    # SAM2セグメンテーション（動的制御版）
    if self.use_dynamic_sam_control and hasattr(self, 'unified_token_space'):
        # 統一トークン空間からの指示トークンを使用
        control_signals = self.sam_controller(
            instruction_tokens=instruction_tokens,
            sam_features=sam_features
        )
        
        # 動的制御でSAM2実行
        sam_outputs = []
        for batch_idx in range(batch_size):
            # 制御信号に基づいてSAM2を実行
            sam_inputs = self.sam_controller.generate_sam2_inputs(
                {k: v[batch_idx] for k, v in control_signals.items()},
                image_embeddings=sam_embeddings[batch_idx]
            )
            
            # SAM2予測
            masks = self.sam2.predict_dynamic(**sam_inputs)
            
            # マスク選択
            selected_mask = self._select_best_mask(
                masks, 
                control_signals['mask_weights'][batch_idx]
            )
            sam_outputs.append(selected_mask)
        
        predicted_masks = torch.stack(sam_outputs)
    else:
        # 既存の静的SAM2処理
        predicted_masks = self._original_sam2_process(...)
```

## 📝 実装ロードマップ

### Phase 1: 基盤準備（1週間）
1. 現在のコードのリファクタリング
2. 設定ファイルへの新規フラグ追加
3. テストコードの準備

### Phase 2: 優先度1実装（2週間）
1. UnifiedTokenSpaceモジュールの実装
2. 既存モデルとの統合
3. 単体テストと性能評価

### Phase 3: 優先度2実装（2週間）
1. EarlyFusionModuleの実装
2. Llama-4との深い統合
3. アブレーション実験

### Phase 4: 優先度3実装（1週間）
1. FoodSpecificExpertsの実装
2. 既存MoEとの統合
3. 食品データでの評価

### Phase 5: 優先度4実装（1週間）
1. SAM2DynamicControllerの実装
2. 動的制御の統合
3. セグメンテーション精度評価

### Phase 6: 統合評価（1週間）
1. 全機能の統合テスト
2. 食品データセットでの総合評価
3. 最適化とチューニング

## 🎯 期待される成果

1. **性能向上**
   - セグメンテーション精度: +10-15%（Sa2VA準拠）
   - 食品認識精度: +15-20%（RoDE準拠）
   - 推論速度: 維持または向上

2. **機能拡張**
   - 材料の自動認識
   - 分量の精密推定
   - 調理法の理解
   - 栄養成分の推定

3. **実用性**
   - レシピ生成への応用
   - 食事管理アプリへの統合
   - 料理教育ツールとしての活用

## 📌 注意事項

- すべての改善は**現在のデュアルエンコーダー構成を維持**
- 後方互換性を保証（既存APIの変更なし）
- 段階的な機能追加により、リスクを最小化
- 各段階で性能評価を実施

## 🔗 参考文献

1. Sa2VA: Marrying SAM2 with LLaVA (arXiv:2501.04001, 2025)
2. RoDE: Linear Rectified Mixture of Diverse Experts for Food LMMs (arXiv:2407.12730, 2024)
3. MixLoRA: Enhancing LLMs Fine-Tuning with LoRA-based MoE (arXiv:2404.15159, 2024)
4. Llama 4 Technical Documentation (Meta, 2025)
5. Customize SAM for Multi-Modal Semantic Segmentation with MoE (arXiv:2412.04220, 2024)