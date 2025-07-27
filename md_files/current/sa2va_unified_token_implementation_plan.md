# Sa2VA風統一トークン空間の段階的実装計画

## 📅 作成日: 2025年1月19日

## 🎯 目的
現在のLlama-4 + Q-Former + SAM2アーキテクチャに、Sa2VA (arXiv:2501.04001) の統一トークン空間アプローチを段階的に統合し、セグメンテーション性能を最大20%向上させる。

## 📊 現状分析

### 現在のアーキテクチャの強み
- **Q-Former**: 既にクロスモーダル特徴抽出を実現（32クエリ、768次元）
- **デュアルパスウェイデコーダ**: Phase 3Bで28.14%性能向上達成
- **MoEアダプター**: 拡張可能な専門エキスパート構造
- **デュアルエンコーダー**: Llama-4（448px）とSAM2（1024px）に最適化

### Sa2VAの核心技術
- **統一トークン空間**: テキスト・画像・動画を単一のLLMトークン空間で処理
- **[SEG]トークン**: LLMの隠れ状態からSAM2制御信号を生成
- **エンドツーエンド学習**: バックプロパゲーションによる統合最適化

## 🚀 段階的実装計画

### Phase 1: 軽量版[SEG]トークン実装（1週間）

#### 目標
Q-Formerの出力を活用して、最小限の追加でSa2VA風の[SEG]トークンを実装

#### 実装内容

```python
# model/seg_token_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

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
        
    def forward(
        self, 
        qformer_outputs: Dict[str, torch.Tensor],
        llama_hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Q-Former出力から[SEG]トークンとSAM2プロンプトを生成
        
        Args:
            qformer_outputs: Q-Formerの出力辞書
            llama_hidden_states: Llama-4の隠れ状態（オプション）
        
        Returns:
            seg_outputs: [SEG]トークンとSAM2制御信号
        """
        batch_size = qformer_outputs['query_embeds'].size(0)
        
        # 1. Q-Formerクエリから[SEG]トークン候補を生成
        query_embeds = qformer_outputs['query_embeds']  # (B, 32, 768)
        
        # 2. 学習可能な[SEG]トークン埋め込みとの注意機構
        seg_embedding = self.seg_token_embedding.expand(batch_size, -1, -1)
        
        # クエリとの類似度計算
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
        seg_token_hidden = self.seg_token_projector(seg_token_qformer)  # (B, 5120)
        
        # 4. Llama隠れ状態との融合（利用可能な場合）
        if llama_hidden_states is not None:
            # 最終層の平均プーリング
            llama_pooled = llama_hidden_states.mean(dim=1)  # (B, 5120)
            # 残差接続で融合
            seg_token_hidden = seg_token_hidden + 0.5 * llama_pooled
        
        # 5. SAM2プロンプト生成
        sam_prompt = self.to_sam_prompt(seg_token_hidden)  # (B, 256)
        
        return {
            'seg_token': seg_token_hidden,  # Llama空間の[SEG]トークン
            'sam_prompt': sam_prompt,  # SAM2制御用プロンプト
            'attention_weights': attention_weights.squeeze(1),  # デバッグ用
        }
```

#### 統合方法

```python
# model/llama4_qformer_sam2.pyへの統合

def __init__(self, config, ...):
    # 既存の初期化
    ...
    # [SEG]トークン生成器の追加
    self.use_seg_token = config.use_seg_token
    if self.use_seg_token:
        from model.seg_token_generator import LightweightSEGTokenGenerator
        self.seg_token_generator = LightweightSEGTokenGenerator(config)

def forward(self, images, input_ids, sam_images=None, ...):
    # 既存の処理
    qformer_outputs = self.qformer(...)
    llama_outputs = self.llama_model(...)
    
    # [SEG]トークン生成（新規追加）
    if self.use_seg_token:
        seg_outputs = self.seg_token_generator(
            qformer_outputs=qformer_outputs,
            llama_hidden_states=llama_outputs.hidden_states[-1]
        )
        
        # SAM2への適用
        if self.sam2_wrapper is not None:
            # 既存のSAM2処理を[SEG]トークンで制御
            sam_outputs = self.sam2_wrapper(
                images=sam_images if sam_images is not None else images,
                prompts=seg_outputs['sam_prompt'],  # [SEG]トークン由来のプロンプト
                use_dynamic_control=True
            )
```

### Phase 2: 部分的統一トークン空間（2週間）

#### 目標
Q-Formerクエリを中心に、部分的な統一トークン空間を実現

#### 実装内容

```python
# model/partial_unified_token_space.py

class PartialUnifiedTokenSpace(nn.Module):
    """
    Q-Formerを中心とした部分的統一トークン空間
    完全統一より実装が簡単で、メモリ効率的
    """
    
    def __init__(self, config):
        super().__init__()
        # 統一次元（Llama-4に合わせる）
        self.unified_dim = config.llama_hidden_size  # 5120
        
        # 各モダリティから統一空間への投影
        self.projectors = nn.ModuleDict({
            'qformer': nn.Linear(768, self.unified_dim),
            'sam': nn.Linear(256, self.unified_dim),
            'text': nn.Identity(),  # Llamaは既に統一次元
        })
        
        # クロスモーダル融合層
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.unified_dim,
                nhead=16,
                dim_feedforward=self.unified_dim * 4,
                activation='gelu',
                batch_first=True
            ) for _ in range(2)  # 2層で軽量化
        ])
        
        # モダリティ位置埋め込み
        self.modality_embeddings = nn.Embedding(3, self.unified_dim)
        
    def forward(
        self,
        qformer_features: torch.Tensor,
        sam_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        部分的統一トークン空間での処理
        """
        tokens = []
        modality_ids = []
        
        # Q-Former特徴（必須）
        qformer_unified = self.projectors['qformer'](qformer_features)
        tokens.append(qformer_unified)
        modality_ids.extend([0] * qformer_unified.size(1))
        
        # SAM特徴（オプション）
        if sam_features is not None:
            sam_unified = self.projectors['sam'](sam_features)
            tokens.append(sam_unified)
            modality_ids.extend([1] * sam_unified.size(1))
        
        # テキスト特徴（オプション）
        if text_features is not None:
            text_unified = self.projectors['text'](text_features)
            tokens.append(text_unified)
            modality_ids.extend([2] * text_unified.size(1))
        
        # トークン結合
        unified_tokens = torch.cat(tokens, dim=1)
        
        # モダリティ埋め込み追加
        batch_size = unified_tokens.size(0)
        modality_ids_tensor = torch.tensor(
            modality_ids, 
            device=unified_tokens.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        modality_embeds = self.modality_embeddings(modality_ids_tensor)
        unified_tokens = unified_tokens + modality_embeds
        
        # クロスモーダル融合
        for layer in self.fusion_layers:
            unified_tokens = layer(unified_tokens)
        
        # モダリティ別に分離（必要に応じて）
        qformer_len = qformer_features.size(1)
        enhanced_qformer = unified_tokens[:, :qformer_len]
        
        return {
            'unified_tokens': unified_tokens,
            'enhanced_qformer': enhanced_qformer,
            'fusion_complete': True
        }
```

### Phase 3: 動的SAM2制御（1週間）

#### 目標
[SEG]トークンによるSAM2の動的制御を実現

#### 実装内容

```python
# model/dynamic_sam_controller.py

class DynamicSAMController(nn.Module):
    """
    [SEG]トークンによるSAM2動的制御
    プロンプトタイプ、しきい値、マスク選択を動的に決定
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.llama_hidden_size
        self.sam_prompt_dim = config.qformer_config['sam_prompt_dim']
        
        # プロンプトタイプ予測器
        self.prompt_type_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # point, box, mask, hybrid
        )
        
        # 動的しきい値予測器
        self.threshold_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # low, med, high thresholds
        )
        
        # マスク品質予測器
        self.mask_quality_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),  # SAM2は3つのマスクを生成
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        seg_token: torch.Tensor,
        sam_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        [SEG]トークンからSAM2制御信号を生成
        """
        batch_size = seg_token.size(0)
        
        # プロンプトタイプの決定
        prompt_logits = self.prompt_type_predictor(seg_token)
        prompt_probs = F.softmax(prompt_logits, dim=-1)
        prompt_type = prompt_probs.argmax(dim=-1)
        
        # しきい値の予測
        thresholds = torch.sigmoid(self.threshold_predictor(seg_token))
        
        # マスク選択重み（SAM特徴が利用可能な場合）
        if sam_features is not None:
            combined = torch.cat([
                seg_token,
                sam_features.mean(dim=(1, 2)) if sam_features.dim() > 2 else sam_features
            ], dim=-1)
            mask_weights = self.mask_quality_predictor(combined)
        else:
            mask_weights = torch.ones(batch_size, 3, device=seg_token.device) / 3
        
        return {
            'prompt_type': prompt_type,
            'prompt_probs': prompt_probs,
            'thresholds': thresholds,
            'mask_weights': mask_weights,
            'control_mode': 'dynamic'
        }
```

## 📊 期待される成果

### 性能向上予測
| Phase | 実装内容 | 期待性能向上 | 実装期間 | リスク |
|-------|---------|------------|----------|--------|
| Phase 1 | 軽量[SEG]トークン | +5-8% | 1週間 | 低 |
| Phase 2 | 部分統一空間 | +10-12% | 2週間 | 中 |
| Phase 3 | 動的制御 | +15-20% | 1週間 | 中 |

### 技術的メリット
1. **既存アーキテクチャとの親和性**: Q-Formerベースで実装
2. **段階的リスク管理**: 各フェーズで効果測定可能
3. **メモリ効率**: 完全統一より軽量
4. **拡張性**: 将来的な完全統一への道筋

## 🔧 実装上の注意事項

### メモリ管理
- 現在のGPU RAM分散設定（2x H100）を維持
- 追加モジュールは`torch.bfloat16`で統一
- 必要に応じてgradient checkpointing使用

### 互換性維持
- 既存APIの変更なし
- `config.use_seg_token`フラグで制御
- 後方互換性を保証

### テスト戦略
- 各フェーズでユニットテスト実装
- A/Bテストによる性能比較
- Lambda Cloud環境での検証

## 📅 実装スケジュール

### Week 1: Phase 1実装
- [ ] `seg_token_generator.py`の実装
- [ ] `llama4_qformer_sam2.py`への統合
- [ ] ユニットテスト作成
- [ ] 性能評価（+5-8%目標）

### Week 2-3: Phase 2実装
- [ ] `partial_unified_token_space.py`の実装
- [ ] 既存モデルとの統合
- [ ] アブレーション実験
- [ ] 性能評価（+10-12%目標）

### Week 4: Phase 3実装
- [ ] `dynamic_sam_controller.py`の実装
- [ ] SAM2ラッパーの拡張
- [ ] 統合テスト
- [ ] 最終性能評価（+15-20%目標）

## 🎯 成功基準

1. **Phase 1**: 基本的な[SEG]トークン生成が動作し、5%以上の性能向上
2. **Phase 2**: 部分統一空間でクロスモーダル理解が向上、10%以上の改善
3. **Phase 3**: 動的制御により多様なセグメンテーションタスクに対応、15%以上の向上

## 📚 参考文献

1. Sa2VA: Marrying SAM2 with LLaVA (arXiv:2501.04001, 2025)
2. 現在のPhase 3B実装（28.14%改善達成）
3. Q-Former (BLIP-2) アーキテクチャ
4. SAM2公式ドキュメント