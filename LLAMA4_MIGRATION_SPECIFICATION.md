# LISA-Llama4 移行実装仕様書

**プロジェクト:** LISA-Gemma → LISA-Llama4-Scout-17B-16E-Instruct  
**作成日:** 2025 年 6 月 20 日  
**バージョン:** v1.0

## 📋 目次

1. [概要と移行戦略](#概要と移行戦略)
2. [技術仕様比較](#技術仕様比較)
3. [段階的実装計画](#段階的実装計画)
4. [デュアルエンコーダー問題への対応](#デュアルエンコーダー問題への対応)
5. [実装詳細](#実装詳細)
6. [検証手順](#検証手順)

---

## 📊 概要と移行戦略

### 移行の背景

- **現状:** Gemma-3-4b-it + SAM による LISA 実装
- **目標:** Llama-4-Scout-17B-16E-Instruct + SAM による高性能 LISA 実装
- **戦略:** 段階的移行によるリスク最小化

### 主要な変更点

1. **VLM の変更:** Gemma-3-4b-it → Llama-4-Scout-17B-16E-Instruct
2. **視覚エンコーダー変更:** SigLIP (896×896) → Llama-4 内蔵ビジョン + SAM (1024×1024)
3. **コンテキスト長変更:** 128K → 10M tokens
4. **アーキテクチャ変更:** Dense → MoE (16 experts)

---

## 🔍 技術仕様比較

| 項目                     | Gemma-3-4b-it                    | Llama-4-Scout-17B-16E-Instruct   |
| ------------------------ | -------------------------------- | -------------------------------- |
| **パラメータ数**         | 4B                               | 17B (activated) / 109B (total)   |
| **アーキテクチャ**       | Dense Transformer                | MoE (16 experts)                 |
| **コンテキスト長**       | 128K tokens                      | 10M tokens                       |
| **ビジョンエンコーダー** | SigLIP (896×896)                 | Native Multimodal                |
| **トークナイザー**       | SentencePiece (262K)             | Llama tokenizer                  |
| **必要 transformers**    | ≥4.46.0                          | ≥4.51.0                          |
| **モデルクラス**         | `Gemma3ForConditionalGeneration` | `Llama4ForConditionalGeneration` |
| **プロセッサー**         | `AutoProcessor` (Gemma3)         | `AutoProcessor` (Llama4)         |
| **特殊トークン**         | BOI token, SEG token             | Image tokens, SEG token          |
| **画像処理**             | pan-and-scan 対応                | Multi-image (up to 5)            |

---

## 📅 段階的実装計画

### Phase 1: 基盤環境準備 (1-2 日)

**目標:** Llama-4 対応の基盤環境構築

#### 1.1 依存関係更新

```bash
# requirements.txt 更新
transformers>=4.51.0
torch>=2.1.0
torchvision>=0.16.0
accelerate>=0.20.0
```

#### 1.2 設定ファイル作成

```python
# config_llama4.py
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_TYPE = "llama4"
VISION_ENCODER_TYPE = "native"  # Llama-4のネイティブビジョン
CONTEXT_LENGTH = 10_000_000  # 10M tokens
MAX_IMAGES_PER_INPUT = 5
IMAGE_TOKEN = "<image>"
SEG_TOKEN = "[SEG]"
HIDDEN_SIZE = 4096  # Llama-4-Scout hidden dimension
VOCAB_SIZE = 128256  # Llama tokenizer vocabulary size
```

### Phase 2: モデルアーキテクチャ実装 (3-4 日)

**目標:** Llama-4 対応の LISA モデル実装

#### 2.1 Llama4-LISA モデル作成

```python
# model/llama4_lisa.py
import torch
import torch.nn as nn
from transformers import (
    Llama4ForConditionalGeneration,
    AutoProcessor
)
from .segment_anything import build_sam_vit_h

class Llama4LisaMetaModel:
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config

        # Llama-4 native multimodal capabilities
        self.native_vision = True
        self.max_images = kwargs.get("max_images", 5)

        self.initialize_lisa_modules(config)

    def initialize_lisa_modules(self, config):
        # SAM (外部セグメンテーション専用)
        self.visual_model = build_sam_vit_h(
            checkpoint=config.sam_checkpoint_path
        )

        # SAMは凍結（Llama-4のビジョンと独立）
        for param in self.visual_model.parameters():
            param.requires_grad = False

        # マスクデコーダーのみ学習可能
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # プロジェクション層
        # Llama-4 hidden_size → SAM hidden_size
        self.text_hidden_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.out_dim),
                nn.Dropout(0.0),
            )
        ])

class Llama4LisaModel(Llama4LisaMetaModel, Llama4ForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # Llama-4固有の設定
        self.config.use_cache = False

class LISALlama4ForCausalLM(Llama4LisaModel):
    def __init__(self, config, **kwargs):
        # 損失重み設定
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)

        # SEGトークンID
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config, **kwargs)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """SAM用視覚特徴抽出 (1024×1024)"""
        with torch.no_grad():
            image_embeddings_list = []
            batch_size = pixel_values.shape[0]

            for i in range(batch_size):
                torch.cuda.empty_cache()
                # SAM image encoder
                embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(embeddings)

            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)

        return image_embeddings
```

#### 2.2 デュアルストリーム処理実装

```python
# utils/llama4_processing.py
from transformers import AutoProcessor
import torch
from PIL import Image
import numpy as np

class Llama4DualStreamProcessor:
    def __init__(self, model_id):
        # Llama-4プロセッサー（ネイティブマルチモーダル）
        self.llama4_processor = AutoProcessor.from_pretrained(model_id)

        # SEGトークン追加
        seg_token = "[SEG]"
        if seg_token not in self.llama4_processor.tokenizer.get_vocab():
            self.llama4_processor.tokenizer.add_tokens([seg_token])

        self.seg_token_id = self.llama4_processor.tokenizer.convert_tokens_to_ids(seg_token)

    def preprocess_dual_stream(self, image_path, text_prompt):
        """
        デュアルストリーム前処理
        1. Llama-4用：ネイティブ処理
        2. SAM用：1024×1024リサイズ
        """
        image = Image.open(image_path).convert('RGB')

        # 1. Llama-4用前処理（ネイティブマルチモーダル）
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt}
            ]
        }]

        llama4_inputs = self.llama4_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 2. SAM用前処理（1024×1024）
        sam_image = self._preprocess_sam_image(image, target_size=1024)

        return {
            "llama4_inputs": llama4_inputs,
            "sam_image": sam_image,
            "seg_token_positions": self._find_seg_tokens(llama4_inputs["input_ids"])
        }

    def _preprocess_sam_image(self, image, target_size=1024):
        """SAM用画像前処理"""
        # アスペクト比保持リサイズ
        w, h = image.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        # パディング
        padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        padded_image.paste(image, (paste_x, paste_y))

        # テンソル変換
        image_array = np.array(padded_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        return image_tensor
```

### Phase 3: データセット適応 (2-3 日)

**目標:** HybridDataset の Llama-4 対応

#### 3.1 データセットクラス更新

```python
# utils/llama4_dataset.py
class Llama4HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir: str,
        llama4_processor: AutoProcessor,
        samples_per_epoch: int = 1000,
        **kwargs
    ):
        self.base_image_dir = base_image_dir
        self.processor = Llama4DualStreamProcessor(
            kwargs.get("model_id", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
        )

        # データセット構成（オリジナルLISAと同じ比率）
        self.dataset_config = {
            "sem_seg": kwargs.get("sample_rate", [9, 3, 3, 1])[0],
            "refer_seg": kwargs.get("sample_rate", [9, 3, 3, 1])[1],
            "vqa": kwargs.get("sample_rate", [9, 3, 3, 1])[2],
            "reason_seg": kwargs.get("sample_rate", [9, 3, 3, 1])[3]
        }

        self._load_datasets()

    def __getitem__(self, idx):
        # データセット選択
        dataset_type = self._select_dataset_type(idx)
        sample = self._get_raw_sample(dataset_type, idx)

        # デュアルストリーム前処理
        processed = self.processor.preprocess_dual_stream(
            sample["image_path"],
            sample["text_prompt"]
        )

        return {
            "llama4_pixel_values": processed["llama4_inputs"]["pixel_values"].squeeze(0),
            "input_ids": processed["llama4_inputs"]["input_ids"].squeeze(0),
            "attention_mask": processed["llama4_inputs"]["attention_mask"].squeeze(0),
            "sam_images": processed["sam_image"],
            "seg_token_positions": processed["seg_token_positions"],
            "ground_truth_mask": sample["mask"],
            "has_mask": sample["has_mask"],
            "dataset_type": dataset_type
        }
```

### Phase 4: 学習ループ実装 (2-3 日)

**目標:** Llama-4 対応学習ループ

#### 4.1 学習関数更新

```python
# train_llama4.py
def train_llama4_lisa():
    # モデル初期化
    model = LISALlama4ForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flex_attention",  # Llama-4推奨
        **lisa_config
    )

    # SEGトークン対応
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    if "[SEG]" not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_tokens(["[SEG]"])
        model.resize_token_embeddings(len(processor.tokenizer))

    # データセット
    dataset = Llama4HybridDataset(
        base_image_dir=config.DATASET_BASE_DIR,
        llama4_processor=processor,
        samples_per_epoch=config.SAMPLES_PER_EPOCH
    )

    # 学習ループ
    for epoch in range(config.NUM_EPOCHS):
        for batch in dataloader:
            # フォワードパス
            outputs = model(
                pixel_values=batch["llama4_pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sam_images=batch["sam_images"],
                seg_token_positions=batch["seg_token_positions"],
                ground_truth_masks=batch["ground_truth_mask"]
            )

            # 損失計算
            total_loss = (
                config.CE_LOSS_WEIGHT * outputs.ce_loss +
                config.DICE_LOSS_WEIGHT * outputs.dice_loss +
                config.BCE_LOSS_WEIGHT * outputs.bce_loss
            )

            # バックプロパゲーション
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### Phase 5: 検証スクリプト移植 (1-2 日)

**目標:** 全検証スクリプトの Llama-4 対応

#### 5.1 検証スクリプト更新

```python
# verify_llama4_model_architecture.py
def verify_llama4_lisa_architecture():
    """Llama4-LISAアーキテクチャ検証"""
    print("🔍 Llama4-LISAアーキテクチャ検証開始")

    # モデル初期化
    model = LISALlama4ForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flex_attention"
    )

    # アーキテクチャ検証
    assert hasattr(model, 'visual_model'), "SAMビジュアルモデルが存在しません"
    assert hasattr(model, 'text_hidden_fcs'), "プロジェクション層が存在しません"

    # パラメータ数確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    print(f"学習可能率: {trainable_params/total_params*100:.2f}%")

    # MoE構造確認
    moe_layers = [m for m in model.modules() if 'MoE' in type(m).__name__]
    print(f"MoEレイヤー数: {len(moe_layers)}")

    print("✅ アーキテクチャ検証完了")
```

---

## 🔄 デュアルエンコーダー問題への対応

### 問題の定義

- **Llama-4 内蔵ビジョン:** ネイティブマルチモーダル（解像度可変）
- **SAM ビジョンエンコーダー:** 1024×1024 固定
- **課題:** 異なる視覚世界の意味統合

### 解決戦略

#### 1. ビジョンパス分離戦略

```python
class DualVisionProcessor:
    def process_image(self, image, text_prompt):
        # パス1: Llama-4ネイティブ処理
        llama4_result = self.llama4_forward(image, text_prompt)

        # パス2: SAM特徴抽出
        sam_features = self.sam_encoder(image)

        # パス3: 特徴統合
        integrated_features = self.integrate_features(
            llama4_result.hidden_states,
            sam_features,
            seg_token_positions
        )

        return integrated_features
```

#### 2. 注意機構による統合

```python
class VisionIntegrationLayer(nn.Module):
    def __init__(self, llama4_dim, sam_dim):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llama4_dim,
            num_heads=8
        )
        self.sam_proj = nn.Linear(sam_dim, llama4_dim)

    def forward(self, llama4_features, sam_features, seg_mask):
        # SAM特徴をLlama-4次元に投影
        sam_proj = self.sam_proj(sam_features)

        # クロスアテンション
        attended_features, _ = self.cross_attention(
            query=llama4_features,
            key=sam_proj,
            value=sam_proj,
            key_padding_mask=~seg_mask
        )

        return attended_features
```

---

## ✅ 検証手順

### ステップ 1: 基本動作確認

```bash
python verify_llama4_model_architecture.py
python verify_llama4_input_formatting.py
python verify_llama4_dataset_integrity.py
```

### ステップ 2: 学習プロセス検証

```bash
python verify_llama4_loss_and_gradients.py
python overfit_llama4_single_batch.py
```

### ステップ 3: 推論検証

```bash
python test_llama4_inference_pipeline.py
```

### ステップ 4: フル学習実行

```bash
python train_llama4.py --config config_llama4.py
```

---

## 🚨 重要な注意事項

### 1. リソース要件

- **メモリ:** 最低 40GB VRAM (Llama-4-Scout 17B + SAM)
- **ストレージ:** ~200GB (モデル + データセット)
- **transformers:** ≥4.51.0 必須

### 2. トークナイザー互換性

- Gemma-3 (262K vocab) → Llama-4 (128K vocab)
- 既存の SEG トークン定義維持
- チャットテンプレート変更に注意

### 3. 学習安定性

- MoE モデルは学習不安定になりやすい
- 勾配クリッピング推奨
- 初期学習率を控えめに設定

### 4. 推論最適化

- flex_attention の活用
- バッチサイズ調整
- KV キャッシュ管理

---

## 📋 実装チェックリスト

### Phase 1: 基盤環境

- [ ] transformers 4.51.0 以上インストール
- [ ] config_llama4.py 作成
- [ ] 依存関係更新

### Phase 2: モデルアーキテクチャ

- [ ] llama4_lisa.py 実装
- [ ] デュアルストリーム処理実装
- [ ] プロジェクション層実装

### Phase 3: データセット

- [ ] Llama4HybridDataset 実装
- [ ] コレート関数更新
- [ ] データローダー動作確認

### Phase 4: 学習

- [ ] train_llama4.py 実装
- [ ] 損失関数統合
- [ ] オプティマイザー設定

### Phase 5: 検証

- [ ] 全検証スクリプト移植
- [ ] 推論パイプライン実装
- [ ] パフォーマンステスト

---

**最終更新:** 2025 年 6 月 20 日  
**作成者:** Claude-4  
**プロジェクト:** LISA-Llama4 移行プロジェクト
