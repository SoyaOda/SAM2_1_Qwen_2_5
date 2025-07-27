# LISA-Llama4 統合モデル プロジェクト概要

## プロジェクト説明
LISAアーキテクチャに基づいて、Llama-4-Scout-17B-16E-Instruct（109Bパラメータ、17B active）とSAM（Segment Anything Model）を統合したマルチモーダルモデルの実装。画像セグメンテーションと言語理解を統合し、自然言語指示に基づく画像セグメンテーションを実現。

## モデルアーキテクチャ

### 1. ベースモデル
- **言語モデル**: Llama-4-Scout-17B-16E-Instruct (meta-llama/Llama-4-Scout-17B-16E-Instruct)
  - 総パラメータ: 109B (MoE: 16 experts)
  - アクティブパラメータ: 17B per token
  - Hidden size: 5120
  - コンテキスト長: 128K tokens
- **ビジョンモデル**: SAM ViT-H
  - エンコーダ: ViT-Huge
  - プロンプト埋め込み次元: 256
  - 入力サイズ: 1024x1024

### 2. 統合アーキテクチャ（シングルエンコーダー構成）
```python
# model/llama4_lisa.py より抜粋
class LisaLlama4ForCausalLM(PreTrainedModel):
    def __init__(self, config):
        # Llama-4マルチモーダルモデル
        self.llama_model = Llama4ForConditionalGeneration.from_pretrained(...)
        
        # SAMビジョンエンコーダ（唯一の画像エンコーダ）
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
        
        # MLPプロジェクタ: Llama隠れ状態 → SAMプロンプト埋め込み
        self.projector = MultiModalProjector(
            llama_hidden_size=5120,
            sam_prompt_embed_dim=256
        )
```

### 3. 主要コンポーネント
- **MultiModalProjector**: LLM出力(5120) → SAM入力(256)への変換
- **シングルエンコーダー**: SAMのみで画像処理（Llama-4の画像エンコーダは使用しない）
- **[SEG]トークン**: セグメンテーション指示の特殊トークン

## 現在の学習設定

### 1. 基本パラメータ（config_linux.py）
```python
# 学習設定
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32  # 実効バッチサイズ = 32
EPOCHS = 10
STEPS_PER_EPOCH = 1000
MODEL_MAX_LENGTH = 131072

# LoRA設定
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"]

# 画像サイズ
LLAMA_IMAGE_SIZE = 448  # Llama-4のタイルサイズ（未使用）
SAM_IMAGE_SIZE = 1024   # SAMエンコーダ入力
```

### 2. 損失関数設定（model/losses.py）
```python
class CompositeLoss(nn.Module):
    def __init__(self):
        self.ce_loss_weight = 1.0      # テキスト生成損失
        self.dice_loss_weight = 0.5    # DICE損失（IoU類似）
        self.bce_loss_weight = 2.0     # Binary Cross Entropy損失
```

### 3. オプティマイザー設定
```python
# train_llama4_lisa_single_process.py より
optimizer = optim.AdamW(
    trainable_params,
    lr=0.0001,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

### 4. データセット構成（HybridDataset）
```python
dataset_string = 'Semantic_Segmentation||Referring_Segmentation||VQA||Reasoning_Segmentation'
sample_rate = [9, 3, 3, 1]  # サンプリング比率
```

## 技術的な特徴

### 1. Model Parallelism
- device_map="auto"でGPU自動分散
- 8 × A100 80GB GPUで109Bモデルを実行
- PEFT適用後のdevice_map保持（Web調査に基づく対策実装）

### 2. メモリ最適化
- LoRAによるパラメータ効率的学習
- bfloat16精度での学習
- 勾配累積による実効バッチサイズ増加

### 3. 現在の制限事項
- attn_implementation = "eager"（flex_attentionにバグあり）
- Flash Attention 2未使用
- Gradient Checkpointing未実装
- 8bit Optimizer未使用

## 質問内容

**このプロジェクトにおいて、バッチサイズや学習パラメータの選択、その他学習関連パラメータは現状で最適か、Webでリサーチして教えてください。Llama-4とSAMの統合モデルをLISAアーキテクチャに基づいて実装している前提で、2025年最新のベストプラクティスを調査してください。**

特に以下の点について：
1. 109Bモデルに対する現在のバッチサイズ設定（実効32）は適切か
2. 学習率0.0001は保守的すぎないか
3. LoRA設定（r=64, alpha=128）は十分か
4. Flash Attention 2やGradient Checkpointingを使用すべきか
5. マルチモーダル統合における損失関数の重み設定は適切か

## 参考情報
- GPU: 8 × NVIDIA A100 80GB (Lambda Cloud)
- フレームワーク: PyTorch 2.0+, Transformers, PEFT
- 学習データ: 4種類のデータセット（セグメンテーション3種 + VQA）
- 目的: 自然言語指示による高精度画像セグメンテーション