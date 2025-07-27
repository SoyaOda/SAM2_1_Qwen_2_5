# LISA-Gemma3 Linux環境セットアップガイド

## 概要

このガイドでは、Windows環境で開発されたLISA-Gemma3プロジェクトをLinux環境に移行するための手順を説明します。

## 前提条件

- Linux環境（Ubuntu 18.04以降推奨）
- Python 3.8以降
- NVIDIA GPU（推奨）とCUDAドライバー
- 十分なディスク容量（データセット + モデルで50GB以上）

## クイックスタート

### 1. 自動セットアップ（推奨）

```bash
# リポジトリをクローン（既に完了している場合はスキップ）
git clone https://github.com/SoyaOda/LISA-Gemma1-new.git
cd LISA-Gemma1-new

# 自動セットアップスクリプトを実行
./setup_linux_environment.sh
```

### 2. 手動セットアップ

#### 2.1 必要パッケージのインストール

```bash
# PyTorchとCUDA（GPUを使用する場合）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# または、Linux用requirements.txtを使用
pip install -r requirements_linux.txt
```

#### 2.2 設定ファイルの調整

`config_linux.py`を編集して、あなたの環境に合わせてパスを設定：

```python
# データセットのベースディレクトリ
DATASET_BASE_DIR = "/path/to/your/LISA-dataset/dataset"

# SAMモデルのパス  
SAM_MODEL_PATH = "/path/to/your/sam_vit_h_4b8939.pth"
```

## 必要なファイルとデータ

### 1. SAMモデルファイル

元の場所: `C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth`
新しい場所: `/mnt/models/sam_vit_h_4b8939.pth`（設定ファイルで変更可能）

### 2. データセットファイル

元の場所: `H:/download/LISA-dataset/dataset`
新しい場所: `/mnt/data/LISA-dataset/dataset`（設定ファイルで変更可能）

期待されるディレクトリ構造:
```
/mnt/data/LISA-dataset/dataset/
├── ade20k/
├── cocostuff/
├── mapillary/
├── refer_seg/
│   └── images/
│       ├── mscoco/
│       └── saiapr_tc-12/
├── llava_dataset/
├── reason_seg/
│   └── ReasonSeg/
└── coco/
```

## 環境確認

### システム要件の確認

```bash
# 環境情報の確認
python config_linux.py

# GPU状況の確認
nvidia-smi

# PyTorchでのGPU認識確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 学習の実行

### デバッグモードでのテスト

```bash
# 少量のデータでテスト実行
python train_ds.py --debug --debug_samples 5

# 全データセットタイプでテスト実行  
python train_full.py --debug --debug_samples 5
```

### 本格的な学習

```bash
# 基本的な学習
python train_ds.py

# 全データセットでの学習
python train_full.py

# DeepSpeedを使用した分散学習
deepspeed --num_gpus=1 train_ds.py --deepspeed_config ds_config.json
```

## トラブルシューティング

### 1. パス関連のエラー

**エラー**: `FileNotFoundError: SAMモデルまたはデータセットが見つからない`

**解決方法**:
1. `config_linux.py`でパスを確認・修正
2. ファイルが実際に存在することを確認
3. ファイルの権限を確認

### 2. GPU関連のエラー

**エラー**: `CUDA out of memory`

**解決方法**:
- `--batch_size` を小さくする（デフォルト: 2）
- `--load_in_4bit` フラグを使用
- `--gradient_checkpointing` を有効にする

### 3. パッケージ関連のエラー

**エラー**: `ModuleNotFoundError`

**解決方法**:
```bash
# 不足パッケージの個別インストール
pip install [missing_package]

# または、全体の再インストール
pip install -r requirements_linux.txt
```

## 最適化のヒント

### 1. メモリ使用量の削減

```bash
# 4bit量子化を使用
python train_ds.py --load_in_4bit

# バッチサイズを調整
python train_ds.py --batch_size 1 --grad_accumulation_steps 16
```

### 2. 高速化

```bash
# flash-attnを使用（事前にインストールが必要）
pip install flash-attn --no-build-isolation

# より多くのワーカーを使用
python train_ds.py --workers 8
```

## 設定ファイルの詳細

### config_linux.py の主要設定

```python
# データセットパス（必須）
DATASET_BASE_DIR = "/mnt/data/LISA-dataset/dataset"
SAM_MODEL_PATH = "/mnt/models/sam_vit_h_4b8939.pth"

# モデル設定
DEFAULT_GEMMA_MODEL = "google/gemma-3-4b-it"
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 0.0003

# LoRA設定
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
```

## Windows環境からの主な変更点

1. **パス形式**: `C:/` や `H:/` → `/mnt/` や `/home/`
2. **パッケージ**: Windows専用パッケージを除外
3. **パス区切り文字**: `\\` → `/`
4. **実行権限**: シェルスクリプトに実行権限を付与

## 関連ファイル

- `config_linux.py` - Linux環境用設定
- `requirements_linux.txt` - Linux用パッケージリスト
- `setup_linux_environment.sh` - 自動セットアップスクリプト
- `train_ds.py` - メイン学習スクリプト（Linux対応済み）
- `train_full.py` - フル学習スクリプト（Linux対応済み）

## サポート

問題が発生した場合は、以下を確認してください：

1. `config_linux.py`の設定が正しいか
2. 必要なファイルが存在するか  
3. 権限設定が適切か
4. GPUドライバーとCUDAが正しくインストールされているか

詳細なログは各学習スクリプトの実行時に確認できます。 