#!/bin/bash

# LISA-Gemma3 Linux環境セットアップスクリプト

echo "=== LISA-Gemma3 Linux環境セットアップ開始 ==="

# エラー時にスクリプトを停止
set -e

# カラー出力用の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. システム情報を確認
print_info "システム情報を確認中..."
echo "OS: $(uname -s)"
echo "カーネル: $(uname -r)"
echo "アーキテクチャ: $(uname -m)"
echo "Python: $(python3 --version 2>/dev/null || echo 'Python3が見つかりません')"

# 2. 必要なディレクトリを作成
print_info "必要なディレクトリを作成中..."
mkdir -p /tmp/lisa-setup
mkdir -p ./logs
mkdir -p ./runs
mkdir -p ./checkpoints

# 3. 仮想環境の確認または作成
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warning "仮想環境が検出されませんでした"
    read -p "新しい仮想環境を作成しますか？ (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "仮想環境 'lisa-gemma3' を作成中..."
        python3 -m venv lisa-gemma3
        source lisa-gemma3/bin/activate
        print_info "仮想環境をアクティベートしました"
    fi
else
    print_info "既存の仮想環境を使用: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}"
fi

# 4. Pythonパッケージの更新
print_info "pipを最新版に更新中..."
pip install --upgrade pip

# 5. PyTorchのインストール
print_info "PyTorchをインストール中..."
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPUが検出されました。CUDA版PyTorchをインストールします..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_warning "NVIDIA GPUが検出されませんでした。CPU版PyTorchをインストールします..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 6. その他の必要なパッケージをインストール
print_info "追加パッケージをインストール中..."
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install peft>=0.4.0
pip install datasets>=2.10.0
pip install tokenizers>=0.13.0

# 7. DeepSpeedのインストール
print_info "DeepSpeedをインストール中..."
pip install deepspeed>=0.10.0

# 8. 科学計算・画像処理ライブラリ
print_info "科学計算・画像処理ライブラリをインストール中..."
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install opencv-python>=4.7.0
pip install scikit-image>=0.20.0
pip install Pillow>=9.4.0
pip install matplotlib>=3.7.0

# 9. 機械学習関連ライブラリ
print_info "機械学習関連ライブラリをインストール中..."
pip install pandas>=2.0.0
pip install tqdm>=4.64.0
pip install tensorboard>=2.13.0
pip install einops>=0.4.1

# 10. セグメンテーション関連
print_info "セグメンテーション関連ライブラリをインストール中..."
pip install segment-anything>=1.0
pip install pycocotools>=2.0.0

# 11. 量子化サポート
print_info "量子化サポートライブラリをインストール中..."
pip install bitsandbytes>=0.41.0

# 12. オプションライブラリ
read -p "開発用ライブラリもインストールしますか？ (jupyter, gradio, etc.) (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "開発用ライブラリをインストール中..."
    pip install jupyter>=1.0.0
    pip install gradio>=4.0.0
    pip install fastapi>=0.100.0
    pip install uvicorn>=0.23.0
fi

# 13. flash-attnのインストール（オプション）
read -p "flash-attnをインストールしますか？（高速化されますが、コンパイル時間が長いです） (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "flash-attnをインストール中（時間がかかる場合があります）..."
    pip install flash-attn>=2.0.0 --no-build-isolation || print_warning "flash-attnのインストールに失敗しました（スキップ）"
fi

# 14. 環境設定の確認
print_info "インストール状況を確認中..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 15. 設定ファイルの確認
print_info "設定ファイルを確認中..."
if [ -f "config_linux.py" ]; then
    python config_linux.py
else
    print_warning "config_linux.pyが見つかりません"
fi

# 16. データセットとモデルのパス確認
print_info "重要なパスの確認..."
print_warning "以下のパスを環境に合わせて設定してください:"
echo "  - SAMモデル: /mnt/models/sam_vit_h_4b8939.pth"
echo "  - データセット: /mnt/data/LISA-dataset/dataset"
echo ""
echo "これらのパスは config_linux.py で変更できます"

# 17. 使用方法の表示
print_info "セットアップ完了！"
echo ""
echo "=== 使用方法 ==="
echo "1. データセットとモデルファイルを適切な場所に配置"
echo "2. config_linux.py でパスを調整"
echo "3. 学習開始:"
echo "   python train_ds.py --debug"
echo "   python train_full.py --debug"
echo ""
echo "=== 追加設定が必要な場合 ==="
echo "1. config_linux.py を編集してパスを修正"
echo "2. requirements_linux.txt を使用:"
echo "   pip install -r requirements_linux.txt"

print_info "セットアップスクリプト完了" 