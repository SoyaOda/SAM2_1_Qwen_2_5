# Lambda Cloud 開発クイックガイド (rsync 版)

## 🎯 核心の開発フロー

### 基本パターン（rsync ベース）

```bash
# 1. ローカルでファイル編集
# お好みのエディタで編集

# 2. Lambda Cloudにrsync転送
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 3. Lambda Cloud上で実行
ssh -i ~/.ssh/lambda_cloud_key ubuntu@YOUR_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python your_file.py"
```

## 🚀 GPU 借用後の初期セットアップ

### 1. Lambda Cloud GPU インスタンス起動

- **8x NVIDIA A100 (80GB)** を選択（推奨）
- **重要**: `lisa-gemma-project-fs` Persistent Filesystem を必ずアタッチ
- IP アドレスをメモ（例: `158.101.123.118`）

### 2. 🎉 一括セットアップ実行（新方式）

**たった 1 つのコマンドで全て完了！**

```bash
# 一括セットアップ実行
python lambda_quick_setup.py --ip YOUR_LAMBDA_IP
```

このコマンドが自動実行する内容：

- ✅ SSH 接続確認
- ✅ Python 環境 & A100\*8 GPU 確認
- ✅ HuggingFace 認証設定
- ✅ WandB 認証設定
- ✅ プロジェクト環境設定
- ✅ 開発用エイリアス自動生成

### 3. 認証設定（事前準備）

**一括セットアップを実行する前に、ローカルで認証を完了してください：**

```bash
# HuggingFace認証（必須）
huggingface-cli login

# WandB認証（必須）
wandb login
```

## 📁 プロジェクト全体転送（推奨方法）

### 初回セットアップ: Lambda 上でバックアップ作成

```bash
# 1. Lambda上の既存codeフォルダをバックアップ
ssh -i ~/.ssh/lambda_cloud_key ubuntu@YOUR_IP "cd /lambda/nfs/lisa-gemma-project-fs && mv code code_backup_$(date +%Y%m%d_%H%M%S)"

# 2. 新しいcodeフォルダを作成
ssh -i ~/.ssh/lambda_cloud_key ubuntu@YOUR_IP "mkdir -p /lambda/nfs/lisa-gemma-project-fs/code"

# 3. ローカルのプロジェクト全体を転送
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
```

### 通常の開発サイクル: 高速同期転送

```bash
# プロジェクト全体の高速同期（変更分のみ転送）
rsync -avz --progress --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
```
