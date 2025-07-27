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

## ⚡ 開発用エイリアス（自動生成）

一括セットアップにより以下のエイリアスが自動生成されます：

```bash
# エイリアス設定ファイルを作成
cat > ~/.lisa_lambda_aliases << 'EOF'
# Lambda Cloud 開発用エイリアス（rsync版）
export LAMBDA_IP="YOUR_IP"  # ここにLambda CloudのIPを設定

# プロジェクト全体転送（高速同期）
alias lc-sync-all='rsync -avz --progress --delete --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" --exclude="lambda_results" --exclude="verification_output" --exclude="vis_output" --exclude=".gitignore" -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@$LAMBDA_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/'

# 個別ファイル転送
alias lc-sync='rsync -avz --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" --exclude=".gitignore" -e "ssh -i ~/.ssh/lambda_cloud_key"'

# Python実行
alias lc-run='ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python"'

# DeepSpeed実行
alias lc-deepspeed='ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && deepspeed"'

# GPU状況確認
alias lc-gpu='ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "nvidia-smi"'

# SSH接続
alias lc-connect='ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP'

# tmuxセッション管理
alias lc-tmux='ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "tmux new-session -d -s training || tmux attach-session -t training"'

# ログ確認
alias lc-logs='ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && tail -f *.log"'
EOF

# エイリアスを有効化
source ~/.lisa_lambda_aliases

# bashrcに永続追加
echo "source ~/.lisa_lambda_aliases" >> ~/.bashrc
```

### エイリアス使用例

```bash
# エイリアス設定でIPを設定後
export LAMBDA_IP="158.101.123.118"

# プロジェクト全体を高速同期
lc-sync-all

# 個別ファイル転送
lc-sync train_deepspeed_with_preprocessed.py ubuntu@$LAMBDA_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# DeepSpeed学習実行
lc-deepspeed train_deepspeed_with_preprocessed.py --preprocessed_checkpoint preprocessed_checkpoints/lisa_gemma_non_distributed_20250630_033545 --epochs 1 --steps_per_epoch 1 --dataset_type sem_seg --samples_per_epoch 1

# GPU状況確認
lc-gpu
```

## 📝 実践的な開発例

### 例 1: 検証スクリプトの開発・実行

```bash
# 1. ローカルでverify_loss_and_gradients.pyを編集
# VSCode、Cursor等で編集

# 2. プロジェクト全体を高速同期
lc-sync-all

# 3. 実行
lc-run verify_loss_and_gradients.py
```

### 例 2: 事前処理チェックポイント作成

```bash
# 1. 改良版事前処理チェックポイント作成スクリプトを同期
lc-sync-all

# 2. 実行
lc-run create_non_distributed_checkpoint.py
```

### 例 3: DeepSpeed 学習実行

```bash
# 1. 学習スクリプトを同期
lc-sync-all

# 2. tmuxセッションで長時間実行
lc-tmux
# tmux内で: deepspeed train_deepspeed_with_preprocessed.py --preprocessed_checkpoint preprocessed_checkpoints/lisa_gemma_non_distributed_20250630_033545 --epochs 1 --steps_per_epoch 1 --dataset_type sem_seg --samples_per_epoch 1
```

### 例 4: 緊急修正（個別ファイル転送）

```bash
# 重要な修正を単一ファイルで緊急転送
lc-sync model/gemma_lisa.py ubuntu@$LAMBDA_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/model/

# すぐに実行
lc-deepspeed train_deepspeed_with_preprocessed.py --preprocessed_checkpoint preprocessed_checkpoints/lisa_gemma_non_distributed_20250630_033545 --epochs 1 --steps_per_epoch 1 --dataset_type sem_seg --samples_per_epoch 1
```

## 🔧 rsync コマンド詳細

### rsync オプション説明

- `-a`: アーカイブモード（パーミッション、タイムスタンプ保持）
- `-v`: 詳細情報表示
- `-z`: 転送時圧縮
- `--progress`: 転送進捗表示
- `--delete`: 削除されたファイルを同期先でも削除
- `--exclude`: 除外パターン指定
- `-e "ssh -i ~/.ssh/lambda_cloud_key"`: SSH 鍵指定

### よく使う rsync コマンド集

```bash
# 1. プロジェクト全体転送（初回・大幅変更時）
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 2. 高速同期転送（日常開発）- チェックポイント保護版
rsync -avz --progress --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='preprocessed_checkpoints' --exclude='weights' --exclude='checkpoints' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 3. 個別ファイル転送（緊急修正）
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" your_file.py ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 4. 特定ディレクトリ転送
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -i ~/.ssh/lambda_cloud_key" model/ ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/model/

# 5. 検証スクリプト一括転送
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -i ~/.ssh/lambda_cloud_key" verify_*.py ubuntu@YOUR_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
```

## 🔧 よく使うコマンド集

### 監視・確認パターン

```bash
# GPU使用状況
lc-gpu

# プロセス確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "ps aux | grep python"

# ディスク使用量
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "df -h"

# tmuxセッション一覧
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "tmux list-sessions"
```

### デバッグパターン

```bash
# ログ確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && tail -f output.log"

# Python環境確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python --version && pip list | grep torch"

# CUDA確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')\""
```

## 🚨 重要な注意点

### 必須の環境変数（自動設定済み）

エイリアス使用時、以下の環境変数が自動設定されます：

- `TF_CPP_MIN_LOG_LEVEL=3`: TensorFlow 警告を抑制
- `TF_ENABLE_ONEDNN_OPTS=0`: TensorFlow パフォーマンス最適化

### A100\*8 GPU 制約

- VRAM: 80GB × 8 = 640GB 総容量
- 分散学習に最適化
- メモリ不足時は`torch.cuda.empty_cache()`が自動実行

### ファイルパス（自動設定済み）

- プロジェクトルート: `/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/`
- Python 環境: `/lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/`
- データセット: `/lambda/nfs/lisa-gemma-project-fs/data/`

### 除外ファイル（チェックポイント保護）

rsync で転送**しない**ファイル・ディレクトリ：

- `.git/` - Git リポジトリ情報
- `__pycache__/` - Python キャッシュ
- `*.pyc` - Python バイトコード
- `lambda_results/` - ローカル実行結果
- `verification_output/` - 検証結果
- `vis_output/` - 可視化結果
- **`preprocessed_checkpoints/`** - 事前処理済みチェックポイント（**保護対象**）
- **`weights/`** - 学習済み重み（**保護対象**）
- **`checkpoints/`** - 一般チェックポイント（**保護対象**）
- `.gitignore` - Git 設定ファイル

**🛡️ チェックポイント保護の重要性:**

- Lambda Cloud 上で作成されたチェックポイントを`--delete`オプションから保護
- `create_non_distributed_checkpoint.py`等で生成されたデータを保持
- 学習済みモデルの誤削除を防止

## 🎯 トラブルシューティング

### セットアップ失敗時

```bash
# 個別認証の再実行
python lambda_quick_setup.py --ip YOUR_IP --skip-wandb  # HuggingFaceのみ
```

### エイリアスが効かない時

```bash
# エイリアス再読み込み
source ~/.lisa_lambda_aliases

# bashrcに永続追加
echo "source ~/.lisa_lambda_aliases" >> ~/.bashrc
```

### GPU 認識しない時

```bash
# NVIDIA設定確認
lc-gpu
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python -c \"import torch; print(torch.cuda.is_available())\""
```

### rsync 転送エラー時

```bash
# SSH接続確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "echo 'Connection OK'"

# パーミッション確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "ls -la /lambda/nfs/lisa-gemma-project-fs/"

# 手動でディレクトリ作成
ssh -i ~/.ssh/lambda_cloud_key ubuntu@$LAMBDA_IP "mkdir -p /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux"
```

### 🎉 分散学習（DeepSpeed）WandB 認証 - 自動対応完了！

**✅ 2025 年 7 月 1 日更新: WandB 認証が完全自動化されました！**

```bash
# 🚀 一括セットアップで全て完了
python lambda_quick_setup.py --ip YOUR_LAMBDA_IP

# ✅ DeepSpeed実行 - 追加認証不要！
lc-deepspeed train_deepspeed_with_preprocessed.py --preprocessed_checkpoint YOUR_CHECKPOINT --epochs 1 --steps_per_epoch 1 --dataset_type sem_seg --samples_per_epoch 1
```

**📝 新しい WandB 認証システムの特徴:**

- ✅ **完全自動化**: 一括セットアップで分散学習対応の WandB 認証完了
- ✅ **DeepSpeed 対応**: プロジェクトディレクトリ+仮想環境での認証実行
- ✅ **ファイルベース**: `wandb_api_key.txt`を自動転送し、確実な認証
- ✅ **追加手順不要**: 分散学習実行前の手動認証が不要
- ✅ **フォールバック**: 失敗時は環境変数方式で自動再試行

**📋 トークン取得方法（初回のみ）:**

1. **wandb_api_key.txt ファイル作成**（推奨）:

   ```bash
   echo "your_wandb_api_key" > wandb_api_key.txt
   ```

2. **ローカル wandb login**:

   ```bash
   wandb login
   ```

3. **環境変数設定**:
   ```bash
   export WANDB_API_KEY=your_wandb_api_key
   ```

---

**🎉 このガイドで 95%の開発作業をカバーできます。迷ったらプロジェクト全体同期から始めてください。**

## 📋 クイックリファレンス

```bash
# 🚀 セットアップ
python lambda_quick_setup.py --ip YOUR_IP

# 📁 プロジェクト全体転送（推奨）
lc-sync-all

# 📁 個別ファイル転送
lc-sync file.py ubuntu@$LAMBDA_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 🏃 Python実行
lc-run file.py

# 🔥 DeepSpeed実行
lc-deepspeed train_script.py

# 🖥️ GPU確認
lc-gpu

# 🔗 接続
lc-connect

# �� tmux
lc-tmux
```
