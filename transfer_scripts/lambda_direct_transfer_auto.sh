#!/bin/bash
# Lambda Cloud間直接データ転送スクリプト（自動実行版）
# lisa-gemma-project-fs → llama4-lisa-project-fs-north-texas

set -e  # エラー時に停止

# 設定
SOURCE_IP="129.213.165.134"
DEST_IP="192.222.53.46"
SSH_KEY="~/.ssh/lambda_cloud_key"
SOURCE_PATH="/lambda/nfs/lisa-gemma-project-fs/data/dataset"
DEST_PATH="/lambda/nfs/llama4-lisa-project-fs-north-texas/data"

echo "🚀 Lambda Cloud間直接データ転送開始（自動実行版）"
echo "📁 転送元: $SOURCE_IP:$SOURCE_PATH"
echo "📁 転送先: $DEST_IP:$DEST_PATH"

# Step 1: 転送先ディレクトリ確認・作成
echo ""
echo "🔍 Step 1: 転送先ディレクトリ確認・作成"
ssh -i $SSH_KEY ubuntu@$DEST_IP "sudo mkdir -p $DEST_PATH && sudo chown -R ubuntu:ubuntu $DEST_PATH && ls -la $DEST_PATH"
echo "✅ 転送先ディレクトリ準備完了"

# Step 2: データサイズ確認
echo ""
echo "📊 Step 2: データサイズ確認"
echo "転送元サイズ:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "du -sh $SOURCE_PATH 2>/dev/null || echo 'サイズ計算中...'"
echo "転送先の空き容量:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "df -h $DEST_PATH"

# Step 3: 転送先でSSHキーを設定（インスタンス間直接転送用）
echo ""
echo "🔑 Step 3: 転送先でSSHキー設定"
# 転送先に一時的なSSHキーをコピー
scp -i $SSH_KEY $SSH_KEY ubuntu@$DEST_IP:~/temp_ssh_key
ssh -i $SSH_KEY ubuntu@$DEST_IP "chmod 600 ~/temp_ssh_key"

# Step 4: 自動転送実行（確認プロンプトなし）
echo ""
echo "🔄 Step 4: インスタンス間直接rsync転送開始（自動実行）"
echo "最適化オプション: -avz --progress --partial-dir=.rsync-partial"

ssh -i $SSH_KEY ubuntu@$DEST_IP "
rsync -avz --progress --partial-dir=.rsync-partial \
  -e 'ssh -i ~/temp_ssh_key -o StrictHostKeyChecking=no' \
  ubuntu@$SOURCE_IP:$SOURCE_PATH/ \
  $DEST_PATH/dataset/
"

echo "✅ 直接転送完了"

# Step 5: 転送検証
echo ""
echo "🔍 Step 5: 転送検証"
echo "転送元ディレクトリ構造:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "find $SOURCE_PATH -maxdepth 2 -type d 2>/dev/null | head -10"
echo ""
echo "転送先ディレクトリ構造:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "find $DEST_PATH/dataset -maxdepth 2 -type d 2>/dev/null | head -10"

# Step 6: 転送後のサイズ比較
echo ""
echo "📊 Step 6: 転送後のサイズ比較"
echo "転送元サイズ:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "du -sh $SOURCE_PATH 2>/dev/null || echo '計算中...'"
echo "転送先サイズ:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "du -sh $DEST_PATH/dataset 2>/dev/null || echo '計算中...'"

# Step 7: セキュリティ: 一時SSHキー削除
echo ""
echo "🔒 Step 7: セキュリティクリーンアップ"
ssh -i $SSH_KEY ubuntu@$DEST_IP "rm -f ~/temp_ssh_key"
echo "✅ 一時SSHキー削除完了"

echo ""
echo "🎉 Lambda Cloud間直接データ転送完了！"
echo "📁 転送先パス: $DEST_IP:$DEST_PATH/dataset"
echo ""
echo "📋 次のステップ:"
echo "1. config_linux.pyのDATASET_BASE_DIRを更新"
echo "   DATASET_BASE_DIR = '/lambda/nfs/llama4-lisa-project-fs-north-texas/data/dataset'"
echo "2. verify_dataset_integrity.pyで転送確認"