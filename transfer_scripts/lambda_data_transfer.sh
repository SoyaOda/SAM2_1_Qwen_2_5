#!/bin/bash
# Lambda Cloud間データ転送スクリプト
# lisa-gemma-project-fs → llama4-lisa-project-fs-north-texas

set -e  # エラー時に停止

# 設定
SOURCE_IP="129.213.165.134"
DEST_IP="192.222.53.46"
SSH_KEY="~/.ssh/lambda_cloud_key"
SOURCE_PATH="/lambda/nfs/lisa-gemma-project-fs/data/dataset"
DEST_PATH="/lambda/nfs/llama4-lisa-project-fs-north-texas/data"

echo "🚀 Lambda Cloud間データ転送開始"
echo "📁 転送元: $SOURCE_IP:$SOURCE_PATH"
echo "📁 転送先: $DEST_IP:$DEST_PATH"

# Step 1: 転送先ディレクトリ確認・作成
echo ""
echo "🔍 Step 1: 転送先ディレクトリ確認・作成"
ssh -i $SSH_KEY ubuntu@$DEST_IP "sudo mkdir -p $DEST_PATH && sudo chown -R ubuntu:ubuntu $DEST_PATH"
echo "✅ 転送先ディレクトリ準備完了"

# Step 2: データサイズ確認
echo ""
echo "📊 Step 2: データサイズ確認"
echo "転送元サイズ:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "du -sh $SOURCE_PATH"
echo "転送先の空き容量:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "df -h $DEST_PATH"

# Step 3: 転送実行確認
echo ""
read -p "転送を実行しますか？ (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "❌ 転送をキャンセルしました"
    exit 1
fi

# Step 4: rsync経由データ転送（最適化オプション付き）
echo ""
echo "🔄 Step 4: rsync経由データ転送開始"
echo "最適化オプション: -avz --progress --partial-dir=.rsync-partial"

# rsyncコマンド実行（中間転送経由）
rsync -avz --progress --partial-dir=.rsync-partial \
  -e "ssh -i $SSH_KEY" \
  ubuntu@$SOURCE_IP:$SOURCE_PATH/ \
  ./temp_dataset_transfer/

echo "✅ ローカル中間転送完了"

# 転送先へ送信
rsync -avz --progress --partial-dir=.rsync-partial \
  -e "ssh -i $SSH_KEY" \
  ./temp_dataset_transfer/ \
  ubuntu@$DEST_IP:$DEST_PATH/dataset/

echo "✅ 転送先への送信完了"

# Step 5: 転送検証
echo ""
echo "🔍 Step 5: 転送検証"
echo "転送元ディレクトリ構造:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "find $SOURCE_PATH -maxdepth 2 -type d | head -10"
echo ""
echo "転送先ディレクトリ構造:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "find $DEST_PATH/dataset -maxdepth 2 -type d | head -10"

# Step 6: 転送後のサイズ比較
echo ""
echo "📊 Step 6: 転送後のサイズ比較"
echo "転送元サイズ:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "du -sh $SOURCE_PATH"
echo "転送先サイズ:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "du -sh $DEST_PATH/dataset"

# Step 7: 中間ファイル削除
echo ""
echo "🧹 Step 7: 中間ファイル削除"
rm -rf ./temp_dataset_transfer/
echo "✅ 中間ファイル削除完了"

echo ""
echo "🎉 Lambda Cloud間データ転送完了！"
echo "📁 転送先パス: $DEST_IP:$DEST_PATH/dataset"
echo ""
echo "📋 次のステップ:"
echo "1. config_linux.pyのDATASET_BASE_DIRを更新"
echo "2. verify_dataset_integrity.pyで転送確認"