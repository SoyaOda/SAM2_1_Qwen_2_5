#!/bin/bash
# データセットサイズ確認スクリプト（別実行用）

set -e

# 設定
SOURCE_IP="129.213.165.134"
DEST_IP="192.222.53.46"
SSH_KEY="$HOME/.ssh/lambda_cloud_key"
SOURCE_PATH="/lambda/nfs/lisa-gemma-project-fs/data/dataset"
DEST_PATH="/lambda/nfs/llama4-lisa-project-fs-north-texas/data/dataset"

echo "📊 データセットサイズ確認開始"
echo "⏱️  これは時間がかかる場合があります（大容量データの場合は数分〜十数分）"

# 転送元サイズ確認
echo ""
echo "🔍 転送元サイズ計算中..."
echo "サーバー: $SOURCE_IP"
echo "パス: $SOURCE_PATH"
start_time=$(date +%s)

ssh -i $SSH_KEY ubuntu@$SOURCE_IP "du -sh $SOURCE_PATH 2>/dev/null || echo 'サイズ計算エラー'"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "✅ 転送元サイズ計算完了（所要時間: ${duration}秒）"

# 転送先サイズ確認（転送後）
echo ""
echo "🔍 転送先サイズ確認"
echo "サーバー: $DEST_IP"
echo "パス: $DEST_PATH"

ssh -i $SSH_KEY ubuntu@$DEST_IP "
if [ -d '$DEST_PATH' ]; then
    echo '転送先サイズ:'
    du -sh $DEST_PATH 2>/dev/null || echo 'サイズ計算中...'
else
    echo '転送先ディレクトリが存在しません: $DEST_PATH'
fi
"

# ディスク使用量詳細
echo ""
echo "💾 転送元ディスク使用量:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "df -h /lambda/nfs/lisa-gemma-project-fs/data"

echo ""
echo "💾 転送先ディスク使用量:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "df -h /lambda/nfs/llama4-lisa-project-fs-north-texas/data"

echo ""
echo "📋 サイズ確認完了"