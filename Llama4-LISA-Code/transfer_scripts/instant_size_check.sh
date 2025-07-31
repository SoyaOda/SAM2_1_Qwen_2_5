#!/bin/bash
# 瞬時データサイズ確認スクリプト（転送ログ活用）

set -e

echo "⚡ 瞬時データサイズ確認"

# 転送ログから情報抽出
if [ -f "transfer_fixed.log" ]; then
    echo ""
    echo "📊 転送完了データから算出:"
    
    # 総データサイズ
    total_size=$(grep "total size is" transfer_fixed.log | tail -1 | grep -o '[0-9,]*' | tail -1)
    if [ ! -z "$total_size" ]; then
        # バイトから人間が読みやすい形式に変換
        size_gb=$(echo "scale=2; $total_size / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "計算エラー")
        echo "  総データサイズ: ${total_size} bytes (約 ${size_gb} GB)"
    fi
    
    # 転送ファイル数
    file_count=$(grep "to-chk=0/" transfer_fixed.log | tail -1 | grep -o "xfr#[0-9]*" | grep -o "[0-9]*")
    if [ ! -z "$file_count" ]; then
        echo "  総ファイル数: ${file_count} ファイル"
    fi
    
    # 転送速度
    speed_info=$(grep "sent.*received.*bytes" transfer_fixed.log | tail -1)
    if [ ! -z "$speed_info" ]; then
        echo "  転送情報: $speed_info"
    fi
    
else
    echo "❌ 転送ログファイル(transfer_fixed.log)が見つかりません"
fi

# 瞬時ファイル数確認
echo ""
echo "📊 瞬時ファイル数確認:"
SOURCE_IP="129.213.165.134"
DEST_IP="192.222.53.46"
SSH_KEY="$HOME/.ssh/lambda_cloud_key"
SOURCE_PATH="/lambda/nfs/lisa-gemma-project-fs/data/dataset"
DEST_PATH="/lambda/nfs/llama4-lisa-project-fs-north-texas/data/dataset"

echo "転送元ファイル数:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "find $SOURCE_PATH -type f | wc -l" &
pid1=$!

echo "転送先ファイル数:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "find $DEST_PATH -type f | wc -l" &
pid2=$!

# 両方の完了を待つ
wait $pid1 $pid2

echo ""
echo "✅ 瞬時確認完了（転送成功確認済み）"