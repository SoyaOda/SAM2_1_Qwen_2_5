#!/bin/bash
# 高速データサイズ確認スクリプト

set -e

# 設定
SOURCE_IP="129.213.165.134"
DEST_IP="192.222.53.46"
SSH_KEY="$HOME/.ssh/lambda_cloud_key"
SOURCE_PATH="/lambda/nfs/lisa-gemma-project-fs/data/dataset"
DEST_PATH="/lambda/nfs/llama4-lisa-project-fs-north-texas/data/dataset"

echo "⚡ 高速データサイズ確認開始"

# 方法1: ファイル数による概算（最高速）
echo ""
echo "📊 方法1: ファイル数比較（数秒で完了）"
echo "転送元ファイル数:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "find $SOURCE_PATH -type f | wc -l"
echo "転送先ファイル数:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "find $DEST_PATH -type f | wc -l"

# 方法2: --apparent-size使用（高速）
echo ""
echo "📊 方法2: 高速サイズ計算（--apparent-size）"
echo "転送元サイズ（高速）:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "du -sh --apparent-size $SOURCE_PATH 2>/dev/null || echo 'エラー'"
echo "転送先サイズ（高速）:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "du -sh --apparent-size $DEST_PATH 2>/dev/null || echo 'エラー'"

# 方法3: 主要ディレクトリ別サイズ（詳細確認用）
echo ""
echo "📊 方法3: 主要ディレクトリ別サイズ"
echo "転送元 - 主要ディレクトリ:"
ssh -i $SSH_KEY ubuntu@$SOURCE_IP "
cd $SOURCE_PATH 2>/dev/null && \
for dir in */; do 
    if [ -d \"\$dir\" ]; then
        size=\$(du -sh --apparent-size \"\$dir\" 2>/dev/null | cut -f1)
        printf \"  %-20s %s\n\" \"\$dir\" \"\$size\"
    fi
done
"

echo "転送先 - 主要ディレクトリ:"
ssh -i $SSH_KEY ubuntu@$DEST_IP "
cd $DEST_PATH 2>/dev/null && \
for dir in */; do 
    if [ -d \"\$dir\" ]; then
        size=\$(du -sh --apparent-size \"\$dir\" 2>/dev/null | cut -f1)
        printf \"  %-20s %s\n\" \"\$dir\" \"\$size\"
    fi
done
"

# 方法4: rsyncの転送ログから容量取得（既に完了した転送の場合）
echo ""
echo "📊 方法4: 転送ログからの容量情報"
if [ -f "transfer_fixed.log" ]; then
    echo "転送ログから総サイズ抽出:"
    grep "total size is" transfer_fixed.log | tail -1
    grep "sent.*received.*bytes" transfer_fixed.log | tail -1
else
    echo "転送ログファイルが見つかりません"
fi

echo ""
echo "✅ 高速サイズ確認完了"