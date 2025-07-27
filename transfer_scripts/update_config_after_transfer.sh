#!/bin/bash
# 転送後の設定ファイル更新スクリプト

set -e

DEST_IP="192.222.53.46"
SSH_KEY="$HOME/.ssh/lambda_cloud_key"

echo "🔧 転送後の設定ファイル更新開始"

# Step 1: config_linux.pyをバックアップ
echo ""
echo "📋 Step 1: config_linux.pyバックアップ作成"
ssh -i $SSH_KEY ubuntu@$DEST_IP "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && cp config_linux.py config_linux.py.backup"
echo "✅ バックアップ作成完了"

# Step 2: 設定ファイル更新
echo ""
echo "🔧 Step 2: DATASET_BASE_DIR更新"
ssh -i $SSH_KEY ubuntu@$DEST_IP "
cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux
sed -i 's|/lambda/nfs/lisa-gemma-project-fs/data/dataset|/lambda/nfs/llama4-lisa-project-fs-north-texas/data/dataset|g' config_linux.py
"
echo "✅ DATASET_BASE_DIR更新完了"

# Step 3: SAMチェックポイントパスも更新（必要に応じて）
echo ""
echo "🔧 Step 3: SAMチェックポイントパス確認・更新"
ssh -i $SSH_KEY ubuntu@$DEST_IP "
cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux
# SAMチェックポイントファイルが存在するか確認
if [ ! -f '/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth' ]; then
    echo '⚠️ SAMチェックポイントが見つかりません。パスを更新します。'
    # 新しいパスに更新
    sed -i 's|/lambda/nfs/lisa-gemma-project-fs/data/weights|/lambda/nfs/llama4-lisa-project-fs-north-texas/data/weights|g' config_linux.py
    # weightディレクトリを作成
    sudo mkdir -p /lambda/nfs/llama4-lisa-project-fs-north-texas/data/weights
    sudo chown -R ubuntu:ubuntu /lambda/nfs/llama4-lisa-project-fs-north-texas/data/weights
    echo '📁 weightsディレクトリ作成完了'
else
    echo '✅ SAMチェックポイントが既存パスに存在します'
fi
"

# Step 4: 変更確認
echo ""
echo "🔍 Step 4: 設定変更確認"
ssh -i $SSH_KEY ubuntu@$DEST_IP "
cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux
echo '変更されたDATASET_BASE_DIR:'
grep 'DATASET_BASE_DIR' config_linux.py
echo ''
echo '変更されたSAM_CHECKPOINT_PATH:'
grep 'SAM_CHECKPOINT_PATH' config_linux.py
"

echo ""
echo "✅ 設定ファイル更新完了！"
echo ""
echo "📋 次のステップ:"
echo "1. verify_dataset_integrity.pyで転送確認:"
echo "   ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.53.46"
echo "   cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux"
echo "   source ../../venvs/lisa_gemma_venv/bin/activate"
echo "   python verify_dataset_integrity.py"