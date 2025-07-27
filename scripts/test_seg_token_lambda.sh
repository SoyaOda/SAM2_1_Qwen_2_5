#!/bin/bash
# scripts/test_seg_token_lambda.sh
# Lambda Cloud環境での[SEG]トークン動作確認スクリプト

echo "=== Sa2VA風[SEG]トークン Lambda Cloud動作確認 ==="
echo "Phase 3C - Phase 1: 軽量版[SEG]トークン実装"
echo "================================================"

# Lambda Cloud環境設定
LAMBDA_HOST="ubuntu@192.222.53.149"
LAMBDA_KEY="~/.ssh/lambda_cloud_key"
REMOTE_DIR="/lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux"

# 1. ローカルファイルの転送
echo "📤 ステップ1: ファイル転送中..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='lambda_results' \
    --exclude='verification_output' \
    --exclude='vis_output' \
    --exclude='.gitignore' \
    -e "ssh -i ${LAMBDA_KEY}" \
    ./ ${LAMBDA_HOST}:${REMOTE_DIR}/

# 2. ユニットテスト実行
echo ""
echo "🧪 ステップ2: ユニットテスト実行..."
ssh -i ${LAMBDA_KEY} ${LAMBDA_HOST} << 'EOF'
cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux
source ../../venvs/lisa_gemma_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=== [SEG]トークン生成器ユニットテスト ==="
python -u tests/test_seg_token_generator.py
EOF

# 3. 統合テスト実行（Phase 3B + [SEG]トークン）
echo ""
echo "🚀 ステップ3: Phase 3B統合テスト（[SEG]トークン有効）..."
ssh -i ${LAMBDA_KEY} ${LAMBDA_HOST} << 'EOF'
cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux
source ../../venvs/lisa_gemma_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Phase 3B統合テスト（[SEG]トークン有効） ==="
timeout 600 python -u test_phase3b_integration_real.py 2>&1 | head -n 500
EOF

# 4. 性能比較用: [SEG]トークン無効化テスト
echo ""
echo "📊 ステップ4: ベースライン測定（[SEG]トークン無効）..."
ssh -i ${LAMBDA_KEY} ${LAMBDA_HOST} << 'EOF'
cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux
source ../../venvs/lisa_gemma_venv/bin/activate

# config_linux.pyを一時的に編集して[SEG]トークンを無効化
cp config_linux.py config_linux.py.bak
sed -i 's/USE_SEG_TOKEN = True/USE_SEG_TOKEN = False/' config_linux.py

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Phase 3B統合テスト（[SEG]トークン無効） ==="
timeout 300 python -u test_phase3b_integration_real.py 2>&1 | grep -E "(性能評価|改善率|inference_time|success)" | head -n 20

# 設定を元に戻す
mv config_linux.py.bak config_linux.py
EOF

# 5. 結果の取得
echo ""
echo "📥 ステップ5: 結果ファイル取得..."
rsync -avz --progress \
    -e "ssh -i ${LAMBDA_KEY}" \
    ${LAMBDA_HOST}:${REMOTE_DIR}/phase3b_real_test_results/ \
    ./lambda_results/seg_token_test/

echo ""
echo "✅ Lambda Cloud動作確認完了"
echo "結果は ./lambda_results/seg_token_test/ に保存されました"
echo ""
echo "📊 期待される改善:"
echo "  - Phase 1（軽量[SEG]トークン）: 5-8%"
echo "  - Phase 2（部分統一空間）: 10-12%"
echo "  - Phase 3（動的制御）: 15-20%"