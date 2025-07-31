#!/bin/bash

# Lambda Cloud修正デプロイスクリプト
echo "🚀 Phase 2修正（2025年公式API準拠）をLambda Cloudにデプロイ中..."

# 修正されたファイルをLambda Cloudに転送
echo "📁 Phase 2修正ファイル転送中..."
rsync -avz --progress -e "ssh -i ~/.ssh/lambda_cloud_key" \
    model/losses_qformer_sam2.py \
    model/qformer.py \
    model/llama4_qformer_sam2.py \
    model/sam2_integration.py \
    ubuntu@192.222.55.105:/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/model/

echo "✅ デプロイ完了"
echo "🧪 テスト実行:"
echo "ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.55.105 \"cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u test/overfit_llama4_qformer_sam2_batch.py 2>&1\""