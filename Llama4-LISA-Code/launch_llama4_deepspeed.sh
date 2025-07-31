#!/bin/bash

# Llama4-LISA DeepSpeed分散学習起動スクリプト
# Lambda Cloud A100×8環境専用

set -e  # エラー時に停止

echo "🚀 Llama4-LISA DeepSpeed分散学習起動"
echo "======================================"

# 環境変数設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# ログディレクトリ作成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs/llama4_deepspeed_${TIMESTAMP}"
mkdir -p ${LOG_DIR}

echo "📁 ログディレクトリ: ${LOG_DIR}"

# DeepSpeed起動パラメータ
DEEPSPEED_CONFIG="ds_config_llama4_moe.json"
NUM_GPUS=8
BATCH_SIZE=1
LEARNING_RATE=1e-4
EPOCHS=3
STEPS_PER_EPOCH=100

echo "⚙️  設定:"
echo "  - GPU数: ${NUM_GPUS}"
echo "  - DeepSpeed設定: ${DEEPSPEED_CONFIG}"
echo "  - バッチサイズ: ${BATCH_SIZE}"
echo "  - 学習率: ${LEARNING_RATE}"
echo "  - エポック数: ${EPOCHS}"

# DeepSpeed分散学習実行
echo ""
echo "🎯 DeepSpeed分散学習開始..."

deepspeed --num_gpus=${NUM_GPUS} \
    train_llama4_deepspeed.py \
    --ds_config=${DEEPSPEED_CONFIG} \
    --epochs=${EPOCHS} \
    --steps_per_epoch=${STEPS_PER_EPOCH} \
    --learning_rate=${LEARNING_RATE} \
    --batch_size=${BATCH_SIZE} \
    --dataset_type=hybrid \
    --output_dir=./outputs/llama4_deepspeed_${TIMESTAMP} \
    --use_wandb \
    --wandb_project=llama4-lisa-deepspeed \
    --checkpoint_interval=50 \
    2>&1 | tee ${LOG_DIR}/training.log

echo ""
echo "✅ DeepSpeed分散学習完了"
echo "📊 ログファイル: ${LOG_DIR}/training.log" 