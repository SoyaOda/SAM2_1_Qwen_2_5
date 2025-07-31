#!/bin/bash

# LISA-Gemma DeepSpeed ZeRO Stage 2 分散学習起動スクリプト
# Lambda Cloud A100*8 環境対応

set -e

# ======================================================================
# フェーズ2.3: DeepSpeed ZeRO Stage 2 統合
# A100*8環境でのマルチGPU分散学習
# ======================================================================

echo "🚀 LISA-Gemma DeepSpeed ZeRO Stage 2 分散学習"
echo "=" * 80
echo "実行時刻: $(date)"
echo "実行環境: Lambda Cloud A100*8 (80GB)"
echo "=" * 80

# ======================================================================
# 環境設定
# ======================================================================

# CUDA可視性設定（8GPU全て使用）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# TensorFlowログ抑制
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# PyTorch分散設定
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# Hugging Face設定
export TOKENIZERS_PARALLELISM=false

# DeepSpeed設定
export DEEPSPEED_LOG_LEVEL="DEBUG"

# ======================================================================
# GPU環境検証
# ======================================================================

echo "📊 GPU環境検証中..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
echo ""

# GPU数の確認
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "🔍 検出されたGPU数: $GPU_COUNT"

if [ "$GPU_COUNT" -ne 8 ]; then
    echo "❌ エラー: 8個のGPUが必要ですが、${GPU_COUNT}個しか検出されませんでした"
    exit 1
fi

echo "✅ 8GPU環境確認完了"
echo ""

# ======================================================================
# 必要ファイル存在確認
# ======================================================================

echo "📋 必要ファイル存在確認中..."

# 必須ファイルリスト
REQUIRED_FILES=(
    "train_deepspeed.py"
    "deepspeed_zero2_config.json"
    "config_linux.py"
    "model/gemma_lisa.py"
    "model/losses.py"
    "utils/dataset.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ エラー: 必須ファイルが見つかりません: $file"
        exit 1
    fi
    echo "✅ $file"
done

echo ""

# ======================================================================
# DeepSpeed設定確認
# ======================================================================

echo "🔧 DeepSpeed設定確認中..."
cat deepspeed_zero2_config.json
echo ""

# ======================================================================
# 学習パラメータ設定
# ======================================================================

# デフォルト設定
EPOCHS=${EPOCHS:-3}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-50}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
DATASET_TYPE=${DATASET_TYPE:-"all"}
BATCH_SIZE=${BATCH_SIZE:-2}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-25}
OUTPUT_DIR=${OUTPUT_DIR:-"./deepspeed_training_output"}

echo "🎯 学習設定:"
echo "  - エポック数: $EPOCHS"
echo "  - エポックあたりステップ数: $STEPS_PER_EPOCH"
echo "  - 学習率: $LEARNING_RATE"
echo "  - データセットタイプ: $DATASET_TYPE"
echo "  - バッチサイズ: $BATCH_SIZE"
echo "  - チェックポイント間隔: $CHECKPOINT_INTERVAL ステップ"
echo "  - 出力ディレクトリ: $OUTPUT_DIR"
echo ""

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# ======================================================================
# DeepSpeed起動
# ======================================================================

echo "🚀 DeepSpeed分散学習を開始..."
echo "コマンド: deepspeed --num_gpus=$GPU_COUNT train_deepspeed.py"
echo ""

# ログファイル名
LOG_FILE="$OUTPUT_DIR/deepspeed_training_$(date +%Y%m%d_%H%M%S).log"

# DeepSpeed実行
deepspeed \
    --num_gpus=$GPU_COUNT \
    --master_port=29500 \
    train_deepspeed.py \
    --epochs=$EPOCHS \
    --steps_per_epoch=$STEPS_PER_EPOCH \
    --learning_rate=$LEARNING_RATE \
    --dataset_type=$DATASET_TYPE \
    --batch_size=$BATCH_SIZE \
    --checkpoint_interval=$CHECKPOINT_INTERVAL \
    --output_dir="$OUTPUT_DIR" \
    --ds_config="deepspeed_zero2_config.json" \
    2>&1 | tee "$LOG_FILE"

# ======================================================================
# 実行結果確認
# ======================================================================

DEEPSPEED_EXIT_CODE=$?

echo ""
echo "=" * 80
echo "🎯 DeepSpeed分散学習実行結果"
echo "=" * 80

if [ $DEEPSPEED_EXIT_CODE -eq 0 ]; then
    echo "✅ DeepSpeed分散学習が正常に完了しました"
    echo ""
    echo "📊 生成されたファイル:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -la "$OUTPUT_DIR"
    fi
    echo ""
    echo "📝 ログファイル: $LOG_FILE"
    echo ""
    echo "🎉 Phase 2.3: DeepSpeed ZeRO Stage 2統合 - 成功"
    echo "   → Phase 3: DeepSpeed ZeRO Stage 3移行の準備が整いました"
else
    echo "❌ DeepSpeed分散学習がエラーで終了しました (終了コード: $DEEPSPEED_EXIT_CODE)"
    echo ""
    echo "📝 エラーログを確認してください: $LOG_FILE"
    echo ""
    echo "🔧 トラブルシューティング:"
    echo "  1. GPU環境の確認: nvidia-smi"
    echo "  2. DeepSpeed設定の確認: deepspeed_zero2_config.json"
    echo "  3. 必要なファイルの存在確認"
    echo "  4. メモリ使用量の確認"
    echo ""
    echo "❌ Phase 2.3: DeepSpeed ZeRO Stage 2統合 - 失敗"
fi

echo "=" * 80
echo "実行終了時刻: $(date)"
echo "=" * 80

exit $DEEPSPEED_EXIT_CODE 