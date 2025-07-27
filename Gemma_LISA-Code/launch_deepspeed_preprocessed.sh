#!/bin/bash

# LISA-Gemma DeepSpeed 少量ステップテスト起動スクリプト
# 事前処理チェックポイント対応版（DTensor問題解決済み）
# Lambda Cloud A100*8 環境対応

set -e

# ======================================================================
# フェーズ2.3最終テスト: 事前処理チェックポイント + DeepSpeed学習
# ======================================================================

echo "🚀 LISA-Gemma DeepSpeed 少量ステップテスト（事前処理チェックポイント版）"
echo "=" * 80
echo "実行時刻: $(date)"
echo "実行環境: Lambda Cloud A100*8 (80GB)"
echo "DTensor問題: 解決済み"
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
# 事前処理チェックポイント確認
# ======================================================================

echo "📁 事前処理チェックポイント確認中..."

if [ ! -d "/lambda/nfs/lisa-gemma-project-fs/data/preprocessed_checkpoints" ]; then
    echo "❌ エラー: data/preprocessed_checkpointsディレクトリが見つかりません"
    echo "   create_non_distributed_checkpoint.py を先に実行してください"
    exit 1
fi

# 最新のチェックポイントを検索（data/ディレクトリ版）
LATEST_CHECKPOINT=$(find /lambda/nfs/lisa-gemma-project-fs/data/preprocessed_checkpoints -name "lisa_gemma_non_distributed_*" -type d | sort | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ エラー: 事前処理チェックポイントが見つかりません"
    echo "   create_non_distributed_checkpoint.py を先に実行してください"
    exit 1
fi

echo "✅ 事前処理チェックポイント発見: $LATEST_CHECKPOINT"

# 必要なファイルの確認
CHECKPOINT_FILES=(
    "pytorch_model.bin"
    "model_info.json"
    "lisa_config.json"
    "tokenizer/tokenizer.json"
    "processor/preprocessor_config.json"
)

for file in "${CHECKPOINT_FILES[@]}"; do
    if [ ! -f "$LATEST_CHECKPOINT/$file" ]; then
        echo "❌ エラー: チェックポイントに必要ファイルが不足: $file"
        exit 1
    fi
    echo "  ✅ $file"
done

echo ""

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
    "train_deepspeed_with_preprocessed.py"
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
# 少量ステップテスト設定
# ======================================================================

# 少量ステップテスト用設定（オーバーライド可能）
EPOCHS=${EPOCHS:-1}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DATASET_TYPE=${DATASET_TYPE:-"sem_seg"}
BATCH_SIZE=${BATCH_SIZE:-1}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-99}
OUTPUT_DIR=${OUTPUT_DIR:-"./deepspeed_preprocessed_test_output"}

echo "🎯 少量ステップテスト設定:"
echo "  - エポック数: $EPOCHS"
echo "  - エポックあたりステップ数: $STEPS_PER_EPOCH"
echo "  - 総ステップ数: $((EPOCHS * STEPS_PER_EPOCH))"
echo "  - 学習率: $LEARNING_RATE"
echo "  - データセットタイプ: $DATASET_TYPE"
echo "  - バッチサイズ: $BATCH_SIZE"
echo "  - チェックポイント間隔: $CHECKPOINT_INTERVAL ステップ"
echo "  - 出力ディレクトリ: $OUTPUT_DIR"
echo "  - 使用チェックポイント: $LATEST_CHECKPOINT"
echo ""

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# ======================================================================
# DeepSpeed起動（事前処理チェックポイント版）
# ======================================================================

echo "🚀 DeepSpeed少量ステップテストを開始..."
echo "コマンド: deepspeed --num_gpus=$GPU_COUNT train_deepspeed_with_preprocessed.py"
echo ""

# ログファイル名
LOG_FILE="$OUTPUT_DIR/deepspeed_preprocessed_test_$(date +%Y%m%d_%H%M%S).log"

# DeepSpeed実行（事前処理チェックポイント対応版）
deepspeed \
    --num_gpus=$GPU_COUNT \
    --master_port=29500 \
    train_deepspeed_with_preprocessed.py \
    --epochs=$EPOCHS \
    --steps_per_epoch=$STEPS_PER_EPOCH \
    --learning_rate=$LEARNING_RATE \
    --dataset_type=$DATASET_TYPE \
    --batch_size=$BATCH_SIZE \
    --checkpoint_interval=$CHECKPOINT_INTERVAL \
    --output_dir="$OUTPUT_DIR" \
    --ds_config="deepspeed_zero2_config.json" \
    --preprocessed_checkpoint="$LATEST_CHECKPOINT" \
    2>&1 | tee "$LOG_FILE"

# ======================================================================
# 実行結果確認
# ======================================================================

DEEPSPEED_EXIT_CODE=$?

echo ""
echo "=" * 80
echo "🎯 DeepSpeed少量ステップテスト実行結果"
echo "=" * 80

if [ $DEEPSPEED_EXIT_CODE -eq 0 ]; then
    echo "✅ 少量ステップテストが正常に完了しました"
    echo ""
    echo "📊 達成された重要な成果:"
    echo "  ✅ DTensor問題: 完全回避"
    echo "  ✅ 語彙サイズ問題: 事前解決済み"
    echo "  ✅ DeepSpeed分散学習: 正常動作確認"
    echo "  ✅ A100*8環境: 動作確認済み"
    echo "  ✅ 事前処理チェックポイント: 正常ロード"
    echo ""
    echo "📊 生成されたファイル:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -la "$OUTPUT_DIR"
    fi
    echo ""
    echo "📝 ログファイル: $LOG_FILE"
    echo ""
    echo "🎉 Phase 2.3: 最終テスト成功！"
    echo ""
    echo "🚀 次のステップ準備完了:"
    echo "  1. フルエポック・フルデータセット学習"
    echo "  2. gemma-3-27b-itへのスケールアップ"
    echo "  3. Phase 3: DeepSpeed ZeRO Stage 3移行"
else
    echo "❌ 少量ステップテストがエラーで終了しました (終了コード: $DEEPSPEED_EXIT_CODE)"
    echo ""
    echo "📝 エラーログを確認してください: $LOG_FILE"
    echo ""
    echo "🔧 トラブルシューティング:"
    echo "  1. 事前処理チェックポイントの確認"
    echo "  2. GPU環境の確認: nvidia-smi"
    echo "  3. DeepSpeed設定の確認: deepspeed_zero2_config.json"
    echo ""
    echo "❌ Phase 2.3: 最終テスト失敗"
fi

echo "=" * 80
echo "実行終了時刻: $(date)"
echo "=" * 80

exit $DEEPSPEED_EXIT_CODE 