# config_linux.py
"""
LISA-Llama4統合モデル統一設定ファイル

このファイルは以下の設定を一元管理します：
- LISA-Llama4統合モデルの基本設定
- LoRA（Parameter Efficient Fine-tuning）設定  
- 分散学習・最適化設定
- パス・ディレクトリ設定

使用方法:
    import config_linux as config
    model_config = config.get_lisa_model_config()
    lora_config = config.get_lora_config()
"""
import os
from pathlib import Path
from typing import Dict, Any
import torch  # Web調査修正: torch.bfloat16使用のため

# ==============================================================================
# 1. 基本パス設定 (Lambda Cloud環境)
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent

# ==============================================================================
# Lambda Cloud インスタンス自動パス検出
# ==============================================================================
def _detect_lambda_path():
    """Lambda Cloudのインスタンスタイプを自動検出してパスを決定"""
    # 候補パス（優先順：central-texas → north-texas）
    path_candidates = [
        "/lambda/nfs/llama4-lisa-project-fs-central-texas",
        "/lambda/nfs/llama4-lisa-project-fs-north-texas"
    ]
    
    for base_path in path_candidates:
        if os.path.exists(base_path):
            print(f"✅ Lambda Cloud パス検出: {base_path}")
            return base_path
    
    # フォールバック: 環境変数または最初の候補
    fallback_path = "/lambda/nfs/llama4-lisa-project-fs-central-texas"
    print(f"⚠️  Lambda Cloud パス検出失敗、フォールバック使用: {fallback_path}")
    return fallback_path

# Lambda Cloud ベースパス自動検出
_LAMBDA_BASE_PATH = _detect_lambda_path()

# データセットベースディレクトリ（自動切り替え対応）
DATASET_BASE_DIR = os.environ.get("LISA_DATASET_BASE_DIR", f"{_LAMBDA_BASE_PATH}/data/dataset")

# SAMチェックポイントパス（ViT-H）（自動切り替え対応）
SAM_CHECKPOINT_PATH = os.environ.get("LISA_SAM_CHECKPOINT_PATH", f"{_LAMBDA_BASE_PATH}/data/weights/sam_vit_h_4b8939.pth")

# SAM2 Checkpoints (2025年ベストプラクティス - Web調査修正版)（自動切り替え対応）
SAM2_CHECKPOINT_PATH = os.environ.get("LISA_SAM2_CHECKPOINT_PATH", f"{_LAMBDA_BASE_PATH}/data/weights/sam2_hiera_large.pt")
SAM2_CONFIG_NAME = "sam2_hiera_l.yaml"  # Web調査ベース正式名
SAM2_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"  # ✅ 修正: 正しいファイル名
SAM2_HF_MODEL_ID = "facebook/sam2-hiera-large"  # ✅ HuggingFaceフォールバック用

# Hugging Faceキャッシュディレクトリ
HF_CACHE_DIR = os.environ.get('HF_HOME', None)

# ログ・出力ディレクトリ
LOG_BASE_DIR = str(PROJECT_ROOT / "runs")
WEIGHTS_DIR = str(PROJECT_ROOT / "weights")

# ==============================================================================
# 2. LISA-Llama4統合モデル設定（実際使用値に統一）
# ==============================================================================
# 使用モデル
LLAMA_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# 高速ロード設定
USE_SAFETENSORS = True  # safetensors形式を優先的に使用
SAFETENSORS_MODEL_PATH = "./models/llama4-scout-safetensors"  # 変換済みモデルパス
LOW_CPU_MEM_USAGE = True  # CPU→GPU転送を最適化（メモリ使用量削減）

# Llama-4-Scout-17B-16E-Instruct設定（2025年1月最新バグ回避）
ATTN_IMPLEMENTATION = "sdpa"  # 最も安定（flex_attentionバグ回避、Issue #37352）
# 注: flex_attentionは推奨だがTypeErrorバグあり、eagerもcausal maskバグあり（Issue #37322）
# ⚠️ 重要: この設定は全スクリプトで統一使用すること（個別指定禁止）
DEVICE_MAP = "auto"                     # GPU自動分散（実使用値）
TORCH_DTYPE = torch.bfloat16            # Llama-4 Scout推奨精度（BFloat16バックワードパス対応・2025年Web調査修正）

# BFloat16精度対応設定（2025年Webリサーチ推奨）
USE_LOSS_SCALING = False                # BFloat16はLoss Scaling不要（自動安定性）
INITIAL_LOSS_SCALE = 2**16              # 初期Loss Scale値（BFloat16では未使用）

# モデル構造パラメータ
LLAMA_HIDDEN_SIZE = 5120               # Llama4-Scout隠れ層サイズ
SAM_PROMPT_EMBED_DIM = 256             # SAM-ViT-H埋め込み次元
LLAMA_IMAGE_SIZE = 448                 # Llama4画像タイルサイズ
SAM_IMAGE_SIZE = 1024                  # SAMエンコーダ入力サイズ
MODEL_MAX_LENGTH = 131072              # Llama4最大コンテキスト長（128K）

# セグメンテーション統一サイズ（重要：全テスト・学習で統一）
SEGMENTATION_MASK_SIZE = 448           # セグメンテーションマスクサイズ（SAM2出力と統一）
SEGMENTATION_IMAGE_SIZE = 448          # セグメンテーション用画像サイズ（Llamaと統一）

# セグメンテーション特別トークン
SEG_TOKEN = "[SEG]"

# ==============================================================================
# 3. LoRA（PEFT）設定（SAM2+MLE論文準拠修正版）
# ==============================================================================
LORA_R = 16                            # 🔄 SAM2+MLE論文準拠（64→16）パラメータ効率最適化
LORA_ALPHA = 32                        # 🔄 SAM2+MLE論文準拠（128→32）2 * r
LORA_DROPOUT = 0.1                     # 🔄 SAM2+MLE論文準拠（0.05→0.1）

# ターゲットモジュール（SAM2+MLE論文準拠）
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention層（必須）
    "gate_proj", "up_proj", "down_proj"      # FFN層（必須）
]

# 🔄 SAM2+MLE論文準拠: モーダル特化target_modules（実際のモジュール名に修正）
SAM2_TARGET_MODULES = [
    "attn.qkv",                              # SAM2 combined QKV projection（ログから判明）
    "attn.proj",                             # SAM2 output projection（ログから判明）
    "mlp.layers.0",                          # SAM2 MLP第1層（ログから判明）
    "mlp.layers.1"                           # SAM2 MLP第2層（ログから判明）
]

QFORMER_TARGET_MODULES = [
    "query", "key", "value", "dense"         # Q-Former標準モジュール
]

# 注: lm_headとembed_tokensは後で個別に凍結解除（edit_config.md推奨）

# 追加学習可能パラメータ（edit_config.md推奨）
ADDITIONAL_TRAINABLE_PARAMS = [
    "lm_head",           # 言語モデルヘッド
    "embed_tokens",      # トークン埋め込み層
    "mask_decoder",      # SAMマスクデコーダー
    "projector",         # マルチモーダルプロジェクター
    "text_hidden_fcs"    # テキスト隠れ層（互換性のため）
]

# ==============================================================================
# 4. 学習・最適化設定（edit_config.md推奨値 + A100 80GB × 8GPU最適化）
# ==============================================================================
# 基本学習設定（NaN完全対策版：2025年Webリサーチ準拠）
LEARNING_RATE = 1e-5                   # AdamW学習率（5e-5 → 1e-5：極保守的設定）
WEIGHT_DECAY = 1e-2                    # 重み減衰（5e-2 → 1e-2：NaN対策）
BETA1 = 0.9                            # Adam beta1（維持）
BETA2 = 0.95                           # Adam beta2（0.999 → 0.95：NaN対策）

# エポック・ステップ設定（NaN完全対策版）
EPOCHS = 2                             # デフォルトエポック数（1-3エポック推奨）
STEPS_PER_EPOCH = 500                  # ステップ/エポック
WARMUP_RATIO = 0.2                     # ウォームアップ比率（0.1 → 0.2：超長期ウォームアップ）
LR_SCHEDULER_TYPE = "linear"           # 学習率スケジューラ（cosine → linear：安定性優先）

# バッチサイズ・勾配設定（NaN完全対策版）
BATCH_SIZE_PER_GPU = 1                 # GPU単位バッチサイズ（17Bモデル用）
GRADIENT_ACCUMULATION_STEPS = 2        # 勾配蓄積ステップ数（4 → 2：さらに安全）
# 実効バッチサイズ = 1 × 2 × 2GPU = 4（極安全設定）

# システム最適化設定（NaN完全対策版 + 2025年数値安定化）
MIXED_PRECISION = True                 # BF16混合精度学習
GRADIENT_CHECKPOINTING = True          # メモリ効率化勾配チェックポイント
DATALOADER_NUM_WORKERS = 4             # データローダワーカー数
GRADIENT_CLIP_NORM = 1.0               # 勾配クリッピング（BFloat16用緩和・2025年Web調査準拠）
USE_LOSS_SCALING = False               # BFloat16はLoss Scaling不要（2025年推奨）
INITIAL_LOSS_SCALE = 2**10             # 初期Loss Scale値（BFloat16では未使用）
LAYERNORM_EPSILON = 1e-5               # LayerNorm epsilon（BFloat16安定値）
ATTENTION_ENTROPY_MONITORING = True    # Attention Entropy Collapse監視（2025年研究準拠）
USE_8BIT_ADAM = True                   # 8bit AdamWオプティマイザ使用（メモリ25%削減）
OPTIM_TYPE = "paged_adamw_32bit"       # edit_config.md推奨: QLoRA使用時のメモリ効率オプティマイザ

# 損失関数設定（edit_config2.md推奨値：LISA原著準拠）
CE_LOSS_WEIGHT = 1.0                   # テキスト生成損失の重み（λtxt）
DICE_LOSS_WEIGHT = 0.5                 # DICE損失の重み（LISA原著準拠）
BCE_LOSS_WEIGHT = 2.0                  # BCE損失の重み（LISA原著準拠）

# 推論設定
MAX_NEW_TOKENS = 100                   # 生成時最大新規トークン数

# 量子化設定（Vision層とMoEルーター用）
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,  # 2025年Web調査修正: torch.bfloat16使用
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "quantize_vision_layers": True,      # Vision層の量子化を有効化
    "quantize_moe_routers": True         # MoEルーターの量子化を有効化
}

# ==============================================================================
# 5. データセット設定（最小限）
# ==============================================================================
# データセット種別（使用される場合の基本設定）
DATASET_SAMPLE_RATES = "9,3,3,1"       # sem_seg:refer_seg:vqa:reason_seg
SAMPLES_PER_EPOCH = 500                # デバッグ用サンプル数

# データセット名（将来の完全学習用）
SEM_SEG_DATA = "ade20k||cocostuff||mapillary||pascal_part||paco_lvis"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 5.5. パス検証関数
# ==============================================================================
def validate_and_report_paths():
    """設定されたパスの検証とレポート出力"""
    print("\n" + "="*60)
    print("Lambda Cloud 環境パス検証レポート")
    print("="*60)
    
    print(f"🔍 検出されたベースパス: {_LAMBDA_BASE_PATH}")
    print(f"📁 データセットパス: {DATASET_BASE_DIR}")
    print(f"   - 存在確認: {'✅ 存在' if os.path.exists(DATASET_BASE_DIR) else '❌ 不存在'}")
    
    if os.path.exists(DATASET_BASE_DIR):
        # 主要データセットの存在確認
        datasets_to_check = [
            "reason_seg/ReasonSeg/train",
            "reason_seg/ReasonSeg/explanatory",
            "refer_seg",
            "ade20k",
            "cocostuff"
        ]
        
        print(f"📊 主要データセット確認:")
        for dataset in datasets_to_check:
            full_path = os.path.join(DATASET_BASE_DIR, dataset)
            status = "✅ 存在" if os.path.exists(full_path) else "❌ 不存在"
            print(f"   - {dataset}: {status}")
    
    print(f"🎯 SAM2チェックポイント: {SAM2_CHECKPOINT_PATH}")
    print(f"   - 存在確認: {'✅ 存在' if os.path.exists(SAM2_CHECKPOINT_PATH) else '❌ 不存在'}")
    
    print("="*60 + "\n")

# ==============================================================================
# 6. 統一設定取得関数
# ==============================================================================
def get_lisa_model_config() -> Dict[str, Any]:
    """
    LISA-Llama4統合モデル設定取得
    
    Returns:
        Dict: LisaLlama4Config用設定辞書
    """
    return {
        "llama_model_id": LLAMA_MODEL_ID,
        "sam_checkpoint_path": SAM_CHECKPOINT_PATH,
        "seg_token": SEG_TOKEN,
        "llama_hidden_size": LLAMA_HIDDEN_SIZE,
        "sam_prompt_embed_dim": SAM_PROMPT_EMBED_DIM,
        "llama_image_size": LLAMA_IMAGE_SIZE,
        "sam_image_size": SAM_IMAGE_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device_map": DEVICE_MAP,
        "torch_dtype": TORCH_DTYPE,
        "segmentation_mask_size": SEGMENTATION_MASK_SIZE,
        "segmentation_image_size": SEGMENTATION_IMAGE_SIZE,
    }

def get_lora_config() -> Dict[str, Any]:
    """
    LoRA設定取得（edit_config.md推奨値準拠）
    
    Returns:
        Dict: LoraConfig用設定辞書
    """
    return {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        "bias": "none",                         # edit_config.md推奨
        "use_rslora": False,                    # r>=64や不安定性が見られる場合にTrue
        "task_type": "CAUSAL_LM"                # タスクタイプを明示的に指定
    }

def get_training_config() -> Dict[str, Any]:
    """
    学習設定取得（分散学習対応）
    
    Returns:
        Dict: 学習設定辞書
    """
    return {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "beta1": BETA1,
        "beta2": BETA2,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "mixed_precision": MIXED_PRECISION,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "use_8bit_adam": USE_8BIT_ADAM,
        "optim_type": OPTIM_TYPE,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "quantization_config": QUANTIZATION_CONFIG,
    }

def get_loss_config() -> Dict[str, float]:
    """
    損失関数設定取得
    
    Returns:
        Dict: 損失関数の重み設定
    """
    return {
        "ce_loss_weight": CE_LOSS_WEIGHT,
        "dice_loss_weight": DICE_LOSS_WEIGHT,
        "bce_loss_weight": BCE_LOSS_WEIGHT,
    }

def get_quantization_config() -> Dict[str, Any]:
    """
    量子化設定取得
    
    Returns:
        Dict: 量子化設定辞書
    """
    return QUANTIZATION_CONFIG

def get_path_config() -> Dict[str, str]:
    """
    パス設定取得
    
    Returns:
        Dict: パス設定辞書
    """
    return {
        "dataset_base_dir": DATASET_BASE_DIR,
        "sam_checkpoint_path": SAM_CHECKPOINT_PATH,
        "hf_cache_dir": HF_CACHE_DIR,
        "log_base_dir": LOG_BASE_DIR,
        "weights_dir": WEIGHTS_DIR,
    }

def get_test_config() -> Dict[str, Any]:
    """
    テスト用統一設定取得
    
    Returns:
        Dict: テスト設定辞書
    """
    return {
        "image_size": SEGMENTATION_IMAGE_SIZE,        # 448
        "mask_size": SEGMENTATION_MASK_SIZE,          # 448
        "llama_image_size": LLAMA_IMAGE_SIZE,         # 448
        "sam_image_size": SAM_IMAGE_SIZE,             # 1024
        "hidden_size": LLAMA_HIDDEN_SIZE,             # 5120
        "sam_prompt_dim": SAM_PROMPT_EMBED_DIM,       # 256
        "seg_token": SEG_TOKEN,                       # "[SEG]"
    }

# ==============================================================================
# 🆕 Sa2VA風[SEG]トークン設定（Phase 3C - Sa2VA統一トークン空間実装）
# ==============================================================================
# Sa2VA (arXiv:2501.04001) 準拠の[SEG]トークン実装設定
USE_SEG_TOKEN = True  # [SEG]トークン生成器を使用するか
USE_MULTI_FRAME_SEG = False  # 動画処理用の拡張版を使用するか
SEG_TOKEN_RETURN_ATTENTION = False  # 注意重みを返すか（デバッグ用）

# 統一トークン空間設定（将来の拡張用）
USE_PARTIAL_UNIFIED_SPACE = True  # Phase 2: 部分的統一トークン空間（有効化）
USE_DYNAMIC_SAM_CONTROL = False  # Phase 3: 動的SAM2制御
USE_ADAPTIVE_TOKEN_COMPRESSION = True  # Phase 2: 適応的トークン圧縮
USE_MOE_INTEGRATION = True  # Phase 2: MoE統合アダプター

def get_seg_token_config() -> Dict[str, Any]:
    """
    Sa2VA風[SEG]トークン設定取得
    
    Returns:
        Dict: [SEG]トークン生成設定
    """
    return {
        # 基本設定
        'use_seg_token': USE_SEG_TOKEN,
        'use_multi_frame': USE_MULTI_FRAME_SEG,
        'return_attention': SEG_TOKEN_RETURN_ATTENTION,
        
        # 次元設定（既存設定から継承）
        'qformer_hidden_size': 768,  # Q-Former隠れ層
        'llama_hidden_size': LLAMA_HIDDEN_SIZE,  # 5120
        'sam_prompt_dim': SAM_PROMPT_EMBED_DIM,  # 256
        
        # 将来の拡張
        'use_partial_unified_space': USE_PARTIAL_UNIFIED_SPACE,
        'use_dynamic_sam_control': USE_DYNAMIC_SAM_CONTROL,
        'use_adaptive_compression': USE_ADAPTIVE_TOKEN_COMPRESSION,
        'use_moe_integration': USE_MOE_INTEGRATION,
        
        # 期待性能向上
        'expected_improvement_phase1': 5.0,  # Phase 1: 5-8%
        'expected_improvement_phase2': 10.0,  # Phase 2: 10-12%
        'expected_improvement_phase3': 15.0,  # Phase 3: 15-20%
    }

# ==============================================================================
# 🔄 SAM2+MLE論文準拠統一設定
# ==============================================================================
def get_mle_config() -> Dict[str, Any]:
    """
    SAM2+MLE論文準拠統一設定取得
    
    論文: "Customize SAM for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts"
    成果: 最大28.14%性能向上実証
    
    Returns:
        Dict: MLE統一設定辞書
    """
    return {
        # LoRAパラメータ (論文準拠)
        'lora_rank': LORA_R,                    # 16 (論文推奨)
        'lora_alpha': LORA_ALPHA,               # 32 (論文推奨)
        'lora_dropout': LORA_DROPOUT,           # 0.1 (論文推奨)
        
        # MoE設定 (論文準拠)
        'moe_top_k': 2,                         # Top-2 expert selection
        'expert_capacity_factor': 1.25,         # 負荷分散
        'num_experts': 3,                       # Llama, SAM2, Q-Former
        
        # モーダル特化target_modules
        'target_modules': {
            'llama': LORA_TARGET_MODULES,       # Transformer標準
            'sam2': SAM2_TARGET_MODULES,        # Hiera ViT特化
            'qformer': QFORMER_TARGET_MODULES   # Q-Former特化
        },
        
        # エキスパート重み (論文準拠)
        'expert_weights': {
            'llama': 0.4,                       # 言語理解・推論
            'sam2': 0.4,                        # 視覚セグメンテーション
            'qformer': 0.2                      # クロスモーダル融合
        },
        
        # 期待効果 (論文実証値)
        'expected_improvement': 28.14,          # % 性能向上
        'parameter_efficiency': 97.5,           # % VRAM削減
        
        # 技術詳細
        'paper_title': "Customize SAM for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts",
        'paper_year': 2024,
        'benchmarks': ['DELIVER', 'MUSES', 'MCubeS']
    }

# ==============================================================================
# 7. 互換性維持（既存コード用）
# ==============================================================================
# 既存のテストスクリプトとの互換性のため、直接アクセス可能な設定を維持
def check_environment() -> bool:
    """
    環境設定確認
    
    Returns:
        bool: 必要なパス・設定が適切かどうか
    """
    required_paths = [
        ("SAMチェックポイント", SAM_CHECKPOINT_PATH),
        ("データセットベース", DATASET_BASE_DIR),
    ]
    
    all_valid = True
    print("=== 環境設定確認 ===")
    
    for name, path in required_paths:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}が見つかりません: {path}")
            all_valid = False
    
    return all_valid

def print_config_summary():
    """設定サマリ出力"""
    print("=== LISA-Llama4統合設定サマリ ===")
    print(f"Model: {LLAMA_MODEL_ID}")
    print(f"SAM: {SAM_CHECKPOINT_PATH}")
    print(f"Attention: {ATTN_IMPLEMENTATION}")
    print(f"Device Map: {DEVICE_MAP}")
    print(f"Dtype: {TORCH_DTYPE}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Batch Size: {BATCH_SIZE_PER_GPU} (per GPU)")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Mixed Precision: {MIXED_PRECISION}")

# ==============================================================================
# メイン実行（設定確認用）
# ==============================================================================
if __name__ == "__main__":
    print_config_summary()
    print("\n")
    env_ok = check_environment()
    
    if env_ok:
        print("\n🎉 設定確認完了: 全て正常です")
    else:
        print("\n⚠️ 設定確認完了: 一部不備があります")