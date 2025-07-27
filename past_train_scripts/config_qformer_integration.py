# config_qformer_integration.py
"""
LISA-Llama4 + Q-Former + SAM2 統合モデル専用設定ファイル

moe_structure_approach.md 提案A実装専用:
- Llama-4-Scout (109B total, 17B active, MoE)
- Q-Former (BLIP-2ベース, 情報ボトルネック解消)
- SAM2 (6倍高速, 画像・動画対応)
- 段階的学習プロトコル (3ステージ)

このファイルは以下の設定を一元管理します：
- 統合モデルアーキテクチャ設定
- Q-Former設定（BLIP-2準拠）
- SAM2統合設定
- 段階的学習プロトコル設定
- MoE対応PEFT設定
"""

import os
from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# 1. 基本パス設定 (Lambda Cloud環境)
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent

# データセットベースディレクトリ
DATASET_BASE_DIR = os.environ.get("LISA_DATASET_BASE_DIR", "/lambda/nfs/lisa-gemma-project-fs/data/dataset")

# SAM/SAM2設定 (Meta公式API使用)
SAM_CHECKPOINT_PATH = os.environ.get("LISA_SAM_CHECKPOINT_PATH", "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth")
SAM2_MODEL_ID = os.environ.get("LISA_SAM2_MODEL_ID", "facebook/sam2-hiera-large")  # 🔄 HuggingFace Hub自動取得

# Hugging Faceキャッシュディレクトリ
HF_CACHE_DIR = os.environ.get('HF_HOME', None)

# ログ・出力ディレクトリ
LOG_BASE_DIR = str(PROJECT_ROOT / "runs_qformer")
WEIGHTS_DIR = str(PROJECT_ROOT / "weights")
QFORMER_WEIGHTS_DIR = str(PROJECT_ROOT / "qformer_weights")

# ==============================================================================
# 2. Llama-4-Scout設定（MoEアーキテクチャ対応）
# ==============================================================================
# 使用モデル (Llama-4-Scout: 109B total, 17B active, 16 experts)
LLAMA_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# アーキテクチャ設定
LLAMA_HIDDEN_SIZE = 5120               # Llama-4-Scout隠れ層サイズ
LLAMA_IMAGE_SIZE = 448                 # Llama4画像タイルサイズ (ネイティブマルチモーダル)
MODEL_MAX_LENGTH = 131072              # Llama4最大コンテキスト長（128K）

# 安定性設定（2025年1月最新）
ATTN_IMPLEMENTATION = "sdpa"           # 最も安定（flex_attentionバグ回避）
DEVICE_MAP = "auto"                    # GPU自動分散（Model Parallelism）
TORCH_DTYPE = "bfloat16"              # 推奨精度

# 高速ロード設定
USE_SAFETENSORS = True
LOW_CPU_MEM_USAGE = True

# ==============================================================================
# 3. Q-Former設定（BLIP-2ベース + Llama4適応）
# ==============================================================================
# 基本アーキテクチャ（BLIP-2準拠）
QFORMER_NUM_QUERIES = 32               # BLIP-2標準（32個の学習可能クエリ）
QFORMER_NUM_LAYERS = 6                 # BLIP-2標準
QFORMER_HIDDEN_SIZE = LLAMA_HIDDEN_SIZE  # 5120: Llama-4に合わせる
QFORMER_NUM_HEADS = 16                 # 5120 / 320 = 16
QFORMER_INTERMEDIATE_SIZE = LLAMA_HIDDEN_SIZE * 4  # 20480: FFNサイズ
QFORMER_DROPOUT = 0.1

# SAM2プロンプト次元
SAM_PROMPT_EMBED_DIM = 256             # SAM2プロンプト埋め込み次元

# ==============================================================================
# 4. SAM2設定（高速・高精度セグメンテーション）
# ==============================================================================
# SAM2モデル設定 (Meta公式API)
SAM2_MODEL_ID = "facebook/sam2-hiera-large"   # Meta公式HuggingFace Hub (自動取得)
SAM2_IMAGE_SIZE = 1024                        # SAM2エンコーダ入力サイズ

# セグメンテーション特別トークン
SEG_TOKEN = "[SEG]"                           # セグメンテーション指示トークン

# ==============================================================================
# 5. 段階的学習プロトコル設定（BLIP-2準拠）
# ==============================================================================
# ステージ1: インターフェースアライメント
STAGE1_EPOCHS = 2                      # Q-Former基礎学習
STAGE1_LEARNING_RATE = 1e-4           # 控えめな学習率
STAGE1_BATCH_SIZE = 4                  # 小バッチサイズ
STAGE1_FREEZE_LLAMA = True            # Llama-4凍結
STAGE1_FREEZE_SAM2 = True             # SAM2凍結
STAGE1_DATASETS = ["refcoco", "coco_stuff", "ade20k"]  # 汎用データ

# ステージ2: フルスタック適応
STAGE2_EPOCHS = 3                      # フルスタック学習
STAGE2_LEARNING_RATE = 5e-5           # より控えめ
STAGE2_BATCH_SIZE = 2                  # さらに小バッチ
STAGE2_FREEZE_LLAMA = True            # Llama-4凍結継続
STAGE2_FREEZE_SAM2_ENCODER = True     # SAM2エンコーダ凍結
STAGE2_TRAIN_SAM2_DECODER = True      # SAM2デコーダ学習
STAGE2_DATASETS = STAGE1_DATASETS     # 同じ汎用データ

# ステージ3: 推論能力専門化
STAGE3_EPOCHS = 1                      # 専門化学習
STAGE3_LEARNING_RATE = 1e-5           # 最も控えめ
STAGE3_BATCH_SIZE = 1                  # 最小バッチ
STAGE3_FREEZE_LLAMA = True            # Llama-4凍結継続
STAGE3_FREEZE_SAM2_ENCODER = True     # SAM2エンコーダ凍結継続
STAGE3_TRAIN_SAM2_DECODER = True      # SAM2デコーダ学習継続
STAGE3_DATASETS = ["reason_seg"]       # 推論型セグメンテーション専用

# ==============================================================================
# 6. MoE対応PEFT設定（MixLoRA風）
# ==============================================================================
# 基本LoRA設定
LORA_R = 64                           # LoRAランク（MoE最適化）
LORA_ALPHA = 128                      # LoRAアルファ（2 * r）
LORA_DROPOUT = 0.05                   # LoRAドロップアウト

# ターゲットモジュール（MoE専門エキスパート育成）
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention層
    "gate_proj", "up_proj", "down_proj"      # FFN/Expert層（重要）
]

# MoE特化設定（将来実装）
MOE_NUM_EXPERT_GROUPS = 4             # 学習対象エキスパートグループ数
MOE_TOP_K = 2                         # 各トークンで活性化するエキスパート数
MOE_LOAD_BALANCE_WEIGHT = 0.01        # 負荷分散重み（削除済み）

# ==============================================================================
# 7. 学習・最適化設定（統合モデル最適化）
# ==============================================================================
# 基本学習設定
LEARNING_RATE = 1e-4                  # デフォルト学習率
WEIGHT_DECAY = 0.05                   # 重み減衰
BETA1 = 0.9                           # Adam beta1
BETA2 = 0.999                         # Adam beta2

# バッチサイズ・勾配設定
BATCH_SIZE_PER_GPU = 1                # GPU単位バッチサイズ（大モデル用）
GRADIENT_ACCUMULATION_STEPS = 16      # 勾配蓄積
GRADIENT_CLIP_NORM = 1.0              # 勾配クリッピング

# システム最適化
MIXED_PRECISION = True                # BF16混合精度
GRADIENT_CHECKPOINTING = True         # メモリ効率化
DATALOADER_NUM_WORKERS = 4            # データローダワーカー数
OPTIM_TYPE = "paged_adamw_32bit"      # QLoRA推奨オプティマイザ

# 学習率スケジューラ
WARMUP_RATIO = 0.03                   # ウォームアップ比率
LR_SCHEDULER_TYPE = "cosine"          # コサイン減衰

# ==============================================================================
# 8. 損失関数設定（統合モデル用）
# ==============================================================================
# 基本損失重み
CE_LOSS_WEIGHT = 1.0                  # テキスト生成損失
DICE_LOSS_WEIGHT = 0.5                # DICE損失
BCE_LOSS_WEIGHT = 2.0                 # BCE損失

# Q-Former学習用追加損失（BLIP-2準拠）
ITC_LOSS_WEIGHT = 1.0                 # Image-Text Contrastive
ITM_LOSS_WEIGHT = 1.0                 # Image-Text Matching  
ITG_LOSS_WEIGHT = 1.0                 # Image-Grounded Text Generation

# ==============================================================================
# 9. 量子化設定（メモリ効率化）
# ==============================================================================
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
}

# ==============================================================================
# 10. データセット設定
# ==============================================================================
# ステージ別データセット
STAGE1_DATASET_CONFIG = {
    "sem_seg_data": "ade20k||cocostuff||pascal_part",
    "refer_seg_data": "refclef||refcoco||refcoco+||refcocog", 
    "sample_rates": "3,3,2,2",  # 均等サンプリング
    "samples_per_epoch": 1000,
}

STAGE2_DATASET_CONFIG = {
    **STAGE1_DATASET_CONFIG,
    "samples_per_epoch": 800,   # 少し減らす
}

STAGE3_DATASET_CONFIG = {
    "reason_seg_data": "ReasonSeg|train",
    "samples_per_epoch": 200,   # 高品質少量データ
}

# 検証データセット
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 11. 設定取得関数（統合モデル用）
# ==============================================================================
def get_qformer_integration_config() -> Dict[str, Any]:
    """統合モデル設定取得"""
    return {
        # Llama-4-Scout設定
        "llama_model_id": LLAMA_MODEL_ID,
        "llama_hidden_size": LLAMA_HIDDEN_SIZE,
        "llama_image_size": LLAMA_IMAGE_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device_map": DEVICE_MAP,
        "torch_dtype": TORCH_DTYPE,
        
        # Q-Former設定
        "qformer_config": {
            "num_queries": QFORMER_NUM_QUERIES,
            "hidden_size": QFORMER_HIDDEN_SIZE,
            "num_layers": QFORMER_NUM_LAYERS,
            "num_heads": QFORMER_NUM_HEADS,
            "intermediate_size": QFORMER_INTERMEDIATE_SIZE,
            "dropout": QFORMER_DROPOUT,
            "sam_prompt_dim": SAM_PROMPT_EMBED_DIM,
        },
        
        # SAM2設定
        "sam2_model_id": SAM2_MODEL_ID,        # Meta公式HuggingFace Hub
        "sam2_image_size": SAM2_IMAGE_SIZE,
        "seg_token": SEG_TOKEN,
    }

def get_progressive_training_config() -> Dict[str, Any]:
    """段階的学習設定取得"""
    return {
        "stage1": {
            "epochs": STAGE1_EPOCHS,
            "learning_rate": STAGE1_LEARNING_RATE,
            "batch_size": STAGE1_BATCH_SIZE,
            "freeze_llama": STAGE1_FREEZE_LLAMA,
            "freeze_sam2": STAGE1_FREEZE_SAM2,
            "datasets": STAGE1_DATASET_CONFIG,
        },
        "stage2": {
            "epochs": STAGE2_EPOCHS,
            "learning_rate": STAGE2_LEARNING_RATE,
            "batch_size": STAGE2_BATCH_SIZE,
            "freeze_llama": STAGE2_FREEZE_LLAMA,
            "freeze_sam2_encoder": STAGE2_FREEZE_SAM2_ENCODER,
            "train_sam2_decoder": STAGE2_TRAIN_SAM2_DECODER,
            "datasets": STAGE2_DATASET_CONFIG,
        },
        "stage3": {
            "epochs": STAGE3_EPOCHS,
            "learning_rate": STAGE3_LEARNING_RATE,
            "batch_size": STAGE3_BATCH_SIZE,
            "freeze_llama": STAGE3_FREEZE_LLAMA,
            "freeze_sam2_encoder": STAGE3_FREEZE_SAM2_ENCODER,
            "train_sam2_decoder": STAGE3_TRAIN_SAM2_DECODER,
            "datasets": STAGE3_DATASET_CONFIG,
        }
    }

def get_moe_peft_config() -> Dict[str, Any]:
    """MoE対応PEFT設定取得"""
    return {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "bias": "none",
        "use_rslora": True,  # MoE安定化
        "task_type": "CAUSAL_LM",
        
        # MoE特化設定（将来実装）
        "moe_config": {
            "num_expert_groups": MOE_NUM_EXPERT_GROUPS,
            "top_k": MOE_TOP_K,
        }
    }

def get_loss_config() -> Dict[str, float]:
    """損失関数設定取得"""
    return {
        "ce_loss_weight": CE_LOSS_WEIGHT,
        "dice_loss_weight": DICE_LOSS_WEIGHT,
        "bce_loss_weight": BCE_LOSS_WEIGHT,
        
        # Q-Former学習用
        "itc_loss_weight": ITC_LOSS_WEIGHT,
        "itm_loss_weight": ITM_LOSS_WEIGHT,
        "itg_loss_weight": ITG_LOSS_WEIGHT,
    }

def get_training_config() -> Dict[str, Any]:
    """学習設定取得"""
    return {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "beta1": BETA1,
        "beta2": BETA2,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mixed_precision": MIXED_PRECISION,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "optim_type": OPTIM_TYPE,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "quantization_config": QUANTIZATION_CONFIG,
    }

def get_path_config() -> Dict[str, str]:
    """パス設定取得"""
    return {
        "dataset_base_dir": DATASET_BASE_DIR,
        "sam_checkpoint_path": SAM_CHECKPOINT_PATH,
        "sam2_checkpoint_path": SAM2_CHECKPOINT_PATH,
        "hf_cache_dir": HF_CACHE_DIR,
        "log_base_dir": LOG_BASE_DIR,
        "weights_dir": WEIGHTS_DIR,
        "qformer_weights_dir": QFORMER_WEIGHTS_DIR,
    }

# ==============================================================================
# 12. 環境確認・互換性関数
# ==============================================================================
def check_integration_environment() -> bool:
    """統合モデル環境確認 (Meta公式API)"""
    print("=== 統合モデル環境確認 ===")
    
    all_valid = True
    
    # SAM2: HuggingFace Hub自動取得
    print(f"✅ SAM2: {SAM2_MODEL_ID} (HuggingFace Hub自動取得)")
    
    # データセットベース
    if os.path.exists(DATASET_BASE_DIR):
        print(f"✅ データセットベース: {DATASET_BASE_DIR}")
    else:
        print(f"⚠️ データセットベースが見つかりません: {DATASET_BASE_DIR}")
        all_valid = False
    
    # 必須パッケージ確認
    try:
        import sam2
        print("✅ segment-anything-2 インストール済み")
    except ImportError:
        print("❌ segment-anything-2 未インストール")
        print("   必須: pip install segment-anything-2")
        all_valid = False
    
    try:
        import huggingface_hub
        print("✅ huggingface_hub インストール済み")
    except ImportError:
        print("❌ huggingface_hub 未インストール")
        print("   必須: pip install huggingface_hub")
        all_valid = False
    
    return all_valid

def print_integration_config_summary():
    """統合モデル設定サマリ出力"""
    print("=== LISA-Llama4 + Q-Former + SAM2 統合設定サマリ ===")
    print(f"🧠 Llama-4: {LLAMA_MODEL_ID}")
    print(f"🔍 Q-Former: {QFORMER_NUM_QUERIES}クエリ, {QFORMER_NUM_LAYERS}層")
    print(f"🎯 SAM2: {SAM2_MODEL_ID}")
    print(f"📚 段階的学習: 3ステージ ({STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS}エポック総計)")
    print(f"⚡ 最適化: {ATTN_IMPLEMENTATION}, {TORCH_DTYPE}, {OPTIM_TYPE}")
    print(f"🔧 LoRA: r={LORA_R}, alpha={LORA_ALPHA}")

# ==============================================================================
# メイン実行（設定確認用）
# ==============================================================================
if __name__ == "__main__":
    print_integration_config_summary()
    print("\n")
    env_ok = check_integration_environment()
    
    if env_ok:
        print("\n🎉 統合モデル設定確認完了: 全て正常です")
    else:
        print("\n⚠️ 統合モデル設定確認完了: 一部不備があります")
    
    print("\n📋 設定関数一覧:")
    print("- get_qformer_integration_config(): 統合モデル設定")
    print("- get_progressive_training_config(): 段階的学習設定")
    print("- get_moe_peft_config(): MoE対応PEFT設定")
    print("- get_loss_config(): 損失関数設定")
    print("- get_training_config(): 学習設定")
    print("- get_path_config(): パス設定")