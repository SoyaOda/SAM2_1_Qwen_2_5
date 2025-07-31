# config_unified.py
"""
SAM2.1 + Qwen2.5-VL統合モデル統一設定ファイル

WindowsローカルとLambda Cloud環境の両方に対応した設定管理
- データセットパスの自動切り替え
- 環境変数による設定オーバーライド対応
- o3_spec2.mdに基づく<SEG>特殊トークン統合設定
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import platform

# ==============================================================================
# 1. 環境検出とベースパス設定
# ==============================================================================

def _detect_environment():
    """実行環境を自動検出（Windows vs Lambda Cloud）"""
    system = platform.system().lower()
    
    # Lambda Cloud検出（/lambda/nfs/ パスが存在する場合）
    lambda_paths = [
        "/lambda/nfs/llama4-lisa-project-fs-central-texas",
        "/lambda/nfs/llama4-lisa-project-fs-north-texas"
    ]
    
    for path in lambda_paths:
        if os.path.exists(path):
            print(f"✅ Lambda Cloud環境検出: {path}")
            return "lambda", path
    
    # Windows環境検出
    if system == "windows":
        print("✅ Windows環境検出")
        return "windows", None
    
    # Linux環境（Lambda以外）
    print("✅ Linux環境検出")
    return "linux", None

# 環境検出
ENVIRONMENT_TYPE, LAMBDA_BASE_PATH = _detect_environment()

# ==============================================================================
# 2. データセットパス設定（環境別）
# ==============================================================================

def _get_dataset_base_dir():
    """環境に応じたデータセットベースディレクトリを取得"""
    # 環境変数による明示的指定があれば最優先
    if "LISA_DATASET_BASE_DIR" in os.environ:
        path = os.environ["LISA_DATASET_BASE_DIR"]
        print(f"🔧 環境変数指定のデータセットパス: {path}")
        return path
    
    # 環境別のデフォルトパス
    if ENVIRONMENT_TYPE == "lambda":
        return f"{LAMBDA_BASE_PATH}/data/dataset"
    elif ENVIRONMENT_TYPE == "windows":
        # Windowsローカル環境のデフォルトパス
        return r"H:\download\LISA-dataset\data\dataset"
    else:
        # Linuxローカル環境のデフォルトパス（WSL2環境でのマウントパス対応）
        if os.path.exists("/mnt/h/download/LISA-dataset/dataset"):
            return "/mnt/h/download/LISA-dataset/dataset"
        return "./dataset"

# データセットベースディレクトリ
DATASET_BASE_DIR = _get_dataset_base_dir()

def _get_sam_checkpoint_path():
    """環境に応じたSAMチェックポイントパスを取得"""
    # 環境変数による明示的指定
    if "LISA_SAM_CHECKPOINT_PATH" in os.environ:
        return os.environ["LISA_SAM_CHECKPOINT_PATH"]
    
    # 環境別のデフォルトパス
    if ENVIRONMENT_TYPE == "lambda":
        return f"{LAMBDA_BASE_PATH}/data/weights/sam_vit_h_4b8939.pth"
    elif ENVIRONMENT_TYPE == "windows":
        return r"H:\download\weights\sam_vit_h_4b8939.pth"
    else:
        return "./weights/sam_vit_h_4b8939.pth"

# SAMチェックポイントパス
SAM_CHECKPOINT_PATH = _get_sam_checkpoint_path()

# ==============================================================================
# 3. SAM2.1設定（o3_spec2.md準拠）
# ==============================================================================

# SAM2.1チェックポイント設定
SAM2_CONFIG_ID = "facebook/sam2-hiera-large"
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

def _get_sam2_checkpoint_path():
    """SAM2.1チェックポイントパスを取得"""
    if "LISA_SAM2_CHECKPOINT_PATH" in os.environ:
        return os.environ["LISA_SAM2_CHECKPOINT_PATH"]
    
    if ENVIRONMENT_TYPE == "lambda":
        return f"{LAMBDA_BASE_PATH}/data/weights/sam2_hiera_large.pt"
    elif ENVIRONMENT_TYPE == "windows":
        return r"H:\download\weights\sam2_hiera_large.pt"
    else:
        return "./weights/sam2_hiera_large.pt"

SAM2_CHECKPOINT_PATH = _get_sam2_checkpoint_path()

# ==============================================================================
# 4. Qwen2.5-VL設定
# ==============================================================================

# Qwenモデル設定
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_IMAGE_SIZE = 448  # Qwen2.5-VLのネイティブ画像サイズ
SAM_IMAGE_SIZE = 1024  # SAM2.1の入力サイズ

# モデル構造パラメータ
QWEN_HIDDEN_SIZE = 2048  # Qwen2.5-VL-3Bの隠れ層サイズ
SAM_PROMPT_EMBED_DIM = 256  # SAMプロンプトエンコーダ次元
MODEL_MAX_LENGTH = 8192  # Qwen2.5-VLの最大コンテキスト長

# Torch設定
TORCH_DTYPE = torch.bfloat16
DEVICE_MAP = "auto"

# アテンション実装設定（安定性重視）
ATTN_IMPLEMENTATION = "flash_attention_2"  # Qwen2.5-VL推奨

# ==============================================================================
# 5. 特殊トークン設定（o3_spec2.md準拠）
# ==============================================================================

# <SEG>特殊トークン設定
SEG_TOKEN = "<SEG>"  # セグメンテーション用特殊トークン
IMAGE_TOKEN = "<image>"  # 画像プレースホルダートークン

# エンドツーエンド学習設定
USE_SEG_TOKEN_GENERATION = True  # <SEG>トークンによるマスク生成を有効化
MASK_QUERY_PROJECTION_DIM = 256  # マスククエリの投影次元

# ==============================================================================
# 6. データセット設定（Llama4-LISA-Code準拠）
# ==============================================================================

# データセット種別とサンプリング比率
DATASET_SAMPLE_RATES = "9,3,3,1"  # sem_seg:refer_seg:vqa:reason_seg
SAMPLES_PER_EPOCH = 500  # デバッグ用サンプル数

# データセット詳細設定
SEM_SEG_DATA = "ade20k||cocostuff||mapillary||pascal_part||paco_lvis"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 7. 学習設定（効率的微調整）
# ==============================================================================

# LoRA設定（効率的微調整）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# ターゲットモジュール（Qwen2.5-VL + SAM2.1対応）
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention層
    "gate_proj", "up_proj", "down_proj"      # FFN層
]

# SAM2.1特有のターゲットモジュール
SAM2_TARGET_MODULES = [
    "trunk.blocks.*.attn.qkv",
    "trunk.blocks.*.attn.proj", 
    "trunk.blocks.*.mlp.layers.0",
    "trunk.blocks.*.mlp.layers.1"
]

# 追加学習可能パラメータ
ADDITIONAL_TRAINABLE_PARAMS = [
    "lm_head",           # 言語モデルヘッド
    "embed_tokens",      # トークン埋め込み層
    "mask_decoder",      # SAMマスクデコーダー
    "mask_query_proj"    # マスククエリ投影層（新規追加）
]

# 学習パラメータ
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
BATCH_SIZE_PER_GPU = 1
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 2
WARMUP_RATIO = 0.1

# 損失関数重み（LISA準拠）
CE_LOSS_WEIGHT = 1.0   # テキスト生成損失
DICE_LOSS_WEIGHT = 0.5 # DICE損失
BCE_LOSS_WEIGHT = 2.0  # バイナリクロスエントロピー損失

# ==============================================================================
# 8. 統一設定取得関数
# ==============================================================================

def get_model_config() -> Dict[str, Any]:
    """統合モデル設定を取得"""
    return {
        "qwen_model_id": QWEN_MODEL_ID,
        "sam2_config_id": SAM2_CONFIG_ID,
        "sam2_checkpoint_path": SAM2_CHECKPOINT_PATH,
        "seg_token": SEG_TOKEN,
        "image_token": IMAGE_TOKEN,
        "qwen_hidden_size": QWEN_HIDDEN_SIZE,
        "sam_prompt_embed_dim": SAM_PROMPT_EMBED_DIM,
        "qwen_image_size": QWEN_IMAGE_SIZE,
        "sam_image_size": SAM_IMAGE_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device_map": DEVICE_MAP,
        "torch_dtype": TORCH_DTYPE,
        "use_seg_token_generation": USE_SEG_TOKEN_GENERATION,
        "mask_query_projection_dim": MASK_QUERY_PROJECTION_DIM,
    }

def get_dataset_config() -> Dict[str, Any]:
    """データセット設定を取得"""
    return {
        "dataset_base_dir": DATASET_BASE_DIR,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "sample_rates": [int(x) for x in DATASET_SAMPLE_RATES.split(",")],
        "sem_seg_data": SEM_SEG_DATA,
        "refer_seg_data": REFER_SEG_DATA,
        "vqa_data": VQA_DATA,
        "reason_seg_data": REASON_SEG_DATA,
        "val_dataset": VAL_DATASET,
    }

def get_training_config() -> Dict[str, Any]:
    """学習設定を取得"""
    return {
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "sam2_target_modules": SAM2_TARGET_MODULES,
        "additional_trainable_params": ADDITIONAL_TRAINABLE_PARAMS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "ce_loss_weight": CE_LOSS_WEIGHT,
        "dice_loss_weight": DICE_LOSS_WEIGHT,
        "bce_loss_weight": BCE_LOSS_WEIGHT,
    }

def get_path_config() -> Dict[str, str]:
    """パス設定を取得"""
    return {
        "environment_type": ENVIRONMENT_TYPE,
        "dataset_base_dir": DATASET_BASE_DIR,
        "sam_checkpoint_path": SAM_CHECKPOINT_PATH,
        "sam2_checkpoint_path": SAM2_CHECKPOINT_PATH,
        "lambda_base_path": LAMBDA_BASE_PATH or "N/A",
    }

# ==============================================================================
# 9. 環境検証機能
# ==============================================================================

def validate_environment() -> bool:
    """環境設定を検証し、レポートを出力"""
    print("\n" + "="*60)
    print("SAM2.1 + Qwen2.5-VL統合モデル環境検証")
    print("="*60)
    
    path_config = get_path_config()
    print(f"🌍 検出環境: {path_config['environment_type']}")
    print(f"📁 データセットパス: {path_config['dataset_base_dir']}")
    print(f"   - 存在確認: {'✅ 存在' if os.path.exists(path_config['dataset_base_dir']) else '❌ 不存在'}")
    
    print(f"🎯 SAM2.1チェックポイント: {path_config['sam2_checkpoint_path']}")
    print(f"   - 存在確認: {'✅ 存在' if os.path.exists(path_config['sam2_checkpoint_path']) else '❌ 不存在'}")
    
    if path_config['environment_type'] == 'lambda':
        print(f"☁️  Lambda Cloudベースパス: {path_config['lambda_base_path']}")
    
    # 主要データセットの存在確認
    if os.path.exists(path_config['dataset_base_dir']):
        print(f"📊 主要データセット確認:")
        datasets_to_check = [
            "reason_seg/ReasonSeg/train",
            "refer_seg",
            "ade20k",
            "cocostuff"
        ]
        
        for dataset in datasets_to_check:
            full_path = os.path.join(path_config['dataset_base_dir'], dataset)
            status = "✅ 存在" if os.path.exists(full_path) else "❌ 不存在"
            print(f"   - {dataset}: {status}")
    
    print("="*60 + "\n")
    
    # すべてのパスが存在するかチェック
    required_paths = [
        path_config['dataset_base_dir'],
        # SAM2.1チェックポイントは動的ダウンロード可能なのでオプション
    ]
    
    all_exist = all(os.path.exists(path) for path in required_paths)
    return all_exist

def print_config_summary():
    """設定サマリを出力"""
    print("=== SAM2.1 + Qwen2.5-VL統合モデル設定サマリ ===")
    print(f"Environment: {ENVIRONMENT_TYPE}")
    print(f"Qwen Model: {QWEN_MODEL_ID}")
    print(f"SAM2.1 Config: {SAM2_CONFIG_ID}")
    print(f"Dataset Path: {DATASET_BASE_DIR}")
    print(f"SEG Token: {SEG_TOKEN}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Batch Size: {BATCH_SIZE_PER_GPU} (per GPU)")
    print(f"Learning Rate: {LEARNING_RATE}")

# ==============================================================================
# メイン実行（設定確認用）
# ==============================================================================

if __name__ == "__main__":
    print_config_summary()
    print("\n")
    env_ok = validate_environment()
    
    if env_ok:
        print("🎉 環境設定完了: 基本パスが確認できました")
    else:
        print("⚠️ 環境設定完了: 一部パスが見つかりませんが、動的作成可能です")