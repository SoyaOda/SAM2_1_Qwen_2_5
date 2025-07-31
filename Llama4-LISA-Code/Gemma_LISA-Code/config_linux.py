import os
from pathlib import Path

# ==============================================================================
# 1. PATHS AND IDENTIFIERS - Lambda Cloud環境対応
# ==============================================================================
# プロジェクトのルートディレクトリ
PROJECT_ROOT = Path(__file__).parent

# データセットのベースディレクトリ（Lambda Cloud環境対応）
DATASET_BASE_DIR = "/lambda/nfs/lisa-gemma-project-fs/data/dataset"

# SAMチェックポイントパス（Lambda Cloud環境対応）
SAM_CHECKPOINT_PATH = "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth"

# Hugging Faceキャッシュディレクトリ（必要に応じて）
HF_CACHE_DIR = os.environ.get('HF_HOME', None)

# ==============================================================================
# 2. モデル識別子とファイル設定
# ==============================================================================
# Hugging Faceモデル識別子
GEMMA_MODEL_ID = "google/gemma-3-4b-it"

# ログと出力の保存先
LOG_BASE_DIR = str(PROJECT_ROOT / "runs")

# Weights保存先
WEIGHTS_DIR = str(PROJECT_ROOT / "weights")

# ==============================================================================
# 3. モデル設定
# ==============================================================================
# 画像サイズ設定
# GEMMA_IMAGE_SIZEはGemma-3のSigLIPエンコーダの要求仕様 (896x896)
GEMMA_IMAGE_SIZE = 896
# SAM_IMAGE_SIZEはSAM-ViTエンコーダの要求仕様 (1024x1024)
SAM_IMAGE_SIZE = 1024
# モデルが処理するトークンの最大長
MODEL_MAX_LENGTH = 2048
# MLPプロジェクタからSAMデコーダへの出力次元 (SAMのプロンプト埋め込み次元と一致)
SEG_PROJECTION_DIM = 256
# セグメンテーション用の特別なトークン（オリジナルLISAに準拠）
SEG_TOKEN = "[SEG]"
# Gemma-3-4bの隠れ層サイズ
GEMMA_HIDDEN_SIZE = 2560

# ==============================================================================
# 4. 訓練ハイパーパラメータ
# ==============================================================================
# 基本的な学習パラメータ
LEARNING_RATE = 1e-4
EPOCHS = 10
STEPS_PER_EPOCH = 500
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95

# バッチサイズとアキュムレーション設定
BATCH_SIZE_PER_GPU = 2  # GPU毎のバッチサイズ（LISA-Gemmaに最適化）
GRADIENT_ACCUMULATION_STEPS = 8  # 勾配蓄積ステップ（実効バッチサイズ = BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS * GPU数）

# システム設定
MIXED_PRECISION = True  # 混合精度学習を有効化（bf16）
GRADIENT_CHECKPOINTING = True  # メモリ効率化のための勾配チェックポイント
DATALOADER_NUM_WORKERS = 4  # データローダーのワーカー数

# ==============================================================================
# 8. accelerate & DTensor 設定
# ==============================================================================
# DTensor問題対策設定
DTENSOR_ALLOW_IMPLICIT_REPLICATION = True  # DTensor暗黙的レプリケーション許可
TORCH_DISTRIBUTED_DEBUG = "OFF"  # 分散デバッグログを抑制

# accelerate設定
ACCELERATE_MIXED_PRECISION = "bf16"  # 混合精度モード (bf16推奨)
ACCELERATE_GRADIENT_CLIPPING = 1.0  # 勾配クリッピング閾値
ACCELERATE_LOGGING_STEPS = 10  # ログ間隔

# 最適化設定
WARMUP_STEPS = 100  # ウォームアップステップ数
WARMUP_RATIO = 0.1  # ウォームアップ比率（WARMUP_STEPSが未設定の場合）
SAVE_STEPS = 500  # チェックポイント保存間隔
LOGGING_STEPS = 10  # ログ出力間隔
EVAL_STEPS = 500  # 評価実行間隔

# 損失関数の重み
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 2.0



# ==============================================================================
# 5. LoRA設定（Parameter-Efficient Fine-Tuning）
# ==============================================================================
LORA_R = 8  # 業界標準に合わせて4→8に最適化（Google/HuggingFace推奨範囲）
LORA_ALPHA = 16  # rank * 2の標準的な設定
LORA_DROPOUT = 0.05

# CRITICAL: Gemmaアーキテクチャに適合したターゲットモジュール
# ユーザー提供のスクリプトにあった "q_proj,k_proj,v_proj" は不完全。
# GemmaではAttention層とFFN層の両方をターゲットにする必要がある。
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ==============================================================================
# 6. データセット設定
# ==============================================================================
# データセットの混合比率
DATASET_SAMPLE_RATES = "9,3,3,1"  # sem_seg, refer_seg, vqa, reason_seg
# 各データセットのエポック毎サンプル数
SAMPLES_PER_EPOCH = 500  # 検証・デバッグ用のサンプル数
# 推論時の最大生成トークン数
MAX_NEW_TOKENS = 100
# 使用するデータセットの指定
SEM_SEG_DATA = "ade20k||cocostuff||mapillary||pascal_part||paco_lvis"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 7. データセット階層構造の定義
# ==============================================================================
DATASET_STRUCTURE = {
    # Semantic Segmentation データセット
    "sem_seg": {
        "ade20k": {
            "path": "ade20k",
            "images": "images",
            "annotations": "annotations",
            "required_files": ["images", "annotations"]
        },
        "cocostuff": {
            "path": "cocostuff",
            "images": "train2017",
            "annotations": "train2017",
            "required_files": ["train2017"]
        },
        "mapillary": {
            "path": "mapillary",
            "images": "training/images",
            "annotations": "training/labels",
            "config": "config_v2.0.json",
            "required_files": ["training", "config_v2.0.json"]
        },
        "pascal_part": {
            "path": "vlpart/pascal_part",
            "json_file": "train.json",
            "images": "VOCdevkit/VOC2010/JPEGImages",
            "required_files": ["train.json", "VOCdevkit"]
        },
        "paco_lvis": {
            "path": "vlpart/paco",
            "annotations": "annotations",
            "required_files": ["annotations"]
        }
    },
    
    # Referring Segmentation データセット
    "refer_seg": {
        "base_path": "refer_seg",
        "datasets": {
            "refcoco": {
                "annotations": "refcoco",
                "images": "images/mscoco/images/train2014"
            },
            "refcoco+": {
                "annotations": "refcoco+",
                "images": "images/mscoco/images/train2014"
            },
            "refcocog": {
                "annotations": "refcocog",
                "images": "images/mscoco/images/train2014"
            },
            "refclef": {
                "annotations": "refclef",
                "images": "images/saiapr_tc-12"
            }
        },
        "required_files": ["refcoco", "refcoco+", "refcocog", "refclef", "images"]
    },
    
    # VQA データセット
    "vqa": {
        "llava_instruct_150k": {
            "path": "llava_dataset",
            "json_file": "llava_instruct_150k.json",
            "required_files": ["llava_instruct_150k.json"]
        }
    },
    
    # Reasoning Segmentation データセット
    "reason_seg": {
        "ReasonSeg": {
            "path": "reason_seg/ReasonSeg",
            "train": "train",
            "val": "val",
            "explanatory": "explanatory",
            "required_files": ["train", "val", "explanatory"]
        }
    },
    
    # Part Segmentation データセット
    "vlpart": {
        "paco": {
            "path": "vlpart/paco",
            "annotations": "annotations",
            "required_files": ["annotations"]
        },
        "pascal_part": {
            "path": "vlpart/pascal_part",
            "annotations": "train.json",
            "images": "VOCdevkit",
            "required_files": ["train.json", "VOCdevkit"]
        }
    }
}

# ==============================================================================
# 8. ユーティリティ関数
# ==============================================================================
def get_dataset_paths() -> dict:
    """データセットへの完全なパスを構築して返す"""
    base_dir = Path(DATASET_BASE_DIR)
    paths = {}
    
    # Semantic Segmentation
    paths["sem_seg"] = {}
    for dataset_name, config in DATASET_STRUCTURE["sem_seg"].items():
        paths["sem_seg"][dataset_name] = str(base_dir / config["path"])
    
    # Referring Segmentation
    paths["refer_seg"] = {}
    refer_base = base_dir / DATASET_STRUCTURE["refer_seg"]["base_path"]
    for dataset_name, config in DATASET_STRUCTURE["refer_seg"]["datasets"].items():
        paths["refer_seg"][dataset_name] = str(refer_base)
    
    # VQA
    paths["vqa"] = {}
    for dataset_name, config in DATASET_STRUCTURE["vqa"].items():
        paths["vqa"][dataset_name] = str(base_dir / config["path"] / config["json_file"])
    
    # Reasoning Segmentation
    paths["reason_seg"] = {}
    for dataset_name, config in DATASET_STRUCTURE["reason_seg"].items():
        paths["reason_seg"][dataset_name] = str(base_dir / config["path"])
    
    return paths

def get_required_weights() -> dict:
    """必要な重みファイルの情報を返す"""
    return {
        "sam_vit_h": {
            "path": SAM_CHECKPOINT_PATH,
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "size_gb": 2.39,
            "description": "SAM ViT-H model checkpoint"
        }
    }

def check_dataset_structure(dataset_name: str, dataset_type: str) -> tuple[bool, list]:
    """特定のデータセットの構造をチェック"""
    base_dir = Path(DATASET_BASE_DIR)
    missing_items = []
    
    try:
        if dataset_type not in DATASET_STRUCTURE:
            return False, [f"Unknown dataset type: {dataset_type}"]
        
        if dataset_name not in DATASET_STRUCTURE[dataset_type]:
            return False, [f"Unknown dataset: {dataset_name} in {dataset_type}"]
        
        config = DATASET_STRUCTURE[dataset_type][dataset_name]
        dataset_path = base_dir / config["path"]
        
        if not dataset_path.exists():
            missing_items.append(f"Dataset directory: {dataset_path}")
            return False, missing_items
        
        # 必要なファイル/ディレクトリをチェック
        for required_item in config.get("required_files", []):
            item_path = dataset_path / required_item
            if not item_path.exists():
                missing_items.append(f"Required item: {item_path}")
        
        return len(missing_items) == 0, missing_items
        
    except Exception as e:
        return False, [f"Error checking {dataset_name}: {str(e)}"]

def check_all_paths() -> bool:
    """重要なパスと重みファイルの存在を検証する"""
    errors = []
    warnings = []
    
    # 1. 基本ディレクトリの存在確認
    base_dir = Path(DATASET_BASE_DIR)
    if not base_dir.exists():
        errors.append(f"データセットベースディレクトリが見つかりません: {base_dir}")
    
    # 2. SAMチェックポイントの確認
    sam_path = Path(SAM_CHECKPOINT_PATH)
    if not sam_path.exists():
        errors.append(f"SAMチェックポイントが見つかりません: {sam_path}")
        weights_info = get_required_weights()["sam_vit_h"]
        errors.append(f"ダウンロード: wget {weights_info['url']} -O {sam_path}")
    
    # 3. 重要なデータセットの確認
    critical_datasets = [
        ("ReasonSeg", "reason_seg"),
        ("ade20k", "sem_seg"),
        ("llava_instruct_150k", "vqa"),
    ]
    
    for dataset_name, dataset_type in critical_datasets:
        is_valid, missing = check_dataset_structure(dataset_name, dataset_type)
        if not is_valid:
            errors.extend([f"{dataset_name}: {item}" for item in missing])
    
    # 4. 結果の表示
    if errors:
        print("❌ 設定検証エラー:")
        for error in errors:
            print(f"  - {error}")
        print("\n対処方法:")
        print("1. 環境変数を設定してパスを変更:")
        print(f"   export LISA_DATASET_BASE_DIR=/path/to/your/dataset")
        print(f"   export LISA_SAM_CHECKPOINT_PATH=/path/to/sam_vit_h_4b8939.pth")
        print("2. または、config_linux.pyの該当パスを直接編集")
        print("3. データセットとSAM重みファイルをダウンロード・配置")
        
        raise FileNotFoundError("必須ファイルが見つかりません。上記の対処方法を実行してください。")
    
    if warnings:
        print("⚠️  警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return True

def print_dataset_info():
    """データセット情報を表示"""
    print("=== データセット設定情報 ===")
    print(f"ベースディレクトリ: {DATASET_BASE_DIR}")
    print(f"SAMチェックポイント: {SAM_CHECKPOINT_PATH}")
    print("\n利用可能なデータセット:")
    
    paths = get_dataset_paths()
    for dataset_type, datasets in paths.items():
        print(f"\n[{dataset_type.upper()}]")
        for name, path in datasets.items():
            status = "✓" if Path(path).exists() else "✗"
            print(f"  {status} {name}: {path}")

if __name__ == "__main__":
    print("=== LISA-Gemma3 設定検証 ===")
    try:
        if check_all_paths():
            print("✅ 設定検証に成功しました！")
            print_dataset_info()
    except FileNotFoundError as e:
        print(f"❌ 設定検証に失敗しました: {e}")
        exit(1) 