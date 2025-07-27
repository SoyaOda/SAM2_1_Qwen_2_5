import os
from pathlib import Path

# ==============================================================================
# 小規模テスト用設定ファイル
# ==============================================================================
# プロジェクトのルートディレクトリ
PROJECT_ROOT = Path(__file__).parent

# 環境変数から設定を取得（デフォルト値付き）
DATASET_BASE_DIR = os.environ.get(
    'LISA_DATASET_BASE_DIR', 
    "/mnt/h/download/LISA-dataset/dataset"  # 既存のパスをフォールバックに
)

# SAMチェックポイントパス（環境変数または既存パス）
SAM_CHECKPOINT_PATH = os.environ.get(
    'LISA_SAM_CHECKPOINT_PATH',
    "/mnt/c/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth"  # 既存のパス
)

# Hugging Faceキャッシュディレクトリ
HF_CACHE_DIR = os.environ.get('HF_HOME', None)

# ==============================================================================
# モデル設定
# ==============================================================================
# Hugging Faceモデル識別子
GEMMA_MODEL_ID = "google/gemma-3-4b-it"

# ログと出力の保存先
LOG_BASE_DIR = str(PROJECT_ROOT / "runs")
WEIGHTS_DIR = str(PROJECT_ROOT / "weights")

# 画像サイズ設定
GEMMA_IMAGE_SIZE = 896  # Gemma-3のSigLIPエンコーダの要求仕様
SAM_IMAGE_SIZE = 1024   # SAM-ViTエンコーダの要求仕様
MODEL_MAX_LENGTH = 2048
SEG_PROJECTION_DIM = 256
SEG_TOKEN = "[SEG]"
GEMMA_HIDDEN_SIZE = 2560

# ==============================================================================
# 小規模テスト用訓練設定
# ==============================================================================
LEARNING_RATE = 1e-4
EPOCHS = 2  # 小規模テスト用に短縮
STEPS_PER_EPOCH = 10  # 小規模テスト用に短縮
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95

# 損失関数の重み
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 2.0

# ==============================================================================
# LoRA設定
# ==============================================================================
LORA_R = 16  # 小規模テスト用に縮小
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

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
# 小規模テスト用データセット設定
# ==============================================================================
# データセットの混合比率（小規模テスト用）
DATASET_SAMPLE_RATES = "1,1,1,1"  # 均等に設定

# 使用するデータセットの指定（小規模テスト用に限定）
SEM_SEG_DATA = "ade20k"  # 1つのデータセットのみ
REFER_SEG_DATA = "refcoco"  # 1つのデータセットのみ
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# データセット階層構造の定義（小規模テスト用）
# ==============================================================================
DATASET_STRUCTURE = {
    "sem_seg": {
        "ade20k": {
            "path": "ade20k",
            "images": "images",
            "annotations": "annotations",
            "required_files": ["images", "annotations"]
        }
    },
    "refer_seg": {
        "base_path": "refer_seg",
        "datasets": {
            "refcoco": {
                "annotations": "refcoco",
                "images": "images/mscoco/images/train2014"
            }
        },
        "required_files": ["refcoco", "images"]
    },
    "vqa": {
        "llava_instruct_150k": {
            "path": "llava_dataset",
            "json_file": "llava_instruct_150k.json",
            "required_files": ["llava_instruct_150k.json"]
        }
    },
    "reason_seg": {
        "ReasonSeg": {
            "path": "reason_seg/ReasonSeg",
            "train": "train",
            "val": "val",
            "explanatory": "explanatory",
            "required_files": ["train", "val"]
        }
    }
}

# ==============================================================================
# ユーティリティ関数
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
    
    print("=== 小規模テスト用設定検証 ===")
    
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
    
    # 3. 小規模テスト用データセットの確認
    test_datasets = [
        ("ReasonSeg", "reason_seg"),
        ("ade20k", "sem_seg"),
    ]
    
    for dataset_name, dataset_type in test_datasets:
        is_valid, missing = check_dataset_structure(dataset_name, dataset_type)
        if not is_valid:
            warnings.extend([f"{dataset_name}: {item}" for item in missing])
    
    # 4. 結果の表示
    if errors:
        print("❌ 設定検証エラー:")
        for error in errors:
            print(f"  - {error}")
        print("\n対処方法:")
        print("1. 環境変数を設定してパスを変更:")
        print(f"   export LISA_DATASET_BASE_DIR=/path/to/your/dataset")
        print(f"   export LISA_SAM_CHECKPOINT_PATH=/path/to/sam_vit_h_4b8939.pth")
        print("2. または、config_small_test.pyの該当パスを直接編集")
        print("3. データセットとSAM重みファイルをダウンロード・配置")
        
        raise FileNotFoundError("必須ファイルが見つかりません。上記の対処方法を実行してください。")
    
    if warnings:
        print("⚠️  警告（小規模テスト用なので一部データセットの欠如は許容）:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return True

def print_dataset_info():
    """データセット情報を表示"""
    print("=== 小規模テスト用データセット設定 ===")
    print(f"ベースディレクトリ: {DATASET_BASE_DIR}")
    print(f"SAMチェックポイント: {SAM_CHECKPOINT_PATH}")
    print(f"エポック数: {EPOCHS}")
    print(f"エポックあたりステップ数: {STEPS_PER_EPOCH}")
    print("\n使用データセット:")
    
    paths = get_dataset_paths()
    for dataset_type, datasets in paths.items():
        print(f"\n[{dataset_type.upper()}]")
        for name, path in datasets.items():
            status = "✓" if Path(path).exists() else "✗"
            print(f"  {status} {name}: {path}")

if __name__ == "__main__":
    print("=== LISA-Gemma3 小規模テスト設定検証 ===")
    try:
        if check_all_paths():
            print("✅ 設定検証に成功しました！")
            print_dataset_info()
    except FileNotFoundError as e:
        print(f"❌ 設定検証に失敗しました: {e}")
        exit(1) 