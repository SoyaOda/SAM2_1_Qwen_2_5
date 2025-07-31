"""
Linux環境用の設定ファイル

Windows環境からLinux環境へ移行するための設定です。
パスを適切にLinux用に変更し、環境固有の設定を管理します。
"""

import os

# データセットのベースディレクトリ
# WSL環境でWindowsファイルシステムにアクセス
DATASET_BASE_DIR = "/mnt/h/download/LISA-dataset/dataset"

# SAMモデルのパス
# WSL環境でWindowsファイルシステムにアクセス
SAM_MODEL_PATH = "/mnt/c/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth"

# デフォルトのモデル設定
DEFAULT_GEMMA_MODEL = "google/gemma-3-4b-it"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_MODEL_MAX_LENGTH = 512

# 学習用デフォルト設定
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUMULATION_STEPS = 8
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_EPOCHS = 3
DEFAULT_STEPS_PER_EPOCH = 1000

# LoRA設定
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj"

# デバッグ設定
DEFAULT_DEBUG_SAMPLES = 10
DEFAULT_DEBUG_DATASET = "ade20k"

# データセット固有のパス
def get_dataset_paths():
    """データセット固有のパスを取得"""
    return {
        "sem_seg": {
            "ade20k": os.path.join(DATASET_BASE_DIR, "ade20k"),
            "cocostuff": os.path.join(DATASET_BASE_DIR, "cocostuff"),
            "mapillary": os.path.join(DATASET_BASE_DIR, "mapillary"),
        },
        "refer_seg": {
            "refcoco": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refcoco+": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refcocog": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refclef": os.path.join(DATASET_BASE_DIR, "refer_seg"),
        },
        "vqa": {
            "llava_instruct_150k": os.path.join(DATASET_BASE_DIR, "llava_dataset"),
        },
        "reason_seg": {
            "ReasonSeg": os.path.join(DATASET_BASE_DIR, "reason_seg/ReasonSeg"),
        }
    }

def check_paths():
    """重要なパスが存在するかチェック"""
    paths_to_check = [
        DATASET_BASE_DIR,
        SAM_MODEL_PATH,
    ]
    
    missing_paths = []
    for path in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("警告: 以下のパスが見つかりません:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nconfig_linux.pyの設定を環境に合わせて修正してください。")
        return False
    
    return True

def get_environment_info():
    """環境情報を取得"""
    import torch
    import platform
    
    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info

if __name__ == "__main__":
    print("=== Linux環境設定チェック ===")
    
    # 環境情報を表示
    env_info = get_environment_info()
    print(f"プラットフォーム: {env_info['platform']}")
    print(f"Python バージョン: {env_info['python_version']}")
    print(f"PyTorch バージョン: {env_info['pytorch_version']}")
    print(f"CUDA 利用可能: {env_info['cuda_available']}")
    
    if env_info['cuda_available']:
        print(f"CUDA バージョン: {env_info['cuda_version']}")
        print(f"GPU 数: {env_info['gpu_count']}")
        for i, name in enumerate(env_info['gpu_names']):
            print(f"  GPU {i}: {name}")
    
    print("\n=== パス設定チェック ===")
    if check_paths():
        print("✓ すべての必要なパスが見つかりました")
    else:
        print("✗ 一部のパスが見つかりません")
    
    print(f"\nデータセットベースディレクトリ: {DATASET_BASE_DIR}")
    print(f"SAMモデルパス: {SAM_MODEL_PATH}") 