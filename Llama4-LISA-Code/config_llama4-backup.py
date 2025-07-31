"""
LISA-Llama4 Configuration File
Llama-4-Scout-17B-16E-Instruct + SAM 構成設定
"""

import os

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_TYPE = "llama4"
VISION_ENCODER_TYPE = "native"  # Llama-4のネイティブビジョン

# Model Architecture Settings
HIDDEN_SIZE = 4096  # Llama-4-Scout hidden dimension
VOCAB_SIZE = 128256  # Llama tokenizer vocabulary size
NUM_EXPERTS = 16  # MoE experts
CONTEXT_LENGTH = 10_000_000  # 10M tokens
MAX_IMAGES_PER_INPUT = 5

# Special Tokens
IMAGE_TOKEN = "<image>"
SEG_TOKEN = "[SEG]"
BOI_TOKEN = "<boi>"  # Beginning of Image
EOI_TOKEN = "</eoi>"  # End of Image

# ============================================================================
# SAM Configuration
# ============================================================================

SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
SAM_IMAGE_SIZE = 1024

# SAM Training Settings
TRAIN_MASK_DECODER = True
OUT_DIM = 256  # SAM feature dimension

# ============================================================================
# Dataset Configuration 
# ============================================================================

DATASET_BASE_DIR = "/lambda_cloud_data/datasets/"

# Dataset mixing ratios (same as original LISA)
SAMPLE_RATE = [9, 3, 3, 1]  # [sem_seg, refer_seg, vqa, reason_seg]

# Dataset specific settings
SEM_SEG_DATA = "ade20k||cityscapes||cocostuff||mapillary||pcontext||pascal_part"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"

SAMPLES_PER_EPOCH = 5000
NUM_EPOCHS = 10

# ============================================================================
# Training Configuration
# ============================================================================

# Training Hyperparameters
LEARNING_RATE = 2e-5  # Conservative for MoE stability
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.95
GRADIENT_CLIP_NORM = 1.0  # Important for MoE training

# Loss Weights (same as original LISA)
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 2.0

# Training Settings
BATCH_SIZE = 1  # Start small due to memory constraints
GRAD_ACCUMULATION_STEPS = 8
EVAL_STEPS = 500
SAVE_STEPS = 1000
LOG_STEPS = 10

# Mixed Precision
USE_FP16 = False
USE_BF16 = True  # Recommended for Llama-4

# DeepSpeed Configuration (if needed)
USE_DEEPSPEED = False
DEEPSPEED_CONFIG = None

# ============================================================================
# Hardware Configuration
# ============================================================================

# Memory Management
MAX_MEMORY_FRACTION = 0.85
EMPTY_CACHE_FREQ = 100  # Clear cache every N steps

# Device Settings
DEVICE = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
NUM_GPUS = 1

# Attention Implementation
ATTN_IMPLEMENTATION = "flex_attention"  # Llama-4 optimized attention

# ============================================================================
# Data Processing Configuration
# ============================================================================

# Image Processing
LLAMA4_IMAGE_PROCESSOR = "native"  # Use Llama-4's native processor
SAM_IMAGE_SIZE = 1024
MAINTAIN_ASPECT_RATIO = True

# Text Processing
MAX_LENGTH = 2048  # Conservative for initial training
TRUNCATION = True
PADDING = "max_length"

# ============================================================================
# Evaluation Configuration
# ============================================================================

EVAL_BATCH_SIZE = 1
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.2
TOP_P = 0.9

# ============================================================================
# Logging and Monitoring
# ============================================================================

# Weights & Biases
USE_WANDB = True
WANDB_PROJECT = "LISA-Llama4"
WANDB_ENTITY = None  # Set your W&B entity
WANDB_NAME = f"llama4-scout-lisa-{MODEL_TYPE}"

# Output Directories
OUTPUT_DIR = "./outputs/llama4_lisa"
LOGGING_DIR = "./logs/llama4_lisa"
CHECKPOINT_DIR = "./checkpoints/llama4_lisa"

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# Verification Settings
# ============================================================================

# Test Data Paths
TEST_IMAGE_PATH = "./test_data/sample_image.jpg"
TEST_PROMPT = "Show me the car in this image. [SEG]"

# Verification Tolerances
LOSS_TOLERANCE = 1e-3
GRADIENT_TOLERANCE = 1e-6
PARAMETER_COUNT_TOLERANCE = 0.01  # 1% tolerance

# ============================================================================
# Advanced Configuration
# ============================================================================

# Model Optimization
USE_GRADIENT_CHECKPOINTING = True
USE_CACHE = False  # Disabled for training

# Quantization (for inference)
USE_QUANTIZATION = False
QUANTIZATION_BITS = 8

# Memory Optimization
OFFLOAD_OPTIMIZER = False
OFFLOAD_PARAMETERS = False

# ============================================================================
# Lambda Cloud Specific Settings
# ============================================================================

# Lambda Cloud data paths
LAMBDA_DATA_ROOT = "/lambda_cloud_data"
LAMBDA_MODELS_ROOT = "/lambda_cloud_models"

# Network settings for model download
HF_CACHE_DIR = "/lambda_cloud_models/huggingface_cache"
TRANSFORMERS_CACHE = HF_CACHE_DIR

# Environment variables setup
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# ============================================================================
# Debug Configuration
# ============================================================================

DEBUG_MODE = False
VERBOSE_LOGGING = True
PROFILE_MEMORY = False
PROFILE_TIME = False

# Debug specific settings
if DEBUG_MODE:
    SAMPLES_PER_EPOCH = 100
    NUM_EPOCHS = 1
    EVAL_STEPS = 50
    SAVE_STEPS = 100
    LOG_STEPS = 5

print("✅ Llama-4 LISA Configuration loaded successfully")
print(f"📊 Model: {MODEL_ID}")
print(f"🎯 Context Length: {CONTEXT_LENGTH:,} tokens")
print(f"🔧 MoE Experts: {NUM_EXPERTS}")
print(f"📁 Output Directory: {OUTPUT_DIR}") 