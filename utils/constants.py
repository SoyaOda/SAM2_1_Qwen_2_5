# utils/constants.py
"""
LISA-Llama4プロジェクト用の共通定数定義
"""

# デフォルトトークン
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_SEG_TOKEN = "[SEG]"

# 無視すべきインデックス
IGNORE_INDEX = -100

# ✅ 仕様準拠:
# Llama-4-Scout 17B (およびGemma-3) は画像を直接扱えるため、
# 独自の画像トークン予約は不要。特別な[SEG]トークンのみ追加する。
SEG_TOKEN = "[SEG]"

# ❌ 不要になった設定:
# GEMMA_START_OF_TURN = "<start_of_turn>"
# GEMMA_END_OF_TURN = "<end_of_turn>"
# GEMMA_BOS_TOKEN = "<bos>"
# GEMMA_EOS_TOKEN = "<eos>"
# GEMMA_IMAGE_TOKEN_NUM = 256
# IMAGE_TOKEN_INDEX = 262146

# システムプロンプト
SYSTEM_PROMPT = """You are a helpful assistant that can analyze images and understand visual content. You can describe what you see in images and answer questions about them."""

LLAMA_SYSTEM_PROMPT = """You are a helpful assistant that can analyze images and perform segmentation tasks. When asked to segment objects, you should respond with the [SEG] token."""

# Llama4専用システムプロンプト例
LLAMA4_SEGMENTATION_PROMPT = """You are LISA (Large-language Instructed Segmentation Assistant), a multimodal AI assistant that can understand images and perform precise object segmentation. When asked to segment objects or regions in images, respond with the [SEG] token to indicate the segmentation mask."""

LLAMA4_VQA_PROMPT = """You are a helpful multimodal AI assistant that can analyze images and answer questions about visual content. Provide accurate, detailed responses based on what you observe in the images."""

# 質問テンプレート
SHORT_QUESTION_LIST = [
    "Can you segment the {class_name} in this image?",
    "Please segment the {class_name}.",
    "Where is the {class_name}? Please segment it.",
    "Can you identify and segment the {class_name}?",
    "Please provide a segmentation mask for the {class_name}.",
]

LONG_QUESTION_LIST = [
    "Can you segment the region described as: {sent}?",
    "Please segment the area that matches: {sent}",
    "Where is the region that {sent}? Please segment it.",
    "Can you identify and segment the area described as: {sent}?",
    "Please provide a segmentation mask for the region: {sent}",
]

EXPLANATORY_QUESTION_LIST = [
    "Can you explain why this region is important?",
    "What makes this area significant?",
    "Why should we focus on this region?",
    "What is special about this area?",
    "Can you provide reasoning for this segmentation?",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

# 画像前処理の定数
SAM_PIXEL_MEAN = [123.675, 116.28, 103.53]
SAM_PIXEL_STD = [58.395, 57.12, 57.375]
SAM_IMAGE_SIZE = 1024
LLAMA_IMAGE_SIZE = 448

# デフォルト設定
DEFAULT_IGNORE_LABEL = 255
DEFAULT_NUM_CLASSES_PER_SAMPLE = 3
DEFAULT_SAMPLES_PER_EPOCH = 500 * 8 * 2 * 10 