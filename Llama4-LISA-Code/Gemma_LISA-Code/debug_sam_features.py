#!/usr/bin/env python3
"""
SAM画像特徴量サイズデバッグスクリプト
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_linux import *
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig

def debug_sam_features():
    print("=== SAM画像特徴量サイズデバッグ ===")
    
    # モデル初期化
    config = LisaGemmaConfig(
        gemma_model_id=GEMMA_MODEL_ID,
        sam_checkpoint_path=SAM_CHECKPOINT_PATH,
        seg_token="<SEG>",
        gemma_hidden_size=2560,
        sam_prompt_embed_dim=256,
    )
    
    model = LisaGemmaForCausalLM(config)
    model = model.cuda()
    
    # テスト画像作成
    image = Image.new('RGB', (512, 512), color='red')
    
    # SAM用の画像前処理（1024x1024にリサイズ）
    sam_image = image.resize((1024, 1024))
    sam_image_tensor = torch.tensor(np.array(sam_image)).permute(2, 0, 1).float()
    sam_image_tensor = sam_image_tensor.unsqueeze(0).cuda()
    
    print(f"SAM入力画像サイズ: {sam_image_tensor.shape}")
    
    # SAMの画像エンコーディング
    with torch.no_grad():
        sam_features = model.sam_image_encoder(sam_image_tensor)
    
    print(f"SAM画像特徴量サイズ: {sam_features.shape}")
    
    # Dense PEのサイズも確認
    dense_pe = model.sam_prompt_encoder.get_dense_pe()
    print(f"Dense PE サイズ: {dense_pe.shape}")
    
    # 正しいdense_embeddingsのサイズを計算
    batch_size, channels, height, width = sam_features.shape
    print(f"正しいdense_embeddingsサイズ: ({batch_size}, {height}, {width})")
    
    # 正しいサイズでdense_embeddingsを作成
    correct_dense_embeddings = torch.zeros(
        (batch_size, height, width),
        device=sam_features.device,
        dtype=torch.bfloat16
    )
    print(f"作成したdense_embeddingsサイズ: {correct_dense_embeddings.shape}")

if __name__ == "__main__":
    debug_sam_features() 