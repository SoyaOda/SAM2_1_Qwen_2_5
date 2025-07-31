#!/usr/bin/env python3
"""
SAMテンソル形状デバッグスクリプト
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_linux import *
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig

def debug_sam_shapes():
    print("=== SAMテンソル形状デバッグ ===")
    
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
    text_prompt = "この画像をセグメント化してください。 <SEG>"
    
    # 入力準備
    gemma_inputs = model.prepare_multimodal_input(image, text_prompt)
    device = next(model.gemma_model.parameters()).device
    gemma_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in gemma_inputs.items()}
    
    print(f"Input IDs shape: {gemma_inputs['input_ids'].shape}")
    print(f"SEG token ID: {model.seg_token_id}")
    
    # Gemmaの推論
    with torch.no_grad():
        gemma_outputs = model.gemma_model(
            **gemma_inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    print(f"Gemma hidden states shape: {gemma_outputs.hidden_states[-1].shape}")
    
    # SEGトークン検出
    input_ids = gemma_inputs["input_ids"]
    seg_positions = (input_ids == model.seg_token_id).nonzero(as_tuple=True)
    print(f"SEG positions: {seg_positions}")
    
    if len(seg_positions[0]) > 0:
        # SEGトークンの隠れ状態抽出
        last_hidden = gemma_outputs.hidden_states[-1]
        print(f"Last hidden shape: {last_hidden.shape}")
        
        seg_embeddings = []
        for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
            seg_hidden = last_hidden[batch_idx, token_idx]
            print(f"SEG hidden shape: {seg_hidden.shape}")
            
            seg_embedding = model.mlp_projector(seg_hidden.unsqueeze(0))
            print(f"MLP output shape: {seg_embedding.shape}")
            seg_embeddings.append(seg_embedding)
        
        # SAM画像前処理
        sam_image = image.resize((1024, 1024))
        sam_image_tensor = torch.tensor(np.array(sam_image)).permute(2, 0, 1).float()
        sam_image_tensor = sam_image_tensor.unsqueeze(0).to(device)
        print(f"SAM image tensor shape: {sam_image_tensor.shape}")
        
        # SAM画像エンコーディング
        with torch.no_grad():
            sam_features = model.sam_image_encoder(sam_image_tensor)
        print(f"SAM features shape: {sam_features.shape}")
        
        # SAMデコーダー入力準備
        seg_embedding = seg_embeddings[0]
        print(f"Single SEG embedding shape: {seg_embedding.shape}")
        
        # 形状変換テスト
        sparse_embeddings_v1 = seg_embedding.unsqueeze(0).unsqueeze(1)  # (1, 1, embed_dim)
        print(f"Sparse embeddings v1 shape: {sparse_embeddings_v1.shape}")
        
        sparse_embeddings_v2 = seg_embedding.unsqueeze(1)  # (1, embed_dim) -> (1, 1, embed_dim)
        print(f"Sparse embeddings v2 shape: {sparse_embeddings_v2.shape}")
        
        # Dense embeddings
        dense_embeddings = torch.zeros((1, 256, 256), device=device, dtype=torch.bfloat16)
        print(f"Dense embeddings shape: {dense_embeddings.shape}")
        
        # SAM Mask Decoderの内部を調べる
        print("\n=== SAM Mask Decoder内部調査 ===")
        
        # output_tokensの形状を確認
        output_tokens = torch.cat(
            [model.sam_mask_decoder.iou_token.weight, model.sam_mask_decoder.mask_tokens.weight], dim=0
        )
        print(f"Raw output_tokens shape: {output_tokens.shape}")
        
        output_tokens_expanded = output_tokens.unsqueeze(0).expand(
            sparse_embeddings_v2.size(0), -1, -1
        )
        print(f"Expanded output_tokens shape: {output_tokens_expanded.shape}")
        
        print(f"Trying to concatenate:")
        print(f"  output_tokens: {output_tokens_expanded.shape}")
        print(f"  sparse_embeddings: {sparse_embeddings_v2.shape}")
        
        try:
            tokens = torch.cat((output_tokens_expanded, sparse_embeddings_v2), dim=1)
            print(f"✓ Concatenation successful: {tokens.shape}")
        except Exception as e:
            print(f"✗ Concatenation failed: {e}")
            
            # さらに詳しく調査
            print(f"\nDetailed analysis:")
            print(f"  output_tokens_expanded.dim(): {output_tokens_expanded.dim()}")
            print(f"  sparse_embeddings_v2.dim(): {sparse_embeddings_v2.dim()}")
            print(f"  output_tokens_expanded.shape: {output_tokens_expanded.shape}")
            print(f"  sparse_embeddings_v2.shape: {sparse_embeddings_v2.shape}")

if __name__ == "__main__":
    debug_sam_shapes() 