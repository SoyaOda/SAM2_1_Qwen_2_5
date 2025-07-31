# debug_sam2_output.py
"""
SAM2の実際の出力を確認するデバッグスクリプト
Lambda Cloud環境で実行して、正確な特徴次元を特定する
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('.')

from model.sam2_integration import get_sam2_wrapper
from model.enhanced_llama4_qformer_sam2 import EnhancedQFormerSegmentationBridge, EnhancedLlamaQFormerSAM2Config
import config_linux

def debug_sam2_output():
    """SAM2の実際の出力を詳細にデバッグ"""
    print("🔍 SAM2出力デバッグ開始...")
    
    # SAM2設定
    sam2_config = {
        'model_id': config_linux.SAM2_HF_MODEL_ID,
        'target_dtype': torch.bfloat16,
        'vos_optimized': True,
        'compile_model': False,  # デバッグ時はコンパイルを無効
        'memory_pathways': 3,
        'mixed_precision': True,
    }
    
    # SAM2ラッパー取得
    sam_wrapper = get_sam2_wrapper(**sam2_config)
    
    # テスト用画像作成
    batch_size = 1
    test_image = torch.randn(batch_size, 3, 448, 448).cuda()
    print(f"📊 テスト画像形状: {test_image.shape}")
    
    # SAM2エンコーダ取得と実行
    if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
        actual_model = sam_wrapper.predictor.model
        if hasattr(actual_model, 'image_encoder'):
            image_encoder = actual_model.image_encoder
            print(f"✅ SAM2 image_encoder取得成功")
            
            # エンコーダ実行
            with torch.no_grad():
                encoded_output = image_encoder(test_image)
            
            print(f"\n📐 エンコーダ出力詳細:")
            
            if isinstance(encoded_output, dict):
                print(f"  - 出力タイプ: 辞書")
                for key, value in encoded_output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - {key}: {value.shape} (dtype: {value.dtype})")
                        # 最初の要素の実際の値も確認
                        if value.numel() > 0:
                            print(f"    最初の要素: {value.flatten()[0].item()}")
            elif isinstance(encoded_output, torch.Tensor):
                print(f"  - 出力タイプ: テンソル")
                print(f"  - 形状: {encoded_output.shape}")
                print(f"  - dtype: {encoded_output.dtype}")
                
                # 次元ごとのサイズを詳細に表示
                if encoded_output.dim() == 4:
                    B, C, H, W = encoded_output.shape
                    print(f"  - バッチサイズ: {B}")
                    print(f"  - チャネル数: {C}")
                    print(f"  - 高さ: {H}")
                    print(f"  - 幅: {W}")
                elif encoded_output.dim() == 3:
                    B, N, C = encoded_output.shape
                    print(f"  - バッチサイズ: {B}")
                    print(f"  - シーケンス長: {N}")
                    print(f"  - 特徴次元: {C}")
            
            # Enhanced modelでの処理を模擬
            print(f"\n🔄 Enhanced modelでの処理シミュレーション...")
            
            # 実際のforward処理の一部を再現
            if hasattr(image_encoder, 'trunk'):
                trunk = image_encoder.trunk
                print(f"  - trunk検出: {type(trunk)}")
                
                # 中間出力を確認
                x = test_image
                if hasattr(trunk, 'patch_embed'):
                    x = trunk.patch_embed(x)
                    print(f"  - patch_embed後: {x.shape}")
                
                # 最初のブロックだけ実行してみる
                if hasattr(trunk, 'blocks') and len(trunk.blocks) > 0:
                    x = trunk.blocks[0](x)
                    if isinstance(x, tuple):
                        x = x[0]
                    print(f"  - 最初のブロック後: {x.shape}")
            
            # _reshape_image_featuresの処理を確認
            print(f"\n🔄 _reshape_image_features処理確認...")
            image_feats = encoded_output
            if isinstance(image_feats, dict) and 'vision_features' in image_feats:
                image_feats = image_feats['vision_features']
            
            if image_feats.dim() == 4:
                B, C, H, W = image_feats.shape
                reshaped = image_feats.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
                print(f"  - リシェイプ前: {image_feats.shape}")
                print(f"  - リシェイプ後: {reshaped.shape}")
                print(f"  - 最終的な特徴次元: {C}")
            elif image_feats.dim() == 3:
                B, N, C = image_feats.shape
                print(f"  - 既に3次元: {image_feats.shape}")
                print(f"  - 最終的な特徴次元: {C}")
            
            # エラーメッセージから推測される形状を確認
            print(f"\n❗ エラーメッセージの解析:")
            print(f"  - エラー: mat1 and mat2 shapes cannot be multiplied (16128x14 and 1408x768)")
            print(f"  - mat1の形状: 16128x14")
            print(f"  - これは恐らく: batch_size * sequence_length x feature_dim")
            print(f"  - 16128 = おそらく batch_size(1) * 16128 tokens")
            print(f"  - 14 = 特徴次元")
            print(f"  - つまり、SAM2の実際の出力特徴次元は14次元という異常に小さい値")
            
            # 実際の次元を確認するため、さらに詳細な調査
            print(f"\n🔬 さらなる詳細調査...")
            
            # トランクの各段階での次元を確認
            if hasattr(image_encoder, 'trunk'):
                trunk = image_encoder.trunk
                
                # embed_dimやhidden_size属性を探す
                for attr in ['embed_dim', 'hidden_size', 'dim', 'num_features', 'width']:
                    if hasattr(trunk, attr):
                        value = getattr(trunk, attr)
                        print(f"  - trunk.{attr}: {value}")
                
                # patch_embedの出力次元
                if hasattr(trunk, 'patch_embed'):
                    if hasattr(trunk.patch_embed, 'proj'):
                        proj = trunk.patch_embed.proj
                        if hasattr(proj, 'out_channels'):
                            print(f"  - patch_embed出力チャネル: {proj.out_channels}")
                
                # pos_embedのサイズ
                if hasattr(trunk, 'pos_embed'):
                    print(f"  - pos_embed形状: {trunk.pos_embed.shape}")


if __name__ == "__main__":
    debug_sam2_output()