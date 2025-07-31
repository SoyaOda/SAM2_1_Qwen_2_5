# debug_sam2_dimensions.py
"""
SAM2エンコーダの実際の出力次元をデバッグ

RuntimeError: mat1 and mat2 shapes cannot be multiplied (16128x14 and 1408x768)
このエラーの原因を調査: SAM2の実際の特徴次元を確認
"""

import torch
import torch.nn as nn
import sys
import os

# 環境設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('.')

from model.sam2_integration import get_sam2_wrapper
import config_linux

def debug_sam2_encoder_output():
    """SAM2エンコーダの出力次元をデバッグ"""
    print("🔍 SAM2エンコーダの出力次元デバッグ開始...")
    
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
    
    # SAM2エンコーダ取得
    if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
        actual_model = sam_wrapper.predictor.model
        if hasattr(actual_model, 'image_encoder'):
            image_encoder = actual_model.image_encoder
            print(f"✅ SAM2 image_encoder取得成功")
            print(f"  - タイプ: {type(image_encoder)}")
        else:
            print("❌ SAM2 image_encoderが見つかりません")
            return
    else:
        print("❌ SAM2 predictor.modelが見つかりません")
        return
    
    # エンコーダの詳細構造を確認
    print(f"\n📋 SAM2 Image Encoderの構造:")
    print(f"  - 利用可能な属性: {[attr for attr in dir(image_encoder) if not attr.startswith('_')][:15]}")
    
    # trunkがある場合の詳細確認
    if hasattr(image_encoder, 'trunk'):
        trunk = image_encoder.trunk
        print(f"  - trunk属性: {type(trunk)}")
        
        if hasattr(trunk, 'blocks'):
            blocks = trunk.blocks
            print(f"  - ブロック数: {len(blocks)}")
            
            # 最初と最後のブロックの確認
            if len(blocks) > 0:
                first_block = blocks[0]
                last_block = blocks[-1]
                print(f"  - 最初のブロック: {type(first_block)}")
                print(f"  - 最後のブロック: {type(last_block)}")
        
        # Hiera固有の設定確認
        if hasattr(trunk, 'stage_ends'):
            stage_ends = trunk.stage_ends
            print(f"  - Hieraステージ境界: {stage_ends}")
        
        # 次元関連の設定確認
        dim_attrs = ['embed_dim', 'num_features', 'hidden_size', 'dim']
        for attr in dim_attrs:
            if hasattr(trunk, attr):
                value = getattr(trunk, attr)
                print(f"  - {attr}: {value}")
    
    print(f"\n🧪 実際のエンコーダ実行テスト...")
    
    try:
        with torch.no_grad():
            # エンコーダ実行
            encoded_output = image_encoder(test_image)
            
            print(f"✅ エンコーダ実行成功")
            print(f"  - 出力タイプ: {type(encoded_output)}")
            
            if isinstance(encoded_output, dict):
                print(f"  - 辞書キー: {list(encoded_output.keys())}")
                for key, value in encoded_output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    - {key}: {value.shape} (dtype: {value.dtype})")
                
                # vision_featuresがある場合
                if 'vision_features' in encoded_output:
                    vision_features = encoded_output['vision_features']
                    print(f"\n📐 vision_features詳細:")
                    print(f"  - 形状: {vision_features.shape}")
                    print(f"  - 実際の特徴次元: {vision_features.shape[-1]}")
            
            elif isinstance(encoded_output, torch.Tensor):
                print(f"  - テンソル形状: {encoded_output.shape}")
                print(f"  - 実際の特徴次元: {encoded_output.shape[-1]}")
            
            else:
                print(f"  - 予期しない出力形式: {type(encoded_output)}")
        
        # 画像特徴のリシェイプテスト
        print(f"\n🔄 リシェイプテスト...")
        
        if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
            image_features = encoded_output['vision_features']
        elif isinstance(encoded_output, torch.Tensor):
            image_features = encoded_output
        else:
            print("❌ 画像特徴の取得に失敗")
            return
        
        print(f"  - 元の形状: {image_features.shape}")
        
        # Enhanced Q-Formerの_reshape_image_featuresロジックをテスト
        if image_features.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = image_features.shape
            reshaped = image_features.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
            print(f"  - リシェイプ後: {reshaped.shape}")
            print(f"  - シーケンス長: {H*W}")
            print(f"  - 特徴次元: {C}")
            
            # BLIP-2 Q-Formerで期待される形状
            print(f"\n🎯 BLIP-2互換性チェック:")
            print(f"  - 現在のencoder_hidden_size設定: 1408")
            print(f"  - 実際のSAM2特徴次元: {C}")
            print(f"  - 一致するか: {'✅' if C == 1408 else '❌'}")
            
            if C != 1408:
                print(f"🔧 修正が必要:")
                print(f"  - encoder_hidden_sizeを {C} に変更")
                print(f"  - または投影層で {C} -> 1408 に変換")
        
        elif image_features.dim() == 3:
            B, N, C = image_features.shape
            print(f"  - 既に3次元: {image_features.shape}")
            print(f"  - シーケンス長: {N}")
            print(f"  - 特徴次元: {C}")
            
            print(f"\n🎯 BLIP-2互換性チェック:")
            print(f"  - 現在のencoder_hidden_size設定: 1408")
            print(f"  - 実際のSAM2特徴次元: {C}")
            print(f"  - 一致するか: {'✅' if C == 1408 else '❌'}")
            
            if C != 1408:
                print(f"🔧 修正が必要:")
                print(f"  - encoder_hidden_sizeを {C} に変更")
        
    except Exception as e:
        print(f"❌ エンコーダ実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_sam2_encoder_output()