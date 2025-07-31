# debug_multiscale_issue.py
"""
マルチスケール特徴抽出の14次元問題をデバッグ
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('.')

from model.sam2_integration import get_sam2_wrapper
from model.multiscale_decoder import MultiScaleFeatureExtractor
import config_linux

def debug_multiscale_issue():
    """マルチスケール特徴抽出の問題を調査"""
    print("🔍 マルチスケール特徴抽出デバッグ開始...\n")
    
    # SAM2設定
    sam2_config = {
        'model_id': config_linux.SAM2_HF_MODEL_ID,
        'target_dtype': torch.bfloat16,
        'vos_optimized': True,
        'compile_model': False,
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
            print(f"✅ SAM2 image_encoder取得成功\n")
            
            # マルチスケール特徴抽出器を作成
            feature_extractor = MultiScaleFeatureExtractor(stages=[3, 6, 9, 12])
            
            # フックを登録
            feature_extractor.register_hooks(image_encoder)
            
            print(f"📋 フック登録状況:")
            print(f"  - 登録されたフック数: {len(feature_extractor.hooks)}")
            
            # 特徴抽出実行
            print(f"\n🧪 マルチスケール特徴抽出実行...")
            multiscale_features = feature_extractor(test_image, image_encoder)
            
            print(f"\n📐 抽出された特徴:")
            for name, feat in multiscale_features.items():
                if isinstance(feat, torch.Tensor):
                    print(f"  - {name}: {feat.shape}")
                    
                    # 詳細な形状分析
                    if feat.dim() == 4:
                        B, C, H, W = feat.shape
                        print(f"    → B={B}, C={C}, H={H}, W={W}")
                        print(f"    → リシェイプ後: [{B}, {H*W}, {C}]")
                        
                        # 14という数値が現れるか確認
                        if C == 14 or H == 14 or W == 14:
                            print(f"    ⚠️ 14が検出されました！")
                            
                        # 144 (embed_dim) に関連する値があるか確認
                        if C == 144 or H*W == 144:
                            print(f"    ✅ 144 (embed_dim)に関連する値を検出")
                    
                    elif feat.dim() == 3:
                        B, N, C = feat.shape
                        print(f"    → B={B}, N={N}, C={C}")
                        
                        if C == 14:
                            print(f"    ⚠️ 特徴次元が14です！")
                        
                        # 16128という数値との関連を確認
                        if N == 16128:
                            print(f"    ⚠️ シーケンス長が16128です！")
                            # 16128 = 127 * 127 = ほぼ128x128
                            possible_hw = int(N ** 0.5)
                            print(f"    → 可能な空間サイズ: {possible_hw}x{possible_hw}")
            
            # 通常のエンコーダ出力も確認
            print(f"\n🔄 通常のエンコーダ出力:")
            with torch.no_grad():
                normal_output = image_encoder(test_image)
                
            if isinstance(normal_output, dict):
                for key, value in normal_output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - {key}: {value.shape}")
            elif isinstance(normal_output, torch.Tensor):
                print(f"  - 出力: {normal_output.shape}")
            
            # Hieraの内部構造を詳しく調査
            if hasattr(image_encoder, 'trunk'):
                trunk = image_encoder.trunk
                print(f"\n🔬 Hiera trunk詳細調査:")
                
                # patch_embedの詳細
                if hasattr(trunk, 'patch_embed'):
                    patch_embed = trunk.patch_embed
                    print(f"  - patch_embed: {type(patch_embed)}")
                    
                    # テスト実行
                    with torch.no_grad():
                        x = test_image
                        x = patch_embed(x)
                        print(f"  - patch_embed出力: {x.shape}")
                        
                        # pos_embedがある場合
                        if hasattr(trunk, 'pos_embed'):
                            pos_embed = trunk.pos_embed
                            print(f"  - pos_embed形状: {pos_embed.shape}")
                            
                            # pos_embedを加算
                            if hasattr(trunk, '_pos_embed'):
                                x = trunk._pos_embed(x)
                                print(f"  - pos_embed適用後: {x.shape}")
                        
                        # 最初のブロックを実行
                        if hasattr(trunk, 'blocks') and len(trunk.blocks) > 0:
                            first_block = trunk.blocks[0]
                            x_before = x.clone()
                            x = first_block(x)
                            
                            # tupleの場合
                            if isinstance(x, tuple):
                                print(f"  - ブロック0出力: tuple長={len(x)}")
                                for i, xi in enumerate(x):
                                    if isinstance(xi, torch.Tensor):
                                        print(f"    - 要素{i}: {xi.shape}")
                                x = x[0]  # 通常最初の要素が特徴
                            else:
                                print(f"  - ブロック0出力: {x.shape}")
                            
                            # 形状変化を確認
                            if x.shape != x_before.shape:
                                print(f"    → 形状変化検出！")
                                
                # stage_endsの値を確認
                if hasattr(trunk, 'stage_ends'):
                    stage_ends = trunk.stage_ends
                    print(f"\n  - Hieraステージ境界: {stage_ends}")
                    
                    # 各ステージでの特徴次元を推測
                    if hasattr(trunk, 'blocks'):
                        for i, stage_end in enumerate(stage_ends):
                            if stage_end < len(trunk.blocks):
                                block = trunk.blocks[stage_end]
                                print(f"  - Stage {i+1} (Block {stage_end}): {type(block)}")


if __name__ == "__main__":
    debug_multiscale_issue()