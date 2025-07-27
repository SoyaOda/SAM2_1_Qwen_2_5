#!/usr/bin/env python3
"""
デュアルエンコーダー構成の簡易動作確認テスト
データセットを使わずに、基本的な画像処理機能をテスト
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append('.')

# 環境設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTHONUNBUFFERED'] = '1'

def test_dual_encoder_functions():
    """デュアルエンコーダー用の画像処理関数を直接テスト"""
    print("=== デュアルエンコーダー画像処理テスト ===")
    
    try:
        # 1. 必要なモジュールのインポート
        from utils.dataset import preprocess_sam_image, preprocess_llama_image, collate_fn
        from transformers import AutoProcessor, AutoTokenizer
        import config_linux
        
        print("✓ モジュールインポート成功")
        
        # 2. テスト画像の作成
        test_image = Image.new('RGB', (512, 512), color='blue')
        print(f"✓ テスト画像作成: {test_image.size}")
        
        # 3. プロセッサの初期化（簡易版）
        llama_model_id = config_linux.get_lisa_model_config()["llama_model_id"]
        try:
            processor = AutoTokenizer.from_pretrained(
                llama_model_id,
                trust_remote_code=True
            )
            processor.image_processor = None  # 画像プロセッサは手動処理
            print(f"✓ トークナイザ初期化成功")
        except Exception as e:
            print(f"❌ トークナイザ初期化エラー: {e}")
            return False
        
        # 4. SAM画像処理テスト
        print("\n--- SAM画像処理テスト ---")
        try:
            sam_pixel_values = preprocess_sam_image(test_image, 1024)
            print(f"✓ SAM画像処理成功: {sam_pixel_values.shape}")
            print(f"  - dtype: {sam_pixel_values.dtype}")
            print(f"  - 値の範囲: [{sam_pixel_values.min():.2f}, {sam_pixel_values.max():.2f}]")
        except Exception as e:
            print(f"❌ SAM画像処理エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. Llama画像処理テスト
        print("\n--- Llama画像処理テスト ---")
        try:
            pixel_values = preprocess_llama_image(test_image, processor, 448)
            print(f"✓ Llama画像処理成功: {pixel_values.shape}")
            print(f"  - dtype: {pixel_values.dtype}")
            print(f"  - 値の範囲: [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
        except Exception as e:
            print(f"❌ Llama画像処理エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 6. デュアルエンコーダー構成の確認
        print("\n=== デュアルエンコーダー構成確認 ===")
        print(f"✅ pixel_values (Llama-4用): {pixel_values.shape} - 448x448")
        print(f"✅ sam_pixel_values (SAM2用): {sam_pixel_values.shape} - 1024x1024")
        print("✅ 両方の画像が正しく処理されました！")
        
        # 7. collate_fn のテスト
        print("\n--- collate_fn テスト ---")
        
        # ダミーバッチデータ作成
        batch = [{
            'input_ids': torch.randint(0, 1000, (50,)),
            'labels': torch.randint(-100, 1000, (50,)),
            'attention_mask': torch.ones(50),
            'pixel_values': pixel_values,
            'sam_pixel_values': sam_pixel_values,
            'ground_truth_mask': None,
            'has_mask': False,
            'seg_token_mask': torch.zeros(50, dtype=torch.bool),
            'image_path': None,
            'text_prompt': "Test prompt [SEG]",
            'formatted_prompt': "User: Test prompt [SEG]",
            'original_size': (512, 512),
            'resize': None,
            'questions': None,
            'sampled_classes': None,
            'dataset_name': 'test'
        }]
        
        try:
            collated = collate_fn(batch)
            print("✓ collate_fn実行成功")
            
            # 結果の確認
            print("\n📊 バッチデータ構造:")
            for key, value in collated.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"  - {key}: List[{len(value)}]")
            
            # デュアルエンコーダー構成の最終確認
            if 'pixel_values' in collated and 'sam_pixel_values' in collated:
                print("\n🎉 成功: デュアルエンコーダー構成が正しく実装されています！")
                print(f"  - Llama-4用画像: {collated['pixel_values'].shape}")
                print(f"  - SAM2用画像: {collated['sam_pixel_values'].shape}")
            else:
                print("\n❌ エラー: 必要な画像データが不足しています")
                return False
                
        except Exception as e:
            print(f"❌ collate_fnエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 8. アダプターテスト
        print("\n--- dataset_adapter テスト ---")
        try:
            from model.dataset_adapter import adapt_dataset_for_qformer
            
            adapted = adapt_dataset_for_qformer(collated)
            print("✓ adapt_dataset_for_qformer実行成功")
            
            if 'images' in adapted and 'sam_images' in adapted:
                print(f"  - images (Llama-4用): {adapted['images'].shape}")
                print(f"  - sam_images (SAM2用): {adapted['sam_images'].shape}")
                print("\n✅ アダプターも正しく動作しています！")
            else:
                print("\n❌ アダプターエラー: 必要なフィールドがありません")
                return False
                
        except Exception as e:
            print(f"❌ アダプターエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n=== すべてのテストが成功しました！ ===")
        return True
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dual_encoder_functions()
    sys.exit(0 if success else 1)