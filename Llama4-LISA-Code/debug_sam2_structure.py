# debug_sam2_structure.py
"""
SAM2の実際の構造を確認するデバッグスクリプト

SAM2 predictor.modelの実際の属性を調査し、
mask_decoder相当の正しいアクセス方法を特定する
"""

import torch
import sys
import os

# CUDA環境設定
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

# プロジェクトパス
sys.path.append('.')
import config_linux

def debug_sam2_structure():
    """SAM2の構造をデバッグ"""
    print("=" * 80)
    print("🔍 SAM2構造デバッグ開始")
    print("=" * 80)
    
    try:
        # SAM2Wrapper初期化
        from model.sam2_integration import get_sam2_wrapper
        
        print("\n1. SAM2Wrapper初期化...")
        sam_wrapper = get_sam2_wrapper(
            model_id=config_linux.SAM2_HF_MODEL_ID,
            target_dtype=torch.bfloat16,
            debug_mode=False  # ローカル環境用にCPUモードを許可
        )
        
        print("\n2. SAM2Wrapper構造確認...")
        print(f"SAM2Wrapper type: {type(sam_wrapper)}")
        print(f"SAM2Wrapper attributes: {dir(sam_wrapper)}")
        
        # predictor属性の確認
        if hasattr(sam_wrapper, 'predictor'):
            print(f"\n3. predictor属性確認...")
            predictor = sam_wrapper.predictor
            print(f"predictor type: {type(predictor)}")
            print(f"predictor attributes: {[attr for attr in dir(predictor) if not attr.startswith('_')]}")
            
            # predictor.model属性の確認
            if hasattr(predictor, 'model'):
                print(f"\n4. predictor.model属性確認...")
                model = predictor.model
                print(f"model type: {type(model)}")
                print(f"model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
                # mask_decoder関連の属性を詳細調査
                print(f"\n5. mask_decoder関連属性詳細調査...")
                
                # 可能性のある属性名を確認
                possible_mask_decoder_names = [
                    'mask_decoder',
                    'sam_mask_decoder', 
                    'mask_head',
                    'decoder',
                    'sam_decoder'
                ]
                
                found_mask_decoder = None
                for attr_name in possible_mask_decoder_names:
                    if hasattr(model, attr_name):
                        print(f"✅ 発見: {attr_name}")
                        attr_obj = getattr(model, attr_name)
                        print(f"  - type: {type(attr_obj)}")
                        print(f"  - methods: {[m for m in dir(attr_obj) if not m.startswith('_') and callable(getattr(attr_obj, m))]}")
                        found_mask_decoder = attr_obj
                        break
                    else:
                        print(f"❌ 未発見: {attr_name}")
                
                # 全属性のタイプを確認
                print(f"\n6. 全model属性の詳細確認...")
                for attr_name in dir(model):
                    if not attr_name.startswith('_'):
                        try:
                            attr_obj = getattr(model, attr_name)
                            attr_type = type(attr_obj)
                            print(f"  {attr_name}: {attr_type}")
                            
                            # decoderやmaskという文字が含まれる属性を特別に確認
                            if 'decoder' in attr_name.lower() or 'mask' in attr_name.lower():
                                print(f"    🎯 重要: {attr_name} - {attr_type}")
                                if hasattr(attr_obj, '__call__'):
                                    print(f"    📞 callable: Yes")
                                
                        except Exception as e:
                            print(f"  {attr_name}: エラー - {e}")
                
                # image_encoder, prompt_encoderの確認
                print(f"\n7. 他の重要コンポーネント確認...")
                for component in ['image_encoder', 'prompt_encoder']:
                    if hasattr(model, component):
                        comp_obj = getattr(model, component)
                        print(f"✅ {component}: {type(comp_obj)}")
                    else:
                        print(f"❌ {component}: 未発見")
                
                # SAM2のpredictメソッドの確認
                print(f"\n8. SAM2 predict関連メソッド確認...")
                predict_methods = [m for m in dir(predictor) if 'predict' in m.lower()]
                print(f"predict関連メソッド: {predict_methods}")
                
                return found_mask_decoder, model
                
            else:
                print("❌ predictor.model属性が見つかりません")
                return None, None
        else:
            print("❌ predictor属性が見つかりません")
            return None, None
            
    except Exception as e:
        print(f"❌ SAM2構造デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_mask_generation(sam_wrapper):
    """マスク生成テスト"""
    print("\n" + "=" * 80)
    print("🎯 SAM2マスク生成テスト")
    print("=" * 80)
    
    try:
        # テスト画像作成
        test_image = torch.randint(0, 255, (1024, 1024, 3), dtype=torch.uint8).numpy()
        
        # 画像設定
        sam_wrapper.set_image(test_image)
        print("✅ 画像設定完了")
        
        # predict_with_promptsメソッドの確認
        if hasattr(sam_wrapper, 'predict_with_prompts'):
            print("✅ predict_with_promptsメソッド確認")
            
            # プロンプトなしでの予測テスト
            dummy_prompts = torch.randn(4, 256)
            results = sam_wrapper.predict_with_prompts(
                prompt_embeddings=dummy_prompts,
                multimask_output=True
            )
            
            print(f"✅ マスク生成成功")
            print(f"  - masks shape: {results['masks'].shape}")
            print(f"  - iou_predictions shape: {results['iou_predictions'].shape}")
            
            return True
        else:
            print("❌ predict_with_promptsメソッドが見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ マスク生成テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 構造デバッグ
    mask_decoder, model = debug_sam2_structure()
    
    # デバッグ結果の要約
    print("\n" + "=" * 80)
    print("📋 デバッグ結果要約")
    print("=" * 80)
    
    if mask_decoder is not None:
        print(f"✅ Mask decoder発見: {type(mask_decoder)}")
    else:
        print("❌ Mask decoder未発見")
    
    if model is not None:
        print(f"✅ SAM2 model確認: {type(model)}")
    else:
        print("❌ SAM2 model未確認")
    
    print("\n🔍 推奨される修正方針:")
    if mask_decoder is not None:
        print("1. 発見されたmask_decoder属性を使用してアクセス方法を修正")
        print("2. multiscale_decoder.pyの初期化コードを更新")
    else:
        print("1. SAM2のpredict_with_promptsメソッドを直接使用する設計に変更")
        print("2. mask_decoderに直接アクセスしない実装に修正")