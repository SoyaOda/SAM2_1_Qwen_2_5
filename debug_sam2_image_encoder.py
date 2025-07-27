# debug_sam2_image_encoder.py
"""
SAM2のimage_encoderの詳細構造を確認するデバッグスクリプト
"""

import torch
import sys
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

sys.path.append('.')
import config_linux

def debug_sam2_image_encoder():
    """SAM2のimage_encoder構造を詳細デバッグ"""
    print("=" * 80)
    print("🔍 SAM2 Image Encoder詳細デバッグ")
    print("=" * 80)
    
    try:
        from model.sam2_integration import get_sam2_wrapper
        
        print("\n1. SAM2Wrapper初期化...")
        sam_wrapper = get_sam2_wrapper(
            model_id=config_linux.SAM2_HF_MODEL_ID,
            target_dtype=torch.bfloat16,
            debug_mode=False
        )
        
        print("\n2. image_encoder構造確認...")
        if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
            model = sam_wrapper.predictor.model
            
            if hasattr(model, 'image_encoder'):
                image_encoder = model.image_encoder
                print(f"✅ image_encoder type: {type(image_encoder)}")
                
                # 主要属性の確認
                print(f"\n3. image_encoder主要属性:")
                for attr in ['trunk', 'neck', 'stages', 'blocks', 'encoder', 'layers']:
                    if hasattr(image_encoder, attr):
                        attr_obj = getattr(image_encoder, attr)
                        print(f"  ✅ {attr}: {type(attr_obj)}")
                        
                        # trunkの詳細確認
                        if attr == 'trunk':
                            print(f"    trunk詳細属性:")
                            for trunk_attr in dir(attr_obj):
                                if not trunk_attr.startswith('_'):
                                    trunk_obj = getattr(attr_obj, trunk_attr)
                                    if not callable(trunk_obj):
                                        print(f"      - {trunk_attr}: {type(trunk_obj)}")
                    else:
                        print(f"  ❌ {attr}: 未発見")
                
                # ダミー画像でテスト
                print(f"\n4. 画像エンコーディングテスト...")
                dummy_image = torch.randn(1, 3, 1024, 1024).cuda()
                
                with torch.no_grad():
                    features = image_encoder(dummy_image)
                
                print(f"✅ エンコード成功")
                if isinstance(features, dict):
                    print(f"  - 出力タイプ: dict")
                    print(f"  - キー: {list(features.keys())}")
                    for key, value in features.items():
                        if hasattr(value, 'shape'):
                            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  - 出力形状: {features.shape}")
                    print(f"  - 出力dtype: {features.dtype}")
                
                # trunk経由でのアクセステスト
                if hasattr(image_encoder, 'trunk'):
                    print(f"\n5. trunk経由でのアクセステスト...")
                    trunk = image_encoder.trunk
                    
                    # blocksの確認（Hieraは直接blocksを持つ可能性）
                    if hasattr(trunk, 'blocks'):
                        print(f"  ✅ trunk.blocks発見: {type(trunk.blocks)}")
                        if isinstance(trunk.blocks, (list, torch.nn.ModuleList)):
                            print(f"    - ブロック数: {len(trunk.blocks)}")
                            for i, block in enumerate(trunk.blocks[:3]):  # 最初の3ブロック
                                print(f"    - Block {i}: {type(block)}")
                    
                    # stagesの確認
                    if hasattr(trunk, 'stages'):
                        print(f"  ✅ trunk.stages発見: {type(trunk.stages)}")
                        if isinstance(trunk.stages, (list, torch.nn.ModuleList)):
                            print(f"    - ステージ数: {len(trunk.stages)}")
                            for i, stage in enumerate(trunk.stages[:2]):  # 最初の2ステージ
                                print(f"    - Stage {i}: {type(stage)}")
                                if hasattr(stage, 'blocks'):
                                    print(f"      blocks数: {len(stage.blocks)}")
                    
                    # stage_endsの確認（Hieraの特徴）
                    if hasattr(trunk, 'stage_ends'):
                        print(f"  ✅ trunk.stage_ends発見: {trunk.stage_ends}")
                        print(f"    - Hieraのステージ区切り位置を示すリスト")
                
            else:
                print("❌ image_encoderが見つかりません")
                
        else:
            print("❌ predictor.modelが見つかりません")
            
    except Exception as e:
        print(f"❌ デバッグエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sam2_image_encoder()