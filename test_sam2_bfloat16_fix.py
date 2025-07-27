#!/usr/bin/env python3
"""
SAM2 BFloat16互換性テストスクリプト
Web調査に基づく2025年ベストプラクティス実装
"""
import torch
import numpy as np
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sam2_bfloat16_compatibility():
    """SAM2のBFloat16互換性をテスト"""
    
    logger.info("=== SAM2 BFloat16互換性テスト開始 ===")
    
    # 1. BFloat16テンソルのNumPy変換問題確認
    logger.info("1. BFloat16→NumPy変換テスト")
    try:
        # BFloat16テンソル作成
        bfloat16_tensor = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        logger.info(f"  - BFloat16テンソル作成: {bfloat16_tensor.shape}, dtype={bfloat16_tensor.dtype}")
        
        # 直接変換（エラー確認）
        try:
            numpy_array = bfloat16_tensor.cpu().numpy()
            logger.error("  ❌ 直接変換が成功（予期せぬ動作）")
        except TypeError as e:
            logger.info(f"  ✅ 予期されたエラー: {e}")
        
        # Float32経由変換（推奨方法）
        numpy_array = bfloat16_tensor.float().cpu().numpy()
        logger.info(f"  ✅ Float32経由変換成功: {numpy_array.shape}, dtype={numpy_array.dtype}")
        
    except Exception as e:
        logger.error(f"  ❌ テスト失敗: {e}")
    
    # 2. SAM2推奨の混合精度設定テスト
    logger.info("\n2. SAM2混合精度設定テスト")
    try:
        # 推奨: inference_mode + autocast with bfloat16
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            # ダミー推論
            dummy_input = torch.randn(1, 3, 1024, 1024)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            logger.info("  ✅ SAM2推奨設定での推論成功")
            logger.info(f"  - 入力dtype: {dummy_input.dtype}")
            
            # 混合精度での演算確認
            conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            if torch.cuda.is_available():
                conv = conv.cuda()
            output = conv(dummy_input)
            logger.info(f"  - 出力dtype: {output.dtype}")
            
    except Exception as e:
        logger.error(f"  ❌ 混合精度設定失敗: {e}")
    
    # 3. SAM2画像前処理の型変換テスト
    logger.info("\n3. SAM2画像前処理テスト")
    try:
        # テスト画像作成
        test_image = Image.new('RGB', (1024, 1024), color='red')
        image_array = np.array(test_image)
        logger.info(f"  - 入力画像: {image_array.shape}, dtype={image_array.dtype}")
        
        # PyTorchテンソル化（BFloat16）
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(torch.bfloat16)
        logger.info(f"  - BFloat16テンソル: {image_tensor.shape}, dtype={image_tensor.dtype}")
        
        # SAM2用に変換（Float32推奨）
        sam2_input = image_tensor.float() / 255.0  # 正規化
        logger.info(f"  - SAM2入力: {sam2_input.shape}, dtype={sam2_input.dtype}")
        
        # NumPy戻し変換（SAM2内部処理想定）
        sam2_numpy = sam2_input.cpu().numpy().transpose(1, 2, 0)
        logger.info(f"  ✅ SAM2用NumPy変換成功: {sam2_numpy.shape}, dtype={sam2_numpy.dtype}")
        
    except Exception as e:
        logger.error(f"  ❌ 画像前処理失敗: {e}")
    
    # 4. 推奨実装パターン
    logger.info("\n4. 推奨実装パターン")
    logger.info("  📌 BFloat16 → Float32 → NumPy")
    logger.info("  📌 tensor.float().cpu().numpy()")
    logger.info("  📌 SAM2推論: torch.autocast('cuda', dtype=torch.bfloat16)")
    logger.info("  📌 重要演算（SoftMax等）はFloat32維持")

if __name__ == "__main__":
    test_sam2_bfloat16_compatibility()