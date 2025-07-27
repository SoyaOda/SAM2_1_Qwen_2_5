#!/usr/bin/env python3
"""
画像とマスクの前処理アライメントをデバッグ
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import preprocess_sam_image, preprocess_mask
import config_linux

def create_test_image_and_mask(size=(600, 400)):
    """テスト用の画像とマスクを作成"""
    # 画像を作成（赤い四角形）
    image = Image.new('RGB', size, color='blue')
    draw = ImageDraw.Draw(image)
    # 左上に赤い四角形を描画
    rect_coords = [50, 50, 200, 150]  # [left, top, right, bottom]
    draw.rectangle(rect_coords, fill='red')
    
    # マスクを作成（同じ位置）
    mask = np.zeros(size[::-1], dtype=np.uint8)  # (h, w)
    mask[50:150, 50:200] = 255
    
    return image, mask, rect_coords

def visualize_preprocessing():
    """前処理の可視化"""
    # テスト画像とマスクを作成
    original_size = (600, 400)
    image, mask, rect_coords = create_test_image_and_mask(original_size)
    
    print(f"元の画像サイズ: {image.size}")
    print(f"元のマスクサイズ: {mask.shape}")
    print(f"赤い四角形の位置: {rect_coords}")
    
    # 前処理を実行
    sam_image_tensor = preprocess_sam_image(image, 1024)
    sam_mask_tensor = preprocess_mask(mask, 1024, original_size[::-1])  # (w,h) -> (h,w)
    
    print(f"\n前処理後:")
    print(f"SAM画像テンソル: {sam_image_tensor.shape}")
    print(f"SAMマスクテンソル: {sam_mask_tensor.shape}")
    
    # 可視化のために逆正規化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    sam_image_denorm = sam_image_tensor * std + mean
    sam_image_denorm = torch.clamp(sam_image_denorm, 0, 1)
    
    # プロット
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元の画像とマスク
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image (600x400)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 前処理後の画像とマスク
    sam_image_np = sam_image_denorm.permute(1, 2, 0).numpy()
    axes[0, 1].imshow(sam_image_np)
    axes[0, 1].set_title('SAM Image (1024x1024)')
    axes[0, 1].grid(True, alpha=0.3)
    
    sam_mask_np = sam_mask_tensor.squeeze().numpy()
    axes[1, 1].imshow(sam_mask_np, cmap='gray')
    axes[1, 1].set_title('SAM Mask (1024x1024)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # オーバーレイ（位置確認）
    # 画像の上にマスクを半透明で重ねる
    overlay = sam_image_np.copy()
    mask_overlay = np.zeros_like(overlay)
    mask_overlay[:, :, 0] = sam_mask_np  # 赤チャンネルにマスクを設定
    overlay = overlay * 0.7 + mask_overlay * 0.3
    
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Overlay (Image + Mask)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 非ゼロピクセルの位置を検出
    image_nonzero = (sam_image_np.sum(axis=2) > 0.01)
    mask_nonzero = (sam_mask_np > 0)
    
    image_y, image_x = np.where(image_nonzero)
    mask_y, mask_x = np.where(mask_nonzero)
    
    if len(image_y) > 0:
        image_bounds = f"Image content: x=[{image_x.min()}, {image_x.max()}], y=[{image_y.min()}, {image_y.max()}]"
    else:
        image_bounds = "Image content: None"
        
    if len(mask_y) > 0:
        mask_bounds = f"Mask content: x=[{mask_x.min()}, {mask_x.max()}], y=[{mask_y.min()}, {mask_y.max()}]"
    else:
        mask_bounds = "Mask content: None"
    
    axes[1, 2].text(0.1, 0.5, f"{image_bounds}\n{mask_bounds}", 
                    transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Content Bounds')
    
    plt.tight_layout()
    plt.savefig('debug_preprocessing_alignment.png', dpi=150)
    print(f"\n結果を保存: debug_preprocessing_alignment.png")
    
    # 詳細なデバッグ情報
    print(f"\n詳細な位置情報:")
    print(f"  {image_bounds}")
    print(f"  {mask_bounds}")
    
    # スケーリング計算の確認
    scale = 1024 / max(original_size)
    new_w = int(original_size[0] * scale)
    new_h = int(original_size[1] * scale)
    print(f"\nスケーリング計算:")
    print(f"  元のサイズ: {original_size}")
    print(f"  スケール: {scale:.3f}")
    print(f"  リサイズ後: ({new_w}, {new_h})")
    print(f"  パディング: 右に{1024-new_w}px, 下に{1024-new_h}px")

def test_real_dataset_sample():
    """実際のデータセットサンプルでテスト"""
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    from utils.dataset import HybridDataset, collate_fn
    
    print("\n=== 実際のデータセットサンプルのテスト ===")
    
    # プロセッサーを初期化
    try:
        processor = AutoProcessor.from_pretrained(
            config_linux.LLAMA_MODEL_ID,
            trust_remote_code=True
        )
    except:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config_linux.LLAMA_MODEL_ID)
        processor = type('MockProcessor', (), {'tokenizer': tokenizer})()
    
    # データセット作成
    dataset = HybridDataset(
        base_image_dir=config_linux.DATASET_BASE_DIR,
        llama_processor=processor,
        samples_per_epoch=1,
        dataset="reason_seg",
        sample_rate=[1.0],
    )
    
    # 最初のサンプルを取得
    sample = dataset[0]
    
    # 画像パスを取得して元画像を読み込む
    if 'image_path' in sample and sample['image_path']:
        print(f"画像パス: {sample['image_path']}")
        try:
            original_image = Image.open(sample['image_path']).convert('RGB')
            print(f"元の画像サイズ: {original_image.size}")
        except:
            print("元画像の読み込みに失敗")

if __name__ == "__main__":
    # テスト画像での検証
    visualize_preprocessing()
    
    # 実際のデータセットでの検証
    test_real_dataset_sample()