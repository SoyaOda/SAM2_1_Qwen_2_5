#!/usr/bin/env python3
"""
シングルエンコーダー構成のデータセット動作確認テスト
第1部フェーズ1の実装が正しく動作することを検証
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import HybridDataset, collate_fn
import config_linux

def visualize_sample(sample, idx, output_dir):
    """サンプルの可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像内の実際のコンテンツ領域を検出（非ゼロピクセル）
    sam_image = sample['sam_pixel_values']
    if sam_image.dim() == 3 and sam_image.shape[0] == 3:
        # 正規化を逆変換して実際の画像を取得
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        sam_image_denorm = sam_image * std + mean
        sam_image_denorm = torch.clamp(sam_image_denorm, 0, 1)
        
        # 黒色でないピクセルを検出（パディング領域を特定）
        non_black = (sam_image_denorm.sum(dim=0) > 0.01).float()
        non_zero_indices = torch.nonzero(non_black, as_tuple=True)
        if len(non_zero_indices[0]) > 0:
            min_y, max_y = non_zero_indices[0].min().item(), non_zero_indices[0].max().item()
            min_x, max_x = non_zero_indices[1].min().item(), non_zero_indices[1].max().item()
            actual_size = (max_y - min_y + 1, max_x - min_x + 1)
            image_bounds = f"[{min_x}, {min_y}] - [{max_x}, {max_y}]"
        else:
            actual_size = (0, 0)
            image_bounds = "No content"
    else:
        actual_size = (0, 0)
        image_bounds = "Invalid"
    
    # マスクの情報を取得（Original-LISA準拠：マスクは元のサイズ）
    if sample['has_mask'] and sample['ground_truth_mask'] is not None:
        mask = sample['ground_truth_mask'].squeeze()
        mask_size = mask.shape
        mask_nonzero = torch.nonzero(mask > 0, as_tuple=True)
        if len(mask_nonzero[0]) > 0:
            mask_min_y, mask_max_y = mask_nonzero[0].min().item(), mask_nonzero[0].max().item()
            mask_min_x, mask_max_x = mask_nonzero[1].min().item(), mask_nonzero[1].max().item()
            mask_bounds = f"[{mask_min_x}, {mask_min_y}] - [{mask_max_x}, {mask_max_y}]"
            mask_info = f"サイズ: {mask_size}, 範囲: {mask_bounds}"
        else:
            mask_info = f"サイズ: {mask_size}, 内容なし"
    else:
        mask_info = "マスクなし"
    
    # 元画像サイズを取得
    original_size = sample.get('original_size', 'Unknown')
    
    # テキスト情報を保存
    text_info = f"""
サンプル {idx}:
元のプロンプト: {sample['text_prompt']}
フォーマット済みプロンプト: {sample['formatted_prompt']}
入力IDの長さ: {sample['input_ids'].shape}
SEGトークンマスク: {sample['seg_token_mask'].sum().item()} 個
画像あり: {'images_for_llama' not in sample}  # Llama用画像がないことを確認
元画像サイズ: {original_size}
SAM画像サイズ: {sample['sam_pixel_values'].shape}
実際の画像コンテンツサイズ: {actual_size[0]} x {actual_size[1]} (パディング除く)
画像コンテンツ位置: {image_bounds}
マスクあり: {sample['has_mask']}
マスク情報: {mask_info}
※ Original-LISA準拠: マスクは元サイズ保持、モデル内で変換
"""
    
    with open(os.path.join(output_dir, f'sample_{idx}_info.txt'), 'w', encoding='utf-8') as f:
        f.write(text_info)
    
    # SAM画像を可視化
    sam_image = sample['sam_pixel_values']
    if sam_image.dim() == 3 and sam_image.shape[0] == 3:
        # 正規化を逆変換（SAM公式パラメータ）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        sam_image = sam_image * std + mean
        sam_image = torch.clamp(sam_image, 0, 1)
        
        # CHW -> HWC
        sam_image_np = sam_image.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(15, 5))
        
        # SAM画像
        plt.subplot(1, 3, 1)
        plt.imshow(sam_image_np)
        plt.title('SAM Image (1024x1024)')
        plt.axis('off')
        
        # マスク（もしあれば）
        if sample['has_mask'] and sample['ground_truth_mask'] is not None:
            plt.subplot(1, 3, 2)
            mask = sample['ground_truth_mask'].squeeze().cpu().numpy()
            plt.imshow(mask, cmap='gray')
            plt.title(f'Ground Truth Mask\n(元サイズ: {mask.shape})')
            plt.axis('off')
            
            # 注意書き
            plt.subplot(1, 3, 3)
            plt.text(0.5, 0.5, 
                     f'Original-LISA準拠:\n\n'
                     f'画像: {sam_image_np.shape[:2]} (リサイズ済み)\n'
                     f'マスク: {mask.shape} (元サイズ保持)\n\n'
                     f'モデル内でSAMが\nマスクを適切に変換します', 
                     ha='center', va='center', fontsize=10,
                     transform=plt.gca().transAxes)
            plt.title('アーキテクチャ説明')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}_visualization.png'))
        plt.close()
    
    print(text_info)

def test_dataset():
    """データセットの動作確認"""
    print("=== シングルエンコーダー構成のデータセット動作確認 ===")
    
    # Llama4プロセッサーを初期化
    print("\n1. Llama4プロセッサーの初期化...")
    try:
        processor = AutoProcessor.from_pretrained(
            config_linux.LLAMA_MODEL_ID,
            trust_remote_code=True
        )
        print("✅ プロセッサー初期化成功")
    except Exception as e:
        print(f"⚠️ プロセッサー初期化エラー: {e}")
        # フォールバック
        from transformers import AutoTokenizer
        processor = type('MockProcessor', (), {
            'tokenizer': AutoTokenizer.from_pretrained(config_linux.LLAMA_MODEL_ID)
        })()
        print("✅ モックプロセッサーで代替")
    
    # データセットを作成（小規模テスト用）
    print("\n2. HybridDatasetの作成...")
    try:
        dataset = HybridDataset(
            base_image_dir=config_linux.DATASET_BASE_DIR,
            llama_processor=processor,
            samples_per_epoch=10,  # テスト用に少数
            dataset="reason_seg",  # ReasonSegのみでテスト
            sample_rate=[1.0],
        )
        print(f"✅ データセット作成成功: {len(dataset)} サンプル")
    except Exception as e:
        print(f"❌ データセット作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # DataLoaderを作成
    print("\n3. DataLoaderの作成...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # デバッグのため0
    )
    
    # サンプルを取得して検証
    print("\n4. サンプルの取得と検証...")
    output_dir = "test_single_encoder_output"
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:  # 最初の2バッチのみテスト
            break
        
        print(f"\n--- バッチ {batch_idx} ---")
        print(f"バッチキー: {list(batch.keys())}")
        
        # シングルエンコーダー構成の検証
        assert 'sam_pixel_values' in batch, "sam_pixel_valuesがありません"
        assert 'images_for_llama' not in batch, "images_for_llamaが含まれています（削除されているはず）"
        assert 'images_for_sam' not in batch, "images_for_samが含まれています（sam_pixel_valuesに置き換えられているはず）"
        
        print(f"✅ シングルエンコーダー構成の検証OK")
        print(f"  - sam_pixel_values: {batch['sam_pixel_values'].shape}")
        print(f"  - input_ids: {batch['input_ids'].shape}")
        print(f"  - formatted_prompts[0]: {batch['formatted_prompts'][0][:100]}...")
        
        # マスク情報の確認
        if batch['has_mask'][0]:
            mask = batch['masks_list'][0]
            print(f"  - マスク[0]のサイズ: {mask.shape} (元サイズ保持)")
        
        # 個別サンプルの可視化
        for i in range(len(batch['input_ids'])):
            sample = {
                'sam_pixel_values': batch['sam_pixel_values'][i],
                'input_ids': batch['input_ids'][i],
                'seg_token_mask': batch['seg_token_mask'][i],
                'has_mask': batch['has_mask'][i],
                'ground_truth_mask': batch['masks_list'][i] if batch['has_mask'][i] else None,  # リストから取得
                'text_prompt': batch['text_prompts'][i],
                'formatted_prompt': batch['formatted_prompts'][i],
                'original_size': batch.get('original_sizes', [None] * len(batch['input_ids']))[i],
            }
            visualize_sample(sample, batch_idx * 2 + i, output_dir)
    
    print(f"\n✅ テスト完了！結果は {output_dir} に保存されました。")

def test_tokenizer_special_tokens():
    """トークナイザの特殊トークン確認"""
    print("\n=== トークナイザ特殊トークンテスト ===")
    
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
    
    tokenizer = processor.tokenizer
    
    # 特殊トークンの追加前の状態を確認
    print(f"語彙サイズ（追加前）: {len(tokenizer)}")
    
    # データセット初期化（[SEG]トークンのみ追加）
    from utils.dataset import setup_seg_token
    seg_token_idx = setup_seg_token(tokenizer, "[SEG]")
    # Llama4のネイティブマルチモーダルでは<image>トークンは不要
    
    print(f"語彙サイズ（追加後）: {len(tokenizer)}")
    
    # テキストのトークン化テスト（<image>なし）
    test_text = "USER: Segment the cat in this image. [SEG]"
    tokens = tokenizer(test_text, add_special_tokens=False)
    print(f"\nテストテキスト: {test_text}")
    print(f"トークンID: {tokens['input_ids']}")
    print(f"デコード結果: {tokenizer.decode(tokens['input_ids'])}")
    
    # 特殊トークンの位置を確認
    input_ids = torch.tensor(tokens['input_ids'])
    seg_mask = (input_ids == seg_token_idx)
    
    print(f"\n[SEG]トークンの位置: {seg_mask.nonzero(as_tuple=True)[0].tolist()}")

if __name__ == "__main__":
    # 特殊トークンのテスト
    test_tokenizer_special_tokens()
    
    # データセットのテスト
    test_dataset()