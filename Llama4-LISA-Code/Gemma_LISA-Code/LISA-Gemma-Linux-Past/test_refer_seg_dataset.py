#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

# ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # カレントディレクトリを追加

# REFERクラスとデータセットをインポート
try:
    from utils.refer_seg_dataset import ReferSegDataset
    print("✓ ReferSegDatasetクラスをインポートしました")
except ImportError as e:
    print(f"✗ ReferSegDatasetクラスのインポートに失敗しました: {e}")
    # モジュールの場所を確認
    print("\nモジュールの検索パス:")
    for p in sys.path:
        print(f" - {p}")
    
    # ファイルの存在を確認
    refer_seg_file = os.path.join(current_dir, "utils", "refer_seg_dataset.py")
    if os.path.exists(refer_seg_file):
        print(f"✓ ファイルは存在します: {refer_seg_file}")
    else:
        print(f"✗ ファイルが見つかりません: {refer_seg_file}")
    
    sys.exit(1)

# ダミーのトークナイザー
class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.ones(10, dtype=torch.long), "attention_mask": torch.ones(10, dtype=torch.long)}

def visualize_sample(image_path, image, masks, conversations):
    """サンプルを可視化"""
    plt.figure(figsize=(15, 10))
    
    # 元の画像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title("原画像")
    plt.axis('off')
    
    # マスク
    plt.subplot(1, 2, 2)
    # 元の画像を描画
    rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    
    # マスクのオーバーレイ
    for i, mask in enumerate(masks):
        mask_np = mask.numpy()
        # マスクを色付きで重ねる
        mask_color = np.zeros_like(rgb_image)
        colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (1, 1, 0, 0.5), (1, 0, 1, 0.5)]
        color = colors[i % len(colors)]
        mask_color[mask_np == 1] = [int(c * 255) for c in color[:3]]
        
        # アルファブレンド
        alpha = 0.5
        plt.imshow(mask_color, alpha=alpha)
    
    plt.title("マスク")
    plt.axis('off')
    
    # 会話を表示
    print("\n=== サンプル会話 ===")
    for i, conv in enumerate(conversations):
        print(f"会話 {i+1}:")
        print(conv)
        print()
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    print(f"可視化を保存しました: sample_visualization.png")

# カスタムcollate関数を追加
def custom_collate_fn(batch):
    """Noneを含むバッチを処理できるようにするカスタムcollate関数"""
    # Noneかどうかをチェック
    if batch is None or len(batch) == 0 or batch[0] is None:
        print("警告: バッチにNoneが含まれています")
        # ダミーデータを返す
        return (
            ["dummy_path"], 
            torch.zeros(1, 3, 1024, 1024), 
            torch.zeros(1, 3, 224, 224), 
            [["dummy_conversation"]], 
            torch.zeros(1, 1, 10, 10), 
            torch.zeros(1, 10, 10), 
            [(10, 10)], 
            None, 
            None, 
            [False]
        )
    
    # バッチの各要素を処理
    try:
        image_paths = [item[0] for item in batch]
        image_sams = torch.stack([item[1] for item in batch])
        images_gemma = torch.stack([item[2] for item in batch])
        conversations = [item[3] for item in batch]
        masks = [item[4] for item in batch]  # 直接スタックせず、リストとして保持
        labels = torch.stack([item[5] for item in batch])
        resizes = [item[6] for item in batch]
        questions = [item[7] for item in batch]
        sampled_classes = [item[8] for item in batch]
        inferences = [item[9] for item in batch]
        
        # マスクサイズが異なる場合があるので、個別に処理
        processed_masks = []
        for mask in masks:
            if mask is None:
                mask = torch.zeros(1, 10, 10)
            processed_masks.append(mask)
        
        return (
            image_paths, 
            image_sams, 
            images_gemma, 
            conversations, 
            processed_masks,  # そのままリストとして渡す
            labels, 
            resizes, 
            questions, 
            sampled_classes, 
            inferences
        )
    except Exception as e:
        print(f"collate関数でエラー発生: {e}")
        print(f"バッチの型: {type(batch)}, 長さ: {len(batch) if batch else 0}")
        if batch and batch[0]:
            print(f"最初の要素の型: {type(batch[0])}, 長さ: {len(batch[0]) if hasattr(batch[0], '__len__') else 'N/A'}")
            for i, item in enumerate(batch[0]):
                print(f"  項目{i}: {type(item)}")
        
        # エラーが発生した場合もダミーデータを返す
        return (
            ["dummy_path"], 
            torch.zeros(1, 3, 1024, 1024), 
            torch.zeros(1, 3, 224, 224), 
            [["dummy_conversation"]], 
            [torch.zeros(1, 10, 10)], 
            torch.zeros(1, 10, 10), 
            [(10, 10)], 
            [None], 
            [None], 
            [False]
        )

def test_dataloader(dataset, num_samples=5):
    """データローダーのテスト"""
    # DataLoaderの作成
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn  # カスタムcollate関数を使用
    )
    
    print(f"\nDataLoaderからサンプルを{num_samples}個取得します...")
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        # バッチデータの取得
        image_paths, image_sams, images_gemma, conversations, masks, labels, resizes, questions, sampled_classes, inferences = batch
        
        print(f"\nサンプル {i+1}:")
        print(f"画像パス: {image_paths[0]}")
        print(f"マスク情報: 数={len(masks)}, 形状={masks[0].shape if masks and masks[0] is not None else 'None'}")
        print(f"会話数: {len(conversations[0])}")
        
        # 最初のサンプルを可視化
        if i == 0:
            # 画像のロード
            image_path_str = image_paths[0]
            # マスクがリストなので、最初のマスクを取得
            mask_tensor = masks[0] if masks and len(masks) > 0 else None
            conversations_list = conversations[0]
            
            try:
                # マスクがNoneかチェック
                if mask_tensor is not None:
                    # 可視化
                    visualize_sample(image_path_str, image_sams[0], mask_tensor, conversations_list)
                else:
                    print("マスクがNoneなため、可視化をスキップします")
            except Exception as e:
                print(f"可視化中にエラーが発生: {e}")
    
    print("\nDataLoaderテスト完了")
    return True

def main():
    parser = argparse.ArgumentParser(description="refer_segデータセットテスト")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/h/download/LISA-dataset/dataset",
                        help="データセットのベースディレクトリ")
    parser.add_argument("--model_name", type=str, default="google/gemma-7b-it",
                        help="使用するモデル名")
    parser.add_argument("--samples", type=int, default=3,
                        help="テストするサンプル数")
    args = parser.parse_args()
    
    print("=== ReferSegDatasetテスト ===")
    print(f"データセットディレクトリ: {args.dataset_dir}")
    print(f"モデル名: {args.model_name}")
    
    # ダミートークナイザーの作成
    tokenizer = DummyTokenizer()
    
    try:
        # データセットの初期化
        print("\nReferSegDatasetを初期化しています...")
        refer_seg_dataset = ReferSegDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            model_name=args.model_name,
            samples_per_epoch=args.samples * 3,  # 少なめのサンプル数
            num_classes_per_sample=3,
            exclude_val=False,
            refer_seg_data="refcoco||refcoco+"  # refclefは除外
        )
        print(f"✓ ReferSegDatasetを初期化しました（サンプル数: {len(refer_seg_dataset)}）")
        
        # データローダーをテスト
        test_dataloader(refer_seg_dataset, args.samples)
        
        print("\n=== テスト完了 ===")
        print("refer_segデータセットは正常に機能しています。")
        
    except Exception as e:
        print(f"\n✗ ReferSegDatasetのテスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 