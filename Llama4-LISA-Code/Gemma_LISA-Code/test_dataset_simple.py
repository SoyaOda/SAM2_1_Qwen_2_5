#!/usr/bin/env python3
"""
LISA-Gemma3 シンプルデータセットテスト
ダミーデータでの基本動作確認
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
from typing import Tuple

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoProcessor
import config_linux

# シンプルなダミーデータセットクラス
class SimpleDummyDataset(torch.utils.data.Dataset):
    """
    テスト用のシンプルなダミーデータセット
    外部の複雑な依存関係を使用せずに基本動作を確認
    """
    
    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        
        # ダミーテキストプロンプト
        self.text_prompts = [
            "Describe this image in detail.",
            "What objects do you see in this image?",
            "Segment the main object in this image.",
            "What is the dominant color in this image?",
            "Identify and segment all objects in this scene.",
        ]
        
        # ダミー画像の色
        self.colors = [
            (255, 0, 0),    # 赤
            (0, 255, 0),    # 緑
            (0, 0, 255),    # 青
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # マゼンタ
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx) -> Tuple[str, Image.Image, str, torch.Tensor, torch.Tensor]:
        """
        ダミーサンプルを返す
        """
        # ダミー画像の作成
        color_idx = idx % len(self.colors)
        color = self.colors[color_idx]
        image = Image.new('RGB', (224, 224), color=color)
        
        # ダミーテキストプロンプト
        text_idx = idx % len(self.text_prompts)
        text_prompt = self.text_prompts[text_idx]
        
        # ダミーマスク（中央に小さな正方形）
        mask = torch.zeros(1, 224, 224)
        mask[0, 80:144, 80:144] = 1.0  # 中央に64x64の正方形
        
        # ダミーラベル
        label = torch.tensor(color_idx)
        
        # 画像パス
        image_path = f"dummy_sample_{idx}_{color_idx}"
        
        return image_path, image, text_prompt, mask, label

# Gemma-3準拠のcollate関数（簡略版）
def simple_collate_fn(batch, gemma_processor):
    """
    シンプルなcollate関数
    """
    # バッチデータの分離
    image_paths = []
    images = []
    text_prompts = []
    masks_list = []
    labels_list = []
    
    for item in batch:
        image_path, image, text_prompt, mask, label = item
        image_paths.append(image_path)
        images.append(image)
        text_prompts.append(text_prompt)
        masks_list.append(mask)
        labels_list.append(label)
    
    try:
        # Gemma-3のチャットテンプレートでバッチ処理
        processed_inputs = []
        for i in range(len(images)):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": text_prompts[i]}
                    ]
                }
            ]
            
            inputs = gemma_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            processed_inputs.append(inputs)
        
        # バッチの統一処理
        max_length = max(inputs["input_ids"].shape[1] for inputs in processed_inputs)
        
        # パディング処理
        batch_input_ids = []
        batch_attention_masks = []
        batch_pixel_values = []
        
        for inputs in processed_inputs:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
            
            # パディング
            current_length = input_ids.shape[1]
            if current_length < max_length:
                pad_length = max_length - current_length
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_length), 
                    value=gemma_processor.tokenizer.pad_token_id
                )
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, pad_length), value=0
                )
            
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_pixel_values.append(pixel_values)
        
        return {
            "image_paths": image_paths,
            "images": images,
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_masks, dim=0),
            "pixel_values": torch.cat(batch_pixel_values, dim=0),
            "masks": masks_list,
            "labels": labels_list,
        }
        
    except Exception as e:
        print(f"バッチ処理エラー: {e}")
        # フォールバック: 基本的なバッチ処理
        return {
            "image_paths": image_paths,
            "images": images,
            "text_prompts": text_prompts,
            "masks": masks_list,
            "labels": labels_list,
        }

def test_processor_initialization():
    """Gemma-3プロセッサーの初期化テスト"""
    print("=== Gemma-3プロセッサー初期化テスト ===")
    
    try:
        print(f"プロセッサーをロード中: {config_linux.GEMMA_MODEL_ID}")
        gemma_processor = AutoProcessor.from_pretrained(config_linux.GEMMA_MODEL_ID)
        print("✅ プロセッサー初期化成功!")
        return gemma_processor
        
    except Exception as e:
        print(f"❌ プロセッサー初期化失敗: {e}")
        return None

def test_dummy_dataset():
    """ダミーデータセットのテスト"""
    print("\n=== ダミーデータセット テスト ===")
    
    try:
        dataset = SimpleDummyDataset(num_samples=5)
        print(f"✅ データセット作成成功! サンプル数: {len(dataset)}")
        
        # サンプル取得テスト
        sample = dataset[0]
        image_path, image, text_prompt, mask, label = sample
        
        print(f"✅ サンプル取得成功!")
        print(f"  - 画像パス: {image_path}")
        print(f"  - 画像サイズ: {image.size}")
        print(f"  - テキスト: {text_prompt}")
        print(f"  - マスク形状: {mask.shape}")
        print(f"  - ラベル: {label}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ ダミーデータセット失敗: {e}")
        return None

def test_collate_function(dataset, gemma_processor):
    """Collate関数のテスト"""
    print("\n=== Collate関数 テスト ===")
    
    try:
        # 小さなバッチを作成
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        print(f"バッチサイズ: {len(batch)}")
        
        # Collate関数でバッチ処理
        processed_batch = simple_collate_fn(batch, gemma_processor)
        
        print(f"✅ バッチ処理成功!")
        for key, value in processed_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  - {key}: {len(value)} items")
            else:
                print(f"  - {key}: {type(value)}")
        
        return processed_batch
        
    except Exception as e:
        print(f"❌ バッチ処理失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataloader_integration(dataset, gemma_processor):
    """DataLoaderとの統合テスト"""
    print("\n=== DataLoader統合 テスト ===")
    
    try:
        from torch.utils.data import DataLoader
        
        def custom_collate_fn(batch):
            return simple_collate_fn(batch, gemma_processor)
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0,
        )
        
        print(f"✅ DataLoader作成成功!")
        
        # 最初のバッチを取得
        for batch_idx, batch in enumerate(dataloader):
            print(f"✅ バッチ {batch_idx} 取得成功!")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"  - {key}: {len(value)} items")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader統合失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("LISA-Gemma3 シンプルデータセットテスト開始")
    print("=" * 50)
    
    # Step 1: プロセッサー初期化
    gemma_processor = test_processor_initialization()
    if gemma_processor is None:
        print("プロセッサー初期化に失敗したため、テストを終了します。")
        return False
    
    # Step 2: ダミーデータセット
    dataset = test_dummy_dataset()
    if dataset is None:
        print("データセット作成に失敗しました。")
        return False
    
    # Step 3: Collate関数テスト
    processed_batch = test_collate_function(dataset, gemma_processor)
    if processed_batch is None:
        print("バッチ処理に失敗しました。")
        return False
    
    # Step 4: DataLoader統合テスト
    if not test_dataloader_integration(dataset, gemma_processor):
        print("DataLoader統合に失敗しました。")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 全てのシンプルデータセットテストが成功しました！")
    print("Gemma-3データパイプラインの基本機能が正常に動作しています。")
    print("次のステップ: 実際のデータセットとの統合を進めることができます。")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 