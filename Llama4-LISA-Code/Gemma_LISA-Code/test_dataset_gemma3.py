#!/usr/bin/env python3
"""
LISA-Gemma3データセットパイプライン テスト
新しいデュアルストリーム・データパイプラインの動作確認
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoProcessor
from utils.dataset import HybridDataset, collate_fn
import config_linux

def test_dataset_initialization():
    """Step 1: データセットの初期化テスト"""
    print("=== Step 1: データセット初期化テスト ===")
    
    try:
        # Gemma-3プロセッサーの初期化
        print(f"Gemma-3プロセッサーをロード中: {config_linux.GEMMA_MODEL_ID}")
        gemma_processor = AutoProcessor.from_pretrained(config_linux.GEMMA_MODEL_ID)
        print("✅ Gemma-3プロセッサーのロードが成功しました")
        
        # データセットの初期化（小さなサンプル数でテスト）
        print("データセットを初期化中...")
        dataset = HybridDataset(
            base_image_dir="./datasets",  # 実際のデータセットディレクトリ（なくてもテスト可能）
            gemma_processor=gemma_processor,
            samples_per_epoch=10,  # テスト用に小さな値
            dataset="vqa",  # VQAデータセットのみでテスト
            sample_rate=[1],  # VQAのみ
        )
        
        print(f"✅ データセット初期化成功!")
        print(f"  - データセット長: {len(dataset)}")
        print(f"  - サブデータセット数: {len(dataset.all_datasets)}")
        
        return dataset, gemma_processor
        
    except Exception as e:
        print(f"❌ データセット初期化失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_dataset_getitem(dataset):
    """Step 2: データセットのアイテム取得テスト"""
    print("\n=== Step 2: データセットアイテム取得テスト ===")
    
    try:
        # ランダムなインデックスでサンプル取得
        idx = 0
        print(f"インデックス {idx} のサンプルを取得中...")
        
        sample = dataset[idx]
        image_path, image, text_prompt, mask, label = sample
        
        print(f"✅ サンプル取得成功!")
        print(f"  - 画像パス: {image_path}")
        print(f"  - 画像タイプ: {type(image)}")
        print(f"  - 画像サイズ: {image.size if hasattr(image, 'size') else 'Unknown'}")
        print(f"  - テキストプロンプト: {text_prompt[:100]}...")  # 最初の100文字
        print(f"  - マスク形状: {mask.shape}")
        print(f"  - ラベル: {label}")
        
        return sample
        
    except Exception as e:
        print(f"❌ サンプル取得失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_collate_function(dataset, gemma_processor):
    """Step 3: Collate関数のテスト"""
    print("\n=== Step 3: Collate関数テスト ===")
    
    try:
        # 小さなバッチを作成
        batch_size = 2
        print(f"バッチサイズ {batch_size} のバッチを作成中...")
        
        batch = []
        for i in range(batch_size):
            sample = dataset[i]
            batch.append(sample)
        
        print(f"✅ バッチ作成成功!")
        print(f"  - バッチサイズ: {len(batch)}")
        
        # Collate関数でバッチ処理
        print("Collate関数でバッチ処理中...")
        processed_batch = collate_fn(batch)
        
        print(f"✅ バッチ処理成功!")
        print(f"  - 入力形状: {processed_batch['input_ids'].shape}")
        print(f"  - アテンションマスク形状: {processed_batch['attention_mask'].shape}")
        print(f"  - 画像テンソル形状: {processed_batch['pixel_values'].shape}")
        print(f"  - 画像パス数: {len(processed_batch['image_paths'])}")
        print(f"  - PIL画像数: {len(processed_batch['images'])}")
        
        return processed_batch
        
    except Exception as e:
        print(f"❌ バッチ処理失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataloader_integration(dataset, gemma_processor):
    """Step 4: DataLoaderとの統合テスト"""
    print("\n=== Step 4: DataLoader統合テスト ===")
    
    try:
        from torch.utils.data import DataLoader
        
        # カスタムcollate関数を使用するDataLoader
        def custom_collate_fn(batch):
            return collate_fn(batch)
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0,  # テスト用に0
        )
        
        print(f"✅ DataLoader作成成功!")
        print(f"  - バッチサイズ: {dataloader.batch_size}")
        print(f"  - データセット長: {len(dataloader.dataset)}")
        
        # 最初のバッチを取得
        print("最初のバッチを取得中...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"✅ バッチ {batch_idx} 取得成功!")
            print(f"  - 入力ID形状: {batch['input_ids'].shape}")
            print(f"  - 画像テンソル形状: {batch['pixel_values'].shape}")
            print(f"  - アテンションマスク形状: {batch['attention_mask'].shape}")
            
            # 最初のバッチだけテスト
            break
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader統合失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Step 5: メモリ使用量の確認"""
    print("\n=== Step 5: メモリ使用量確認 ===")
    
    try:
        if torch.cuda.is_available():
            print(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
            print(f"現在のメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"最大メモリ使用量: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        else:
            print("CPU使用中")
        
        # CPU使用量も確認
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"CPUメモリ使用量: {memory_info.rss / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ メモリ使用量確認失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    print("LISA-Gemma3データセットパイプライン テスト開始")
    print("=" * 60)
    
    # Step 1: データセット初期化
    dataset, gemma_processor = test_dataset_initialization()
    if dataset is None or gemma_processor is None:
        print("データセット初期化に失敗したため、テストを終了します。")
        return False
    
    # Step 2: アイテム取得テスト
    sample = test_dataset_getitem(dataset)
    if sample is None:
        print("サンプル取得に失敗しました。")
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
    
    # Step 5: メモリ使用量確認
    test_memory_usage()
    
    print("\n" + "=" * 60)
    print("🎉 全てのデータセットテストが成功しました！")
    print("新しいGemma-3デュアルストリーム・データパイプラインが正常に動作しています。")
    print("次のステップ: 学習スクリプトの実装に進むことができます。")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 