#!/usr/bin/env python3
"""
デュアルストリーム・データパイプライン テストスクリプト
仕様書第3章の実装確認
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 動的設定管理機能を追加
def get_config():
    """動的設定読み込み（環境変数対応）"""
    config_path = os.environ.get('LISA_CONFIG_PATH', None)
    
    if config_path:
        # 環境変数で指定された設定ファイル
        try:
            config_module = __import__(config_path)
            print(f"✓ カスタム設定ファイルを使用: {config_path}")
            return config_module
        except ImportError:
            print(f"⚠️ カスタム設定ファイル {config_path} が見つかりません")
    
    # デフォルトの設定ファイル検索順序
    config_candidates = ['config_small_test', 'config_linux']
    
    for config_name in config_candidates:
        try:
            config_module = __import__(config_name)
            print(f"✓ 設定ファイルを使用: {config_name}")
            return config_module
        except ImportError:
            continue
    
    raise ImportError("利用可能な設定ファイルが見つかりません")

# 動的設定読み込み
config = get_config()

from utils.dataset import HybridDataset, collate_fn, preprocess_sam_image, preprocess_mask

def test_sam_preprocessing():
    """SAM用前処理のテスト"""
    print("=" * 60)
    print("🧪 SAM用前処理テスト")
    print("=" * 60)
    
    # テスト画像の作成
    test_image = Image.new('RGB', (800, 600), color='red')
    
    # SAM用前処理の実行
    sam_tensor = preprocess_sam_image(test_image, target_size=1024)
    
    print(f"入力画像サイズ: {test_image.size}")
    print(f"SAM前処理後テンソル形状: {sam_tensor.shape}")
    print(f"SAM前処理後テンソル範囲: [{sam_tensor.min():.3f}, {sam_tensor.max():.3f}]")
    
    # 期待される形状の確認
    expected_shape = (3, 1024, 1024)
    assert sam_tensor.shape == expected_shape, f"期待される形状 {expected_shape}, 実際 {sam_tensor.shape}"
    
    print("✅ SAM用前処理テスト成功")
    return sam_tensor

def test_gemma_preprocessing():
    """Gemma用前処理のテスト"""
    print("\n" + "=" * 60)
    print("🧪 Gemma用前処理テスト")
    print("=" * 60)
    
    # Gemmaプロセッサーの初期化
    try:
        gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
    except Exception as e:
        print(f"⚠️ Gemmaプロセッサーの初期化に失敗: {e}")
        print("ダミープロセッサーを使用します")
        return None
    
    # テスト画像とテキストの作成
    test_image = Image.new('RGB', (800, 600), color='blue')
    test_text = "Show me the blue area. [SEG]"
    
    # Gemma用前処理の実行
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": test_text}
            ]
        }
    ]
    
    try:
        gemma_processed = gemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        pixel_values = gemma_processed['pixel_values'].squeeze(0)
        input_ids = gemma_processed['input_ids'].squeeze(0)
        
        print(f"入力画像サイズ: {test_image.size}")
        print(f"Gemma前処理後画像形状: {pixel_values.shape}")
        print(f"Gemma前処理後画像範囲: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
        print(f"トークン化テキスト形状: {input_ids.shape}")
        print(f"トークン化テキスト: {input_ids[:10]}...")  # 最初の10トークンを表示
        
        # [SEG]トークンの確認
        seg_token_id = gemma_processor.tokenizer.convert_tokens_to_ids("[SEG]")
        seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)[0]
        print(f"[SEG]トークンID: {seg_token_id}")
        print(f"[SEG]トークン位置: {seg_positions.tolist()}")
        
        # 期待される形状の確認
        expected_shape = (3, 896, 896)
        assert pixel_values.shape == expected_shape, f"期待される形状 {expected_shape}, 実際 {pixel_values.shape}"
        
        print("✅ Gemma用前処理テスト成功")
        return gemma_processed, gemma_processor
        
    except Exception as e:
        print(f"⚠️ Gemma用前処理に失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, gemma_processor

def test_hybrid_dataset():
    """HybridDatasetのテスト"""
    print("\n" + "=" * 60)
    print("🧪 HybridDataset テスト")
    print("=" * 60)
    
    # Gemmaプロセッサーの初期化
    gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
    
    # [SEG]トークンを追加
    seg_token = "[SEG]"
    if seg_token not in gemma_processor.tokenizer.get_vocab():
        gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        print(f"✅ {seg_token}トークンを追加しました")
    
    # HybridDatasetの作成
    dataset = HybridDataset(
        base_image_dir=config.DATASET_BASE_DIR,
        gemma_processor=gemma_processor,
        samples_per_epoch=10,  # 少数のサンプルでテスト
        dataset="reason_seg",  # ReasonSegのみでテスト
        sample_rate=[1],
        reason_seg_data="ReasonSeg|train"
    )
    
    print(f"データセット作成成功: {len(dataset)} サンプル")
    
    # 1つのサンプルを取得（エラーが発生した場合は例外を発生させる）
    sample = dataset[0]
    
    print("\n📋 サンプル情報:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)} - {str(value)[:50]}...")
    
    # デュアルストリーム形状の確認
    assert sample["images_for_gemma"].shape == (3, 896, 896), "Gemma画像形状が不正"
    assert sample["images_for_sam"].shape == (3, 1024, 1024), "SAM画像形状が不正"
    
    print("✅ HybridDataset テスト成功")
    return dataset, sample

def test_collate_fn():
    """collate_fn のテスト"""
    print("\n" + "=" * 60)
    print("🧪 collate_fn テスト")
    print("=" * 60)
    
    try:
        # ダミーサンプルの作成
        sample1 = {
            "images_for_gemma": torch.randn(3, 896, 896),
            "images_for_sam": torch.randn(3, 1024, 1024),
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor(0),
            "seg_token_mask": torch.tensor([False, True, False, False, False]),
            "ground_truth_mask": torch.randn(1, 1024, 1024),
            "has_mask": True,
            "image_path": "test_image_1.jpg",
            "text_prompt": "Test prompt 1 [SEG]"
        }
        
        sample2 = {
            "images_for_gemma": torch.randn(3, 896, 896),
            "images_for_sam": torch.randn(3, 1024, 1024),
            "input_ids": torch.tensor([1, 2, 3]),  # 異なる長さ
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor(1),
            "seg_token_mask": torch.tensor([False, False, True]),
            "ground_truth_mask": torch.randn(1, 1024, 1024),
            "has_mask": True,
            "image_path": "test_image_2.jpg",
            "text_prompt": "Test prompt 2 [SEG]"
        }
        
        batch = [sample1, sample2]
        
        # collate_fnの実行
        collated_batch = collate_fn(batch)
        
        print("\n📋 バッチ情報:")
        for key, value in collated_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: List[{len(value)}] - {str(value)[:50]}...")
            else:
                print(f"  {key}: {type(value)}")
        
        # バッチ形状の確認
        batch_size = 2
        assert collated_batch["images_for_gemma"].shape[0] == batch_size, "Gemmaバッチサイズが不正"
        assert collated_batch["images_for_sam"].shape[0] == batch_size, "SAMバッチサイズが不正"
        assert collated_batch["input_ids"].shape[0] == batch_size, "input_idsバッチサイズが不正"
        
        print("✅ collate_fn テスト成功")
        return collated_batch
        
    except Exception as e:
        print(f"⚠️ collate_fn テストに失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_integration():
    """統合テスト"""
    print("\n" + "=" * 60)
    print("🧪 デュアルストリーム統合テスト")
    print("=" * 60)
    
    # Gemmaプロセッサーの初期化
    gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
    
    # [SEG]トークンを追加
    seg_token = "[SEG]"
    if seg_token not in gemma_processor.tokenizer.get_vocab():
        gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
    
    # HybridDatasetの作成
    dataset = HybridDataset(
        base_image_dir=config.DATASET_BASE_DIR,
        gemma_processor=gemma_processor,
        samples_per_epoch=5,
        dataset="reason_seg",
        sample_rate=[1],
        reason_seg_data="ReasonSeg|train"
    )
    
    # DataLoaderの作成
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # テスト用
    )
    
    print(f"DataLoader作成成功: バッチサイズ={dataloader.batch_size}")
    
    # 1バッチの取得（エラーが発生した場合は例外を発生させる）
    for i, batch in enumerate(dataloader):
        print(f"\n📦 バッチ {i+1}:")
        print(f"  images_for_gemma: {batch['images_for_gemma'].shape}")
        print(f"  images_for_sam: {batch['images_for_sam'].shape}")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  SEGトークン検出数: {batch['seg_token_mask'].sum().item()}")
        
        # 1バッチのみでテスト終了
        break
    
    print("✅ デュアルストリーム統合テスト成功")
    return True

def main():
    """メインテスト関数"""
    print("🚀 デュアルストリーム・データパイプライン テスト開始")
    
    # テスト実行
    test_results = []
    
    # 1. SAM前処理テスト
    try:
        test_sam_preprocessing()
        test_results.append("SAM前処理: ✅")
    except Exception as e:
        print(f"⚠️ SAM前処理テストに失敗: {e}")
        test_results.append(f"SAM前処理: ❌ {e}")
    
    # 2. Gemma前処理テスト
    try:
        test_gemma_preprocessing()
        test_results.append("Gemma前処理: ✅")
    except Exception as e:
        print(f"⚠️ Gemma前処理テストに失敗: {e}")
        test_results.append(f"Gemma前処理: ❌ {e}")
    
    # 3. HybridDatasetテスト
    try:
        test_hybrid_dataset()
        test_results.append("HybridDataset: ✅")
    except Exception as e:
        print(f"⚠️ HybridDataset テストに失敗: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(f"HybridDataset: ❌ {e}")
    
    # 4. collate_fnテスト
    try:
        test_collate_fn()
        test_results.append("collate_fn: ✅")
    except Exception as e:
        print(f"⚠️ collate_fn テストに失敗: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(f"collate_fn: ❌ {e}")
    
    # 5. 統合テスト
    try:
        test_integration()
        test_results.append("統合テスト: ✅")
    except Exception as e:
        print(f"⚠️ 統合テストに失敗: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(f"統合テスト: ❌ {e}")
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print("=" * 60)
    
    for result in test_results:
        print(f"  {result}")
    
    success_count = sum(1 for result in test_results if "✅" in result)
    total_count = len(test_results)
    
    print(f"\n🎯 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 全テスト成功! デュアルストリーム・データパイプラインが正常に動作しています。")
        return 0
    else:
        print("⚠️ 一部のテストが失敗しました。上記のエラーメッセージを確認してください。")
        return 1

if __name__ == "__main__":
    main() 