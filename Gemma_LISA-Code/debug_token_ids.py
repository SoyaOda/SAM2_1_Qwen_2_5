#!/usr/bin/env python3
"""
トークンIDの範囲を確認するデバッグスクリプト
"""

import torch
from transformers import AutoProcessor
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import HybridDataset, collate_fn

def get_config():
    """動的設定読み込み"""
    config_candidates = ['config_linux', 'config_small_test']
    
    for config_name in config_candidates:
        try:
            config_module = __import__(config_name)
            print(f"✓ 設定ファイルを使用: {config_name}")
            return config_module
        except ImportError:
            continue
    
    raise ImportError("利用可能な設定ファイルが見つかりません")

def main():
    config = get_config()
    
    print("="*60)
    print("トークンID範囲チェック")
    print("="*60)
    
    # プロセッサーの初期化
    processor = AutoProcessor.from_pretrained('google/gemma-3-4b-it')
    tokenizer = processor.tokenizer
    
    print(f"\n1. トークナイザー情報:")
    print(f"   - 語彙サイズ: {len(tokenizer)}")
    print(f"   - 語彙サイズ (vocab_size): {tokenizer.vocab_size}")
    
    # 特殊トークンの確認
    print(f"\n2. 特殊トークン:")
    print(f"   - BOS token ID: {tokenizer.bos_token_id}")
    print(f"   - EOS token ID: {tokenizer.eos_token_id}")
    print(f"   - PAD token ID: {tokenizer.pad_token_id}")
    
    # SEGトークンの確認
    if "[SEG]" in tokenizer.get_vocab():
        seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
        print(f"   - [SEG] token ID: {seg_token_id}")
    else:
        print("   - [SEG] token: 未追加")
    
    # データセットから1バッチ取得
    print(f"\n3. データセットからサンプル取得:")
    
    dataset = HybridDataset(
        base_image_dir=config.DATASET_BASE_DIR,
        gemma_processor=processor,
        dataset='reason_seg',
        samples_per_epoch=2
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    
    print(f"   - input_ids shape: {input_ids.shape}")
    print(f"   - 最小値: {input_ids.min().item()}")
    print(f"   - 最大値: {input_ids.max().item()}")
    
    # 範囲外のトークンをチェック
    vocab_size = len(tokenizer)
    out_of_range = (input_ids >= vocab_size) | (input_ids < -1000)  # 負の値は画像トークン
    
    if out_of_range.any():
        print(f"\n⚠️  範囲外のトークンID検出:")
        out_of_range_ids = input_ids[out_of_range].unique()
        for token_id in out_of_range_ids:
            count = (input_ids == token_id).sum().item()
            print(f"   - ID {token_id}: {count}個")
    else:
        print("\n✅ すべてのトークンIDが有効範囲内")
    
    # 負のトークンIDの確認
    negative_ids = input_ids < 0
    if negative_ids.any():
        print(f"\n4. 負のトークンID (画像トークン):")
        unique_negative = input_ids[negative_ids].unique()
        print(f"   - 種類: {len(unique_negative)}")
        print(f"   - 範囲: {unique_negative.min().item()} ~ {unique_negative.max().item()}")
        print(f"   - 総数: {negative_ids.sum().item()}")

if __name__ == "__main__":
    main() 