#!/usr/bin/env python3
"""
DTensor問題解決 - DeepSpeed直接使用戦略
accelerateを bypassing してDTensor問題を回避
"""

import json
import os
import subprocess
import sys

def create_deepspeed_config():
    """DeepSpeed ZeRO Stage 1設定を作成"""
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto", 
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "none"
            },
            "offload_param": {
                "device": "none"
            }
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "wall_clock_breakdown": False
    }
    
    with open("ds_config_dtensor_fix.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ DeepSpeed設定ファイル作成: ds_config_dtensor_fix.json")
    return "ds_config_dtensor_fix.json"

def create_deepspeed_train_script():
    """DeepSpeed用学習スクリプト作成"""
    script_content = '''#!/usr/bin/env python3
"""
DeepSpeed直接使用でDTensor問題回避版train.py
"""

import deepspeed
import torch
from model.gemma_lisa import LisaGemmaForCausalLM
from transformers import GemmaTokenizer, TrainingArguments
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, 
                       help="local rank passed from distributed launcher")
    
    # DeepSpeed args
    parser = deepspeed.add_config_arguments(parser)
    
    # Model args
    parser.add_argument("--model_name", type=str, 
                       default="google/gemma-3-4b-it")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_steps", type=int, default=100)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # DeepSpeed初期化
    deepspeed.init_distributed()
    
    print("🔧 モデル初期化（DTensor回避モード）")
    
    # tokenizer読み込み
    tokenizer = GemmaTokenizer.from_pretrained(args.model_name)
    
    # LISA-Gemmaモデル初期化（DTensor環境外）
    model = LisaGemmaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # DeepSpeedが制御
    )
    
    print("🚀 モデル読み込み成功")
    
    # 語彙サイズ拡張（DTensor環境外で実行）
    current_vocab_size = model.gemma_model.config.vocab_size
    required_vocab_size = current_vocab_size + 1 + 256  # SEG + IMG tokens
    
    print(f"📊 語彙サイズ拡張: {current_vocab_size} → {required_vocab_size}")
    model.gemma_model.resize_token_embeddings(required_vocab_size, mean_resizing=False)
    print("✅ 語彙サイズ拡張成功（DTensorエラー回避）")
    
    # オプティマイザー
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # DeepSpeedエンジン初期化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer
    )
    
    print("🎯 DeepSpeed初期化成功")
    
    # 簡単な学習ループ
    model_engine.train()
    for step in range(args.max_steps):
        # ダミーデータ
        input_ids = torch.randint(0, required_vocab_size, (2, 50)).to(model_engine.device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        model_engine.backward(loss)
        model_engine.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print("🎉 DeepSpeed学習完了！DTensor問題を回避して正常動作")
    
    # モデル保存
    model_engine.save_checkpoint(args.output_dir)
    print(f"💾 チェックポイント保存: {args.output_dir}")

if __name__ == "__main__":
    main()
'''
    
    with open("train_deepspeed_dtensor_fix.py", "w") as f:
        f.write(script_content)
    
    print("✅ DeepSpeed学習スクリプト作成: train_deepspeed_dtensor_fix.py")
    return "train_deepspeed_dtensor_fix.py"

def main():
    print("🎯 DTensor問題解決 - DeepSpeed直接使用戦略")
    print("=" * 60)
    
    # Step 1: DeepSpeed設定作成
    print("\n📦 Step 1: DeepSpeed設定作成")
    config_file = create_deepspeed_config()
    
    # Step 2: DeepSpeed学習スクリプト作成
    print("\n🛠️ Step 2: DeepSpeed学習スクリプト作成")
    train_script = create_deepspeed_train_script()
    
    # Step 3: 実行可能にする
    print("\n🔧 Step 3: スクリプトを実行可能にする")
    os.chmod(train_script, 0o755)
    
    # Step 4: 使用方法表示
    print("\n🚀 使用方法:")
    print("=" * 40)
    print("# Lambda Cloud A100*1での実行:")
    print(f"deepspeed --num_gpus=1 {train_script} --deepspeed {config_file}")
    print()
    print("# Lambda Cloud A100*8での実行:")  
    print(f"deepspeed --num_gpus=8 {train_script} --deepspeed {config_file}")
    print()
    print("🎯 利点:")
    print("- accelerateを完全にバイパス")
    print("- DTensor環境を回避")
    print("- resize_token_embeddingsが正常動作")
    print("- ZeRO Stage 1で効率的な分散学習")
    
    print("\n📋 作成されたファイル:")
    print(f"- {config_file}")
    print(f"- {train_script}")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1) 