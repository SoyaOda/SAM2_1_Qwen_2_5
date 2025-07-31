#!/usr/bin/env python3
"""
LISA-Llama4 デバッグ用簡易学習スクリプト
単一GPUでの動作確認用

実行方法:
python train_debug.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LISA-Llama4モデルとユーティリティ
from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
from utils.dataset import HybridDataset, collate_fn
from peft import LoraConfig, get_peft_model, TaskType
import config_linux


def parse_args():
    parser = argparse.ArgumentParser(description="LISA-Llama4 デバッグ学習")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--check_gradients", action="store_true", default=True)
    parser.add_argument("--save_checkpoint", action="store_true", default=False)
    return parser.parse_args()


def check_gradients(model, step):
    """勾配フローの確認"""
    print(f"\n--- Step {step}: 勾配チェック ---")
    
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            
            if param_norm == 0:
                zero_grad_count += 1
            
            # 主要なパラメータの勾配を表示
            if any(key in name for key in ["lora", "projector", "embed"]):
                print(f"  {name}: grad_norm={param_norm:.6f}")
    
    total_norm = total_norm ** 0.5
    print(f"総勾配ノルム: {total_norm:.6f}")
    print(f"勾配を持つパラメータ: {param_count}")
    print(f"ゼロ勾配パラメータ: {zero_grad_count}")
    
    return total_norm


def main():
    args = parse_args()
    
    print("=" * 80)
    print("LISA-Llama4 デバッグ学習")
    print("=" * 80)
    print(f"デバイス: {args.device}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"ステップ数: {args.num_steps}")
    print(f"学習率: {args.lr}")
    print("=" * 80)
    
    # モデル初期化
    print("\n1. モデル初期化中...")
    model_config = LisaLlama4Config(**config_linux.get_lisa_model_config())
    model = LisaLlama4ForCausalLM(model_config)
    
    # LoRA適用
    print("\n2. LoRA適用中...")
    lora_config_dict = config_linux.get_lora_config()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **lora_config_dict
    )
    model = get_peft_model(model, lora_config)
    
    # エンベディング層とLMヘッドを凍結
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = False
    
    # マルチモーダルプロジェクタは学習可能
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
    
    model.print_trainable_parameters()
    model = model.to(args.device)
    
    # データセット準備
    print("\n3. データセット準備中...")
    processor = model.llama_processor
    
    dataset = HybridDataset(
        base_image_dir=config_linux.DATASET_BASE_DIR,
        llama_processor=processor,
        samples_per_epoch=args.num_steps * args.batch_size,
        precision="fp32",
        dataset="reason_seg",  # デバッグ用に1種類のみ
        sample_rate=[1.0],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # デバッグ用にシングルプロセス
        collate_fn=lambda x: collate_fn(x, tokenizer=processor.tokenizer, local_rank=0)
    )
    
    # オプティマイザ
    print("\n4. オプティマイザ設定中...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # 学習ループ
    print("\n5. 学習開始")
    print("=" * 80)
    
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader, total=args.num_steps)):
        if step >= args.num_steps:
            break
        
        # データをGPUに移動
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # フォワードパス
        try:
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            
            print(f"\nStep {step}: Loss = {loss.item():.6f}")
            
            # バックワード
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配チェック
            if args.check_gradients:
                grad_norm = check_gradients(model, step)
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # パラメータ更新
            optimizer.step()
            
            total_loss += loss.item()
            
        except Exception as e:
            print(f"\n❌ Step {step} でエラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 結果サマリ
    print("\n" + "=" * 80)
    print("学習完了!")
    print(f"平均損失: {total_loss / args.num_steps:.6f}")
    print("=" * 80)
    
    # チェックポイント保存（オプション）
    if args.save_checkpoint:
        checkpoint_path = f"./debug_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': args.num_steps,
            'loss': total_loss / args.num_steps,
        }, checkpoint_path)
        print(f"\nチェックポイント保存: {checkpoint_path}")


if __name__ == "__main__":
    # 警告を抑制
    import warnings
    warnings.filterwarnings("ignore")
    
    # CUDA設定
    torch.cuda.empty_cache()
    
    # メイン実行
    main()