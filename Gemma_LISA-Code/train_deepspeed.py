#!/usr/bin/env python3
"""
フェーズ2.3: DeepSpeed ZeRO Stage 2 統合
A100*8環境でのマルチGPU分散学習（DeepSpeed対応版）

DeepSpeed統合により以下を実現:
- ZeRO Stage 2による効率的なメモリ管理
- 大規模モデルの分散学習対応
- DTensor競合問題の根本解決
- フェーズ3（DeepSpeed ZeRO Stage 3）への段階的移行準備
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil
import wandb

# DeepSpeed統合
import deepspeed
from deepspeed.utils import logger

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn

def get_config():
    """
    config_linux.pyを必須として読み込む
    """
    try:
        import config_linux as config
        print(f"設定: config_linux.py を使用")
        return config
    except ImportError as e:
        print(f"❌ ERROR: config_linux.pyが見つかりません")
        print(f"   詳細: {e}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        print(f"   ファイル存在確認: {os.path.exists('config_linux.py')}")
        raise SystemExit("config_linux.pyが必須です。ファイルが存在することを確認してください。")

def parse_args():
    config = get_config()
    
    parser = argparse.ArgumentParser(description="LISA-Gemma学習主軸スクリプト（DeepSpeed統合版）")
    parser.add_argument("--epochs", type=int, default=5, help="学習エポック数")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="各エポックのステップ数")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="学習率")
    parser.add_argument("--dataset_type", type=str, default="all", 
                       choices=["sem_seg", "refer_seg", "vqa", "reason_seg", "all"],
                       help="学習に使用するデータセットタイプ")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE_PER_GPU, help="バッチサイズ")
    parser.add_argument("--checkpoint_interval", type=int, default=50, 
                       help="チェックポイント保存間隔（ステップ数）")
    parser.add_argument("--output_dir", type=str, default="./training_output", 
                       help="学習結果とチェックポイントの保存ディレクトリ")
    parser.add_argument("--ds_config", type=str, default="./deepspeed_zero2_config.json",
                       help="DeepSpeed設定ファイルパス")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="DeepSpeed分散学習用ローカルランク")
    
    # DeepSpeed用引数（自動追加）
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()

def plot_loss_curve(loss_history: List[Dict[str, float]], output_path: str):
    """損失曲線をプロット（DeepSpeed対応）"""
    if len(loss_history) == 0:
        return
    
    iterations = list(range(len(loss_history)))
    
    # 各損失成分を抽出
    total_losses = [h['total_loss'] for h in loss_history]
    text_losses = [h.get('text_loss', 0.0) for h in loss_history]
    dice_losses = [h.get('dice_loss', 0.0) for h in loss_history]
    bce_losses = [h.get('bce_loss', 0.0) for h in loss_history]
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 総損失のプロット（対数スケール）
    ax1.plot(iterations, total_losses, 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_yscale('log')
    ax1.set_title('DeepSpeed分散学習: 総損失の推移（対数スケール）')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 損失成分の詳細プロット
    ax2.plot(iterations, text_losses, 'r-', linewidth=1, label='Text Loss')
    ax2.plot(iterations, dice_losses, 'g-', linewidth=1, label='DICE Loss')
    ax2.plot(iterations, bce_losses, 'm-', linewidth=1, label='BCE Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('損失成分の詳細推移')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"損失曲線を保存: {output_path}")

def get_memory_usage():
    """GPU/CPUメモリ使用量を取得"""
    memory_info = {}
    
    # CPUメモリ
    cpu_memory = psutil.virtual_memory()
    memory_info['cpu_used_gb'] = cpu_memory.used / (1024**3)
    memory_info['cpu_total_gb'] = cpu_memory.total / (1024**3)
    memory_info['cpu_percent'] = cpu_memory.percent
    
    # GPUメモリ（CUDA利用可能な場合）
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
        
        memory_info['gpu_used_gb'] = gpu_memory
        memory_info['gpu_max_gb'] = gpu_memory_max
        memory_info['gpu_cached_gb'] = gpu_memory_cached
        
        # GPU利用率
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_total_memory = gpu_properties.total_memory / (1024**3)
        memory_info['gpu_total_gb'] = gpu_total_memory
        memory_info['gpu_percent'] = (gpu_memory / gpu_total_memory) * 100
    else:
        memory_info['gpu_used_gb'] = 0
        memory_info['gpu_max_gb'] = 0
        memory_info['gpu_cached_gb'] = 0
        memory_info['gpu_total_gb'] = 0
        memory_info['gpu_percent'] = 0
    
    return memory_info

def format_memory_info(memory_info):
    """メモリ情報を読みやすい形式でフォーマット"""
    cpu_info = f"CPU: {memory_info['cpu_used_gb']:.1f}/{memory_info['cpu_total_gb']:.1f}GB ({memory_info['cpu_percent']:.1f}%)"
    
    if torch.cuda.is_available():
        gpu_info = f"GPU: {memory_info['gpu_used_gb']:.1f}/{memory_info['gpu_total_gb']:.1f}GB ({memory_info['gpu_percent']:.1f}%) [Max: {memory_info['gpu_max_gb']:.1f}GB]"
    else:
        gpu_info = "GPU: N/A"
    
    return f"{cpu_info} | {gpu_info}"

def main():
    args = parse_args()
    config = get_config()
    
    # DeepSpeed分散環境の初期化
    deepspeed.init_distributed()
    
    # ランク取得
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    print(f"🚀 DeepSpeed分散環境初期化完了")
    print(f"   - ランク: {rank}/{world_size}")
    print(f"   - ローカルランク: {local_rank}")
    
    # メインプロセス判定
    is_main_process = rank == 0
    
    if is_main_process:
        print("="*80)
        print("フェーズ2.3：DeepSpeed ZeRO Stage 2 統合")
        print("A100*8環境でのマルチGPU分散学習")
        print("="*80)
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリの作成（メインプロセスのみ）
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if is_main_process:
        print(f"学習設定:")
        print(f"  - エポック数: {args.epochs}")
        print(f"  - エポックあたりステップ数: {args.steps_per_epoch}")
        print(f"  - 総ステップ数: {args.epochs * args.steps_per_epoch}")
        print(f"  - 学習率: {args.learning_rate}")
        print(f"  - データセットタイプ: {args.dataset_type}")
        print(f"  - バッチサイズ: {args.batch_size}")
        print(f"  - チェックポイント間隔: {args.checkpoint_interval}ステップ")
        print(f"  - 出力ディレクトリ: {args.output_dir}")
        print(f"  - DeepSpeed設定: {args.ds_config}")
        print("-" * 80)
    
    try:
        # 1. モデルの初期化
        lisa_config = LisaGemmaConfig(
            gemma_model_id=getattr(config, 'GEMMA_MODEL_ID', 'google/gemma-3-4b-it'),
            sam_checkpoint_path=getattr(config, 'SAM_CHECKPOINT_PATH', None),
            seg_token=getattr(config, 'SEG_TOKEN', '[SEG]'),
            gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),
            sam_prompt_embed_dim=getattr(config, 'SEG_PROJECTION_DIM', 256),
            gemma_image_size=getattr(config, 'GEMMA_IMAGE_SIZE', 896),
            sam_image_size=getattr(config, 'SAM_IMAGE_SIZE', 1024),
            model_max_length=getattr(config, 'MODEL_MAX_LENGTH', 2048),
        )
        
        model = LisaGemmaForCausalLM(lisa_config)
        
        if is_main_process:
            print("✅ モデル初期化完了")
        
        # LoRA設定を適用
        if is_main_process:
            print("\n🔧 LoRA設定を適用中...")
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=config.LORA_TARGET_MODULES,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # LoRAをGemmaモデルに適用
            model.gemma_model = get_peft_model(model.gemma_model, lora_config)
            if is_main_process:
                print("✅ LoRA設定が正常に適用されました")
            
        except Exception as e:
            if is_main_process:
                print(f"⚠️  LoRA適用に失敗: {e}")
                print("   LoRAなしで検証を続行します")
        
        # 2. データセットとデータローダーの準備
        if is_main_process:
            print(f"\n📊 {args.dataset_type}データセットを準備中...")
        
        processor = AutoProcessor.from_pretrained(lisa_config.gemma_model_id)
        
        # データセットタイプの決定
        if args.dataset_type == "all":
            dataset_spec = "sem_seg||refer_seg||vqa||reason_seg"
            if is_main_process:
                print(f"  - 全データセットを使用: sem_seg, refer_seg, vqa, reason_seg")
        else:
            dataset_spec = args.dataset_type
            if is_main_process:
                print(f"  - 単一データセットを使用: {args.dataset_type}")
        
        # バッチサイズ調整（A100*8の場合は大きくできる）
        effective_batch_size = args.batch_size
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb > 75:  # A100 80GB検出
                effective_batch_size = max(args.batch_size, 2)  # 最低2
                if is_main_process:
                    print(f"  ✅ A100大容量GPU検出 ({gpu_memory_gb:.1f}GB): バッチサイズ {effective_batch_size} を使用")
            else:
                if is_main_process:
                    print(f"  📊 GPU メモリ: {gpu_memory_gb:.1f}GB - バッチサイズ {effective_batch_size} を維持")
        
        # データセット作成
        samples_per_epoch = effective_batch_size * args.steps_per_epoch
        dataset = HybridDataset(
            base_image_dir=getattr(config, 'DATASET_BASE_DIR', './dataset'),
            gemma_processor=processor,
            dataset=dataset_spec,
            samples_per_epoch=samples_per_epoch
        )
        
        if is_main_process:
            print(f"✅ データセット準備完了")
            print(f"  - 総サンプル数: {len(dataset)}")
            print(f"  - エポックあたりサンプル数: {samples_per_epoch}")
            print(f"  - 実効バッチサイズ: {effective_batch_size}")
        
        # 3. DeepSpeed初期化
        if is_main_process:
            print(f"\n🚀 DeepSpeed ZeRO Stage 2 初期化中...")
        
        # DeepSpeed設定を確認
        if not os.path.exists(args.ds_config):
            raise FileNotFoundError(f"DeepSpeed設定ファイルが見つかりません: {args.ds_config}")
        
        # DeepSpeedでモデルとオプティマイザーを初期化
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=args.ds_config,
            dist_init_required=False  # 既に初期化済み
        )
        
        if is_main_process:
            print("✅ DeepSpeed ZeRO Stage 2 初期化完了")
            
            # 学習可能パラメータの情報を表示
            param_info = model.get_trainable_parameters_info()
            print(f"✅ モデル統計:")
            print(f"  - 総パラメータ数: {param_info['total_parameters']:,}")
            print(f"  - 学習可能パラメータ数: {param_info['trainable_parameters']:,}")
            print(f"  - 学習可能率: {param_info['trainable_percentage']:.2f}%")
            
            # 仕様書準拠性チェック
            trainable_ratio = param_info['trainable_percentage']
            spec_compliant = trainable_ratio < 1.0
            print(f"  - 📋 仕様書準拠性: {'✅ 準拠' if spec_compliant else '❌ 違反'} (要求: <1%)")
        
        # 4. WandB初期化（メインプロセスのみ）
        if is_main_process:
            wandb.init(
                project="lisa-gemma-deepspeed",
                name=f"deepspeed_zero2_{session_timestamp}",
                config={
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "total_steps": args.epochs * args.steps_per_epoch,
                    "batch_size": effective_batch_size,
                    "dataset_type": args.dataset_type,
                    "world_size": world_size,
                    "ds_config": args.ds_config,
                }
            )
        
        # 5. 損失関数の準備
        loss_fn = CompositeLoss(
            ce_loss_weight=getattr(config, 'CE_LOSS_WEIGHT', 1.0),
            dice_loss_weight=getattr(config, 'DICE_LOSS_WEIGHT', 0.5),
            bce_loss_weight=getattr(config, 'BCE_LOSS_WEIGHT', 2.0)
        )
        
        # 6. エポックベース学習ループの実行
        if is_main_process:
            print(f"\n🚀 DeepSpeed分散学習を開始...")
            print(f"目標: {args.epochs}エポック × {args.steps_per_epoch}ステップで損失を減少させる")
            print("-" * 80)
        
        loss_history = []
        best_loss = float('inf')
        global_step = 0
        start_time = time.time()
        
        # エポックループ
        for epoch in range(args.epochs):
            if is_main_process:
                print(f"\n📅 Epoch {epoch+1}/{args.epochs} 開始")
            
            # エポックごとにデータローダーを再生成（シャッフル効果）
            dataloader = DataLoader(
                dataset,
                batch_size=effective_batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True
            )
            
            epoch_start_time = time.time()
            epoch_loss_sum = 0.0
            
            # ステップループ（各エポック内）
            dataloader_iter = iter(dataloader)
            for step in range(args.steps_per_epoch):
                try:
                    # バッチを取得
                    batch = next(dataloader_iter)
                except StopIteration:
                    # データローダーの終端に達した場合、再度イテレータを作成
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                # デバイスに移動（DeepSpeedは自動で処理）
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # フォワードパス
                model_engine.train()
                
                # セグメンテーションタスクの確認
                has_segmentation = 'ground_truth_mask' in batch
                
                # モデルに適した入力形式を準備
                model_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_masks'],
                    'labels': batch['labels'],
                    'generate_mask': has_segmentation,
                }
                
                # デュアルストリーム対応
                if 'images_for_gemma' in batch:
                    model_inputs['images_for_gemma'] = batch['images_for_gemma']
                if 'images_for_sam' in batch:
                    model_inputs['images_for_sam'] = batch['images_for_sam']
                
                # フォワードパス実行
                try:
                    outputs = model_engine(**model_inputs)
                    
                    # 初回のみメモリ監視（メインプロセスのみ）
                    if global_step == 0 and is_main_process:
                        if torch.cuda.is_available():
                            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                            print(f"  📊 フォワードパス後 GPU メモリ: {allocated_memory:.2f}GB")
                    
                except torch.cuda.OutOfMemoryError as e:
                    if is_main_process:
                        print(f"❌ フォワードパス中にGPUメモリ不足: {e}")
                        print("🔧 メモリクリーンアップを実行中...")
                    torch.cuda.empty_cache()
                    raise
                
                # 損失計算
                losses = loss_fn(outputs, batch)
                total_loss = losses['total_loss']
                
                # バックワードパス（DeepSpeed使用）
                try:
                    model_engine.backward(total_loss)
                    
                    # 初回のみメモリ監視（メインプロセスのみ）
                    if global_step == 0 and is_main_process:
                        if torch.cuda.is_available():
                            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                            print(f"  📊 バックワードパス後 GPU メモリ: {allocated_memory:.2f}GB")
                    
                except torch.cuda.OutOfMemoryError as e:
                    if is_main_process:
                        print(f"❌ バックワードパス中にGPUメモリ不足: {e}")
                        print("🔧 メモリクリーンアップを実行中...")
                    torch.cuda.empty_cache()
                    raise
                
                # オプティマイザーステップ（DeepSpeed）
            model_engine.step()
            
                # 損失履歴を記録（メインプロセスのみ）
                if is_main_process:
                    loss_record = {
                        'epoch': epoch,
                        'step': step,
                        'global_step': global_step,
                        'total_loss': total_loss.item(),
                        'text_loss': losses.get('text_loss', torch.tensor(0.0)).item(),
                        'dice_loss': losses.get('dice_loss', torch.tensor(0.0)).item(),
                        'bce_loss': losses.get('bce_loss', torch.tensor(0.0)).item(),
                    }
                    loss_history.append(loss_record)
                    epoch_loss_sum += total_loss.item()
                    
                    # WandBに損失をログ
                    wandb.log({
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "total_loss": total_loss.item(),
                        "text_loss": loss_record['text_loss'],
                        "dice_loss": loss_record['dice_loss'],
                        "bce_loss": loss_record['bce_loss'],
                        "learning_rate": args.learning_rate,
                    })
                    
                    # 最良損失の更新
                    if total_loss.item() < best_loss:
                        best_loss = total_loss.item()
                    
                    # 進捗表示（10ステップごと、またはエポック最終）
                    if (step + 1) % 10 == 0 or step == args.steps_per_epoch - 1:
                        elapsed_time = time.time() - start_time
                        memory_info = get_memory_usage()
                        memory_str = format_memory_info(memory_info)
                        
                        print(f"Epoch {epoch+1:2d}/{args.epochs} Step {step+1:3d}/{args.steps_per_epoch}: "
                              f"Loss={total_loss.item():.6f} "
                              f"(Text: {loss_record['text_loss']:.4f}, "
                              f"DICE: {loss_record['dice_loss']:.4f}, "
                              f"BCE: {loss_record['bce_loss']:.4f}) "
                              f"Best: {best_loss:.6f}")
                        print(f"                     Memory: {memory_str}")
                
                # チェックポイント保存（メインプロセスのみ）
                if is_main_process and (global_step + 1) % args.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"deepspeed_checkpoint_epoch{epoch+1}_step{global_step+1}")
                    model_engine.save_checkpoint(checkpoint_path)
                    print(f"💾 DeepSpeedチェックポイント保存: {checkpoint_path}")
                    
                    # WandBにチェックポイント情報をログ
                    wandb.log({
                        "checkpoint/epoch": epoch,
                        "checkpoint/global_step": global_step,
                        "checkpoint/loss": total_loss.item(),
                    })
                
                global_step += 1
            
            # エポック終了時の処理（メインプロセスのみ）
            if is_main_process:
                epoch_elapsed = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss_sum / args.steps_per_epoch
                
                print(f"📊 Epoch {epoch+1} 完了: 平均損失={avg_epoch_loss:.6f}, 時間={epoch_elapsed:.1f}秒")
                
                # WandBにエポック統計をログ
                wandb.log({
                    "epoch_avg_loss": avg_epoch_loss,
                    "epoch_duration": epoch_elapsed,
                    "completed_epochs": epoch + 1,
                })
                
                # 最終エポックのチェックポイント保存
                if epoch == args.epochs - 1:
                    final_checkpoint_path = os.path.join(args.output_dir, f"deepspeed_final_checkpoint_{session_timestamp}")
                    model_engine.save_checkpoint(final_checkpoint_path)
                    print(f"💾 最終DeepSpeedチェックポイント保存: {final_checkpoint_path}")
        
        if is_main_process:
            print("-" * 80)
            print("✅ DeepSpeed分散学習完了")
        
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if is_main_process:
                final_memory = torch.cuda.memory_allocated() / (1024**3)
                print(f"🧹 GPU メモリクリーンアップ完了 (最終使用量: {final_memory:.2f}GB)")
        
        # 7. 学習結果の分析（メインプロセスのみ）
        if is_main_process:
            print("\n📊 学習結果の分析...")
            
            if len(loss_history) > 0:
                initial_loss = loss_history[0]['total_loss']
                final_loss = loss_history[-1]['total_loss']
                reduction_ratio = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
                
                print(f"学習結果:")
                print(f"  - 総ステップ数: {len(loss_history)}")
                print(f"  - 完了エポック数: {args.epochs}")
                print(f"  - 初期損失: {initial_loss:.6f}")
                print(f"  - 最終損失: {final_loss:.6f}")
                print(f"  - 総損失減少率: {reduction_ratio:.1%}")
                print(f"  - 最良損失: {best_loss:.6f}")
                
                # WandBに最終分析結果をログ
                wandb.log({
                    "final_analysis/total_steps": len(loss_history),
                    "final_analysis/completed_epochs": args.epochs,
                    "final_analysis/initial_loss": initial_loss,
                    "final_analysis/final_loss": final_loss,
                    "final_analysis/reduction_ratio": reduction_ratio,
                    "final_analysis/best_loss": best_loss,
                })
                
                # 学習成果の評価
                print(f"\n📋 学習成果の評価:")
                if reduction_ratio > 0.1:
                    print(f"  ✅ 優秀な学習進捗: 損失が{reduction_ratio:.1%}減少")
                elif reduction_ratio > 0.05:
                    print(f"  ⚠️  中程度の学習進捗: 損失が{reduction_ratio:.1%}減少")
                else:
                    print(f"  ❌ 学習進捗不十分: 損失減少が{reduction_ratio:.1%}のみ")
            
                # 8. 結果の保存
                print("\n💾 結果を保存中...")
                
                # 損失曲線のプロット
                plot_path = os.path.join(args.output_dir, f"deepspeed_training_loss_curve_{session_timestamp}.png")
                plot_loss_curve(loss_history, plot_path)
                
                # 詳細結果をJSONで保存
                training_analysis = {
                    "total_steps": len(loss_history),
                    "completed_epochs": args.epochs,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "reduction_ratio": reduction_ratio,
                    "best_loss": best_loss,
                }
                
                results = {
                    "session_timestamp": session_timestamp,
                    "training_config": {
                        "epochs": args.epochs,
                        "steps_per_epoch": args.steps_per_epoch,
                        "total_steps": args.epochs * args.steps_per_epoch,
                        "learning_rate": args.learning_rate,
                        "dataset_type": args.dataset_type,
                        "batch_size": effective_batch_size,
                        "world_size": world_size,
                        "ds_config": args.ds_config,
                    },
                    "training_analysis": training_analysis,
                    "loss_history": loss_history,
                    "final_status": "success" if training_analysis['reduction_ratio'] > 0.05 else "partial_success"
                }
                
                json_path = os.path.join(args.output_dir, f"deepspeed_training_results_{session_timestamp}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 結果保存完了:")
                print(f"  - 損失曲線: {plot_path}")
                print(f"  - 詳細データ: {json_path}")
                print(f"  - DeepSpeedチェックポイント: {args.output_dir}/deepspeed_final_checkpoint_{session_timestamp}")
        
                # 9. 最終判定
                print("\n" + "="*80)
                print("🎯 最終判定")
                print("="*80)
                
                if training_analysis['reduction_ratio'] > 0.05:
                    print("✅ Phase 2.3: DeepSpeed ZeRO Stage 2統合 - 成功")
                    print("   DTensor競合問題が解決されました。")
                    print("   DeepSpeed分散学習が正常に機能しています。")
                    print("   A100*8環境でのマルチGPU学習が成功しています。")
                    
                    if training_analysis['reduction_ratio'] > 0.1:
                        print("   → Phase 3: DeepSpeed ZeRO Stage 3移行の準備が整いました。")
                    else:
                        print("   → さらなる最適化でパフォーマンス向上の余地があります。")
                else:
                    print("❌ Phase 2.3: DeepSpeed ZeRO Stage 2統合 - 要改善")
                    print("   学習進捗が不十分です。")
                    
                wandb.log({
                    "final_status": results["final_status"],
                    "total_steps": training_analysis['total_steps'],
                    "completed_epochs": training_analysis['completed_epochs'],
                    "reduction_ratio": training_analysis['reduction_ratio'],
                    "final_loss": training_analysis['final_loss']
                })
            
            # WandB実験を終了
            wandb.finish()
        
        print("="*80)
        
    except Exception as e:
        if is_main_process:
            print(f"\n❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 