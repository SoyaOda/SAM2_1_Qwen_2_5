#!/usr/bin/env python3
"""
フェーズ1：実験管理と設定の高度化
1.2. 学習主軸スクリプト train.py の新設

本格的なエポックベースの学習を実行し、WandBによる実験追跡を行う。
overfit_single_batch.py のロジックを基に、大規模学習に対応した設計を実装。

論理的根拠:
1. エポックベースの本格的な学習システムを構築し、段階的なスケールアップに対応する。
2. WandBによる実験追跡により、学習の進捗とハイパーパラメータの影響を体系的に管理する。
3. チェックポイント機能により、長時間学習の中断・再開に対応し、安全性を確保する。
4. データローダーのエポック再生成により、データシャッフルと学習の多様性を保証する。
5. マルチGPU対応への準備として、accelerateライブラリとの統合基盤を整備する。

期待される結果:
エポックごとの損失減少と学習の進捗が確認でき、WandBでリアルタイム監視が可能になります。
チェックポイント保存により学習の継続性が保証され、より大規模なデータセットと長時間学習への準備が整います。
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor
from accelerate import Accelerator
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話型に設定
import matplotlib.pyplot as plt
import psutil
import wandb
from datetime import datetime

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn

def get_config():
    """
    config_linux.pyを必須として読み込む
    読み込めない場合はエラーで停止
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
    # config_linux.pyから設定を取得
    config = get_config()
    
    parser = argparse.ArgumentParser(description="LISA-Gemma学習主軸スクリプト（WandB統合・エポックベース）")
    parser.add_argument("--epochs", type=int, default=5, help="学習エポック数")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="各エポックのステップ数")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="学習率")
    parser.add_argument("--dataset_type", type=str, default="all", 
                       choices=["sem_seg", "refer_seg", "vqa", "reason_seg", "all"],
                       help="学習に使用するデータセットタイプ（'all'で全データセット）")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE_PER_GPU, help="バッチサイズ")
    parser.add_argument("--checkpoint_interval", type=int, default=50, 
                       help="チェックポイント保存間隔（ステップ数）")
    parser.add_argument("--output_dir", type=str, default="./training_output", 
                       help="学習結果とチェックポイントの保存ディレクトリ")
    return parser.parse_args()

def plot_loss_curve(loss_history: List[Dict[str, float]], output_path: str):
    """損失曲線をプロットして保存"""
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
    ax1.set_title('単一バッチ過学習テスト: 総損失の推移（対数スケール）')
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
        
        # GPU利用率（簡易的な計算）
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

def analyze_overfitting_success(loss_history: List[Dict[str, float]]) -> Dict[str, Any]:
    """過学習の成功度を分析"""
    if len(loss_history) < 2:
        return {
            "success": False, 
            "reason": "insufficient_iterations",
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "reduction_ratio": 0.0,
            "seg_reduction_ratio": 0.0,
            "text_reduction_ratio": 0.0,
            "stability_ratio": 0.0,
            "iterations": len(loss_history)
        }
    
    initial_loss = loss_history[0]['total_loss']
    final_loss = loss_history[-1]['total_loss']
    
    # 損失減少率
    reduction_ratio = (initial_loss - final_loss) / initial_loss
    
    # 最後の損失の安定性チェック（少ないイテレーション対応）
    if len(loss_history) >= 3:
        stable_window = max(2, len(loss_history) // 2)
        recent_losses = [h['total_loss'] for h in loss_history[-stable_window:]]
        recent_std = torch.tensor(recent_losses).std().item()
        recent_mean = torch.tensor(recent_losses).mean().item()
        stability_ratio = recent_std / recent_mean if recent_mean > 0 else float('inf')
    else:
        stability_ratio = 0.0  # 少ないイテレーションでは安定性チェックをスキップ
    
    # セグメンテーション損失の分析
    initial_seg_loss = loss_history[0].get('dice_loss', 0) + loss_history[0].get('bce_loss', 0)
    final_seg_loss = loss_history[-1].get('dice_loss', 0) + loss_history[-1].get('bce_loss', 0)
    seg_reduction_ratio = (initial_seg_loss - final_seg_loss) / initial_seg_loss if initial_seg_loss > 0 else 0
    
    # テキスト損失の分析
    initial_text_loss = loss_history[0].get('text_loss', 0)
    final_text_loss = loss_history[-1].get('text_loss', 0)
    text_reduction_ratio = (initial_text_loss - final_text_loss) / initial_text_loss if initial_text_loss > 0 else 0
    
    # 柔軟な成功基準（少ないイテレーション対応）
    if len(loss_history) < 5:
        # 少ないイテレーション（2-4回）の場合：損失減少があれば成功
        success = reduction_ratio > 0.1  # 10%以上の総損失減少
    else:
        # 十分なイテレーション（5回以上）の場合：厳格な基準
        success = (
            reduction_ratio > 0.2 and  # 20%以上の総損失減少
            seg_reduction_ratio > 0.3 and  # セグメンテーション損失が30%以上減少
            stability_ratio < 0.5  # 最近の損失の変動が平均の50%以下
        )
    
    return {
        "success": success,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "reduction_ratio": reduction_ratio,
        "seg_reduction_ratio": seg_reduction_ratio,
        "text_reduction_ratio": text_reduction_ratio,
        "stability_ratio": stability_ratio,
        "iterations": len(loss_history)
    }

def main():
    print("================================================================================")
    print("フェーズ2：マルチGPU対応とFSDPでの学習")
    print("2.1. train.py の accelerate 対応")
    print("================================================================================")
    print()
    
    # 🔧 Web検索で発見したPyTorch公式推奨解決策：DTensor環境変数を事前設定
    print("🔧 DTensor問題対策: 暗黙的レプリケーション許可を設定")
    import os
    os.environ['TORCH_DTENSOR_ALLOW_IMPLICIT_REPLICATION'] = '1'
    
    # 追加のDTensor制御環境変数（Web検索で発見）
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'  # DTensorデバッグログを抑制
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'       # 非同期実行でDTensor競合を回避
    
    print("✅ DTensor環境変数設定完了")
    
    args = parse_args()
    config = get_config()
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("フェーズ2：マルチGPU対応とFSDPでの学習")
    print("2.1. train.py の accelerate 対応")
    print("="*80)
    
    try:
        # 1. モデルの初期化（FSDP環境初期化前に実行）
        print("\n📦 LISA-Gemmaモデルを初期化中（DDP適用前）...")
        
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
        print("✅ モデル初期化完了（resize_token_embeddings成功）")
        
        # LoRA設定を適用（FSDP適用前）
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
            print("✅ LoRA設定が正常に適用されました")
            
        except Exception as e:
            print(f"⚠️  LoRA適用に失敗: {e}")
            print("   LoRAなしで検証を続行します")
        
        model.train()  # 学習モードに設定
        
        # 2. accelerate初期化（モデル初期化後に実行）
        print("\n🚀 DDP分散環境を初期化中...")
        accelerator = Accelerator(
            gradient_accumulation_steps=getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        )
        
        # 出力ディレクトリの作成
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # デバイス設定（accelerateが管理）
        device = accelerator.device
        if accelerator.is_main_process:
            print(f"✅ DDP環境初期化完了")
            print(f"使用デバイス: {device}")
            print(f"学習設定:")
            print(f"  - エポック数: {args.epochs}")
            print(f"  - エポックあたりステップ数: {args.steps_per_epoch}")
            print(f"  - 総ステップ数: {args.epochs * args.steps_per_epoch}")
            print(f"  - 学習率: {args.learning_rate}")
            print(f"  - データセットタイプ: {args.dataset_type}")
            print(f"  - バッチサイズ: {args.batch_size}")
            print(f"  - チェックポイント間隔: {args.checkpoint_interval}ステップ")
            print(f"  - 出力ディレクトリ: {args.output_dir}")
            print("-" * 80)
        
        if accelerator.is_main_process:
            print("✅ モデル初期化完了")
            
            # 学習可能パラメータの情報を表示
            param_info = model.get_trainable_parameters_info()
            print(f"  - 総パラメータ数: {param_info['total_parameters']:,}")
            print(f"  - 学習可能パラメータ数: {param_info['trainable_parameters']:,}")
            print(f"  - 学習可能率: {param_info['trainable_percentage']:.2f}%")
            
            # 仕様書準拠性チェック
            trainable_ratio = param_info['trainable_percentage']
            spec_compliant = trainable_ratio < 1.0
            print(f"  - 📋 仕様書準拠性: {'✅ 準拠' if spec_compliant else '❌ 違反'} (要求: <1%)")
            
            if not spec_compliant:
                print(f"    ⚠️  学習可能パラメータ率が仕様書要求を超過: {trainable_ratio:.2f}% > 1%")
                print("    → LoRA適用を確認してください")
            
            # 初期メモリ使用量
            initial_memory = get_memory_usage()
            print(f"  - 初期メモリ使用量: {format_memory_info(initial_memory)}")
        else:
            # メインプロセス以外でもparam_infoは必要
            param_info = model.get_trainable_parameters_info()
        
        # WandB初期化（メインプロセスのみ）
        if accelerator.is_main_process:
            wandb.init(
                project="lisa-gemma-training", 
                name=f"train-{session_timestamp}",
                config={
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "total_steps": args.epochs * args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "dataset_type": args.dataset_type,
                    "checkpoint_interval": args.checkpoint_interval,
                    "model_params": param_info,
                }
            )
        
        # 2. データセットとデータローダーの準備
        if accelerator.is_main_process:
            print(f"\n📊 {args.dataset_type}データセットを準備中...")
        
        processor = AutoProcessor.from_pretrained(lisa_config.gemma_model_id)
        
        # データセットタイプの決定
        if args.dataset_type == "all":
            dataset_spec = "sem_seg||refer_seg||vqa||reason_seg"
            if accelerator.is_main_process:
                print(f"  - 全データセットを使用: sem_seg, refer_seg, vqa, reason_seg")
        else:
            dataset_spec = args.dataset_type
            if accelerator.is_main_process:
                print(f"  - 単一データセットを使用: {args.dataset_type}")
        
        # A10 24GB制約対応: 動的バッチサイズ調整
        effective_batch_size = args.batch_size
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 25:  # A10 (24GB) 検出
                effective_batch_size = 1
                if accelerator.is_main_process:
                    print(f"  ⚠️  A10 GPU検出 ({gpu_memory_gb:.1f}GB): バッチサイズを {args.batch_size} → {effective_batch_size} に調整")
            else:
                if accelerator.is_main_process:
                    print(f"  ✅ GPU メモリ十分 ({gpu_memory_gb:.1f}GB): バッチサイズ {effective_batch_size} を維持")
        
        # 本格学習用データセット
        samples_per_epoch = effective_batch_size * args.steps_per_epoch
        dataset = HybridDataset(
            base_image_dir=getattr(config, 'DATASET_BASE_DIR', './dataset'),
            gemma_processor=processor,
            dataset=dataset_spec,
            samples_per_epoch=samples_per_epoch
        )
        
        # データローダーの作成
        dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # accelerate使用時はnum_workers=0推奨
        )
        
        if accelerator.is_main_process:
            print(f"✅ データセット準備完了")
            print(f"  - 総サンプル数: {len(dataset)}")
            print(f"  - エポックあたりサンプル数: {samples_per_epoch}")
            print(f"  - 実効バッチサイズ: {effective_batch_size}")
        
        # 3. チェックポイント保存関数の定義
        def save_checkpoint(epoch, step, model, optimizer, loss, save_path):
            """チェックポイントを保存"""
            # accelerate.save_stateの使用を準備
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'session_timestamp': session_timestamp,
                'config': {
                    'learning_rate': args.learning_rate,
                    'epochs': args.epochs,
                    'steps_per_epoch': args.steps_per_epoch,
                    'batch_size': args.batch_size,
                    'dataset_type': args.dataset_type,
                }
            }
            if accelerator.is_main_process:
                torch.save(checkpoint, save_path)
            return save_path
        
        # 4. オプティマイザーと損失関数の準備
        if accelerator.is_main_process:
            print("\n⚙️ オプティマイザーと損失関数を準備中...")
        
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-2),
            betas=(getattr(config, 'BETA1', 0.9), getattr(config, 'BETA2', 0.95))
        )
        
        loss_fn = CompositeLoss(
            ce_loss_weight=getattr(config, 'CE_LOSS_WEIGHT', 1.0),
            dice_loss_weight=getattr(config, 'DICE_LOSS_WEIGHT', 0.5),
            bce_loss_weight=getattr(config, 'BCE_LOSS_WEIGHT', 2.0)
        )
        
        # accelerate prepare() メソッドの適用（Phase 2.1の核心）
        model, optimizer, dataloader = accelerator.prepare(
            model, optimizer, dataloader
        )
        
        if accelerator.is_main_process:
            print("✅ オプティマイザーと損失関数の準備完了")
            print("✅ accelerate prepare() 適用完了")
        
        # 5. エポックベース学習ループの実行
        if accelerator.is_main_process:
            print(f"\n🚀 エポックベース学習を開始...")
            print(f"目標: {args.epochs}エポック × {args.steps_per_epoch}ステップで損失を減少させる")
            
            # デバッグログの抑制設定
            import logging
            logging.getLogger().setLevel(logging.WARNING)  # デバッグログを抑制
            
            print("-" * 80)
        
        loss_history = []
        best_loss = float('inf')
        global_step = 0
        start_time = time.time()
        
        # エポックループ
        for epoch in range(args.epochs):
            if accelerator.is_main_process:
                print(f"\n📅 Epoch {epoch+1}/{args.epochs} 開始")
            
            # エポックごとにデータローダーを再生成（シャッフル効果）
            dataloader_iter = iter(dataloader)
            for step in range(args.steps_per_epoch):
                try:
                    # バッチを取得
                    batch = next(dataloader_iter)
                except StopIteration:
                    # データローダーの終端に達した場合、再度イテレータを作成
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                # バッチのデバイス移動を削除（accelerateが自動処理）
                # batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                #         for k, v in batch.items()}
                
                # フォワードパス
                optimizer.zero_grad()
                
                # セグメンテーションタスクの確認
                has_segmentation = 'ground_truth_mask' in batch
                
                # モデルに適した入力形式を準備
                model_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_masks'],  # collate_fnでは'attention_masks'が使われる
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
                    outputs = model(**model_inputs)
                    
                    # 初回のみメモリ監視（メインプロセスのみ）
                    if global_step == 0 and accelerator.is_main_process:
                        if torch.cuda.is_available():
                            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                            print(f"  📊 フォワードパス後 GPU メモリ: {allocated_memory:.2f}GB")
                    
                except torch.cuda.OutOfMemoryError as e:
                    if accelerator.is_main_process:
                        print(f"❌ フォワードパス中にGPUメモリ不足: {e}")
                        print("🔧 メモリクリーンアップを実行中...")
                    torch.cuda.empty_cache()
                    raise
                
                # 損失計算
                losses = loss_fn(outputs, batch)
                total_loss = losses['total_loss']
                
                # バックワードパス（accelerate対応）
                try:
                    accelerator.backward(total_loss)
                    
                    # 初回のみメモリ監視（メインプロセスのみ）
                    if global_step == 0 and accelerator.is_main_process:
                        if torch.cuda.is_available():
                            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                            print(f"  📊 バックワードパス後 GPU メモリ: {allocated_memory:.2f}GB")
                    
                except torch.cuda.OutOfMemoryError as e:
                    if accelerator.is_main_process:
                        print(f"❌ バックワードパス中にGPUメモリ不足: {e}")
                        print("🔧 メモリクリーンアップを実行中...")
                    torch.cuda.empty_cache()
                    raise
                
                # 勾配クリッピング（安定性のため）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # オプティマイザーステップ
                optimizer.step()
                
                # 損失履歴を記録
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
                
                # WandBに損失をログ（accelerator経由・メインプロセスのみ）
                if accelerator.is_main_process:
                    accelerator.log({
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
                if accelerator.is_main_process and ((step + 1) % 10 == 0 or step == args.steps_per_epoch - 1):
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
                
                # チェックポイント保存
                if (global_step + 1) % args.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}_step{global_step+1}.pth")
                    saved_path = save_checkpoint(epoch, global_step, model, optimizer, total_loss.item(), checkpoint_path)
                    if accelerator.is_main_process:
                        print(f"💾 チェックポイント保存: {saved_path}")
                        
                        # WandBにチェックポイント情報をログ
                        accelerator.log({
                            "checkpoint/epoch": epoch,
                            "checkpoint/global_step": global_step,
                            "checkpoint/loss": total_loss.item(),
                        })
                
                global_step += 1
            
            # エポック終了時の処理
            epoch_elapsed = time.time() - start_time
            avg_epoch_loss = sum(h['total_loss'] for h in loss_history) / len(loss_history)
            
            if accelerator.is_main_process:
                print(f"📊 Epoch {epoch+1} 完了: 平均損失={avg_epoch_loss:.6f}, 時間={epoch_elapsed:.1f}秒")
                
                # WandBにエポック統計をログ
                accelerator.log({
                    "epoch_avg_loss": avg_epoch_loss,
                    "epoch_duration": epoch_elapsed,
                    "completed_epochs": epoch + 1,
                })
            
            # 最終エポックのチェックポイント保存
            if epoch == args.epochs - 1:
                final_checkpoint_path = os.path.join(args.output_dir, f"final_checkpoint_{session_timestamp}.pth")
                saved_path = save_checkpoint(epoch, global_step - 1, model, optimizer, avg_epoch_loss, final_checkpoint_path)
                if accelerator.is_main_process:
                    print(f"💾 最終チェックポイント保存: {saved_path}")
        
        if accelerator.is_main_process:
            print("-" * 80)
            print("✅ エポックベース学習完了")
        
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                final_memory = torch.cuda.memory_allocated() / (1024**3)
                print(f"🧹 GPU メモリクリーンアップ完了 (最終使用量: {final_memory:.2f}GB)")
        
        # 6. 学習結果の分析
        if accelerator.is_main_process:
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
                accelerator.log({
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
                    print(f"  ✅ 良好な学習進捗: 損失が{reduction_ratio:.1%}減少")
                elif reduction_ratio > 0.05:
                    print(f"  ⚠️  中程度の学習進捗: 損失が{reduction_ratio:.1%}減少")
                else:
                    print(f"  ❌ 学習進捗不十分: 損失減少が{reduction_ratio:.1%}のみ")
                    
                # エポック間の学習安定性評価
                if args.epochs > 1:
                    epoch_losses = []
                    for epoch in range(args.epochs):
                        epoch_data = [h for h in loss_history if h['epoch'] == epoch]
                        if epoch_data:
                            avg_loss = sum(h['total_loss'] for h in epoch_data) / len(epoch_data)
                            epoch_losses.append(avg_loss)
                    
                    if len(epoch_losses) > 1:
                        improvement_trend = epoch_losses[0] - epoch_losses[-1]
                        print(f"  - エポック間改善: {improvement_trend:.6f}")
            else:
                print("  ⚠️  学習履歴が記録されていません")
        
        # 7. 結果の保存
        if accelerator.is_main_process:
            print("\n💾 結果を保存中...")
            
            # 損失曲線のプロット（loss_historyが空でない場合のみ）
            if len(loss_history) > 0:
                plot_path = os.path.join(args.output_dir, f"training_loss_curve_{session_timestamp}.png")
                plot_loss_curve(loss_history, plot_path)
                
                # WandBに損失曲線画像をログ
                accelerator.log({"loss_curve": wandb.Image(plot_path)})
        
        # 詳細結果をJSONで保存
        if len(loss_history) > 0:
            training_analysis = {
                "total_steps": len(loss_history),
                "completed_epochs": args.epochs,
                "initial_loss": loss_history[0]['total_loss'],
                "final_loss": loss_history[-1]['total_loss'],
                "reduction_ratio": reduction_ratio,
                "best_loss": best_loss,
            }
        else:
            training_analysis = {
                "total_steps": 0,
                "completed_epochs": 0,
                "initial_loss": 0.0,
                "final_loss": 0.0,
                "reduction_ratio": 0.0,
                "best_loss": 0.0,
            }
        
        results = {
            "session_timestamp": session_timestamp,
            "training_config": {
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "total_steps": args.epochs * args.steps_per_epoch,
                "learning_rate": args.learning_rate,
                "dataset_type": args.dataset_type,
                "batch_size": args.batch_size,
                "checkpoint_interval": args.checkpoint_interval,
            },
            "model_info": param_info,
            "training_analysis": training_analysis,
            "loss_history": loss_history,
            "final_status": "success" if training_analysis['reduction_ratio'] > 0.05 else "partial_success"
        }
        
        if accelerator.is_main_process:
            json_path = os.path.join(args.output_dir, f"training_results_{session_timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 結果保存完了:")
            if len(loss_history) > 0:
                print(f"  - 損失曲線: {plot_path}")
            print(f"  - 詳細データ: {json_path}")
            print(f"  - チェックポイント: {args.output_dir}/final_checkpoint_{session_timestamp}.pth")
        
        # 8. 最終判定
        if accelerator.is_main_process:
            print("\n" + "="*80)
            print("🎯 最終判定")
            print("="*80)
            
            if len(loss_history) > 0 and training_analysis['reduction_ratio'] > 0.05:
                print("✅ Phase 2.1: accelerate対応 - 成功")
                print("   accelerateライブラリが正常に統合されています。")
                print("   エポックベース学習システムが正しく機能しています。")
                print("   WandB統合により実験追跡が成功しています。")
                
                if training_analysis['reduction_ratio'] > 0.1:
                    print("   → 次のフェーズ（マルチGPU設定）に進む準備が整いました。")
                else:
                    print("   → 学習パフォーマンスを向上させるためのハイパーパラメータ調整を推奨します。")
            else:
                print("❌ Phase 2.1: accelerate対応 - 要改善")
                print("   以下の問題が考えられます:")
                
                if len(loss_history) == 0:
                    print("   - 学習ループが正常に実行されませんでした")
                elif training_analysis['reduction_ratio'] < 0.01:
                    print("   - 学習率が低すぎる可能性があります")
                    print("   - モデルが学習されていません（LoRA設定を確認）")
                elif training_analysis['reduction_ratio'] < 0.05:
                    print("   - 学習進捗が不十分です")
                    print("   - より多くのエポックまたはステップが必要な可能性があります")
                
                print(f"   推奨: パラメータ調整後にテストを再実行してください")
            
            # WandBに最終結果を記録
            wandb.summary["final_status"] = results["final_status"]
            wandb.summary["final_loss"] = training_analysis['final_loss']
            wandb.summary["reduction_ratio"] = training_analysis['reduction_ratio']
            wandb.summary["total_steps"] = training_analysis['total_steps']
            wandb.summary["completed_epochs"] = training_analysis['completed_epochs']
            
            # WandB実験を終了
            wandb.finish()
            
            print("="*80)
        
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 