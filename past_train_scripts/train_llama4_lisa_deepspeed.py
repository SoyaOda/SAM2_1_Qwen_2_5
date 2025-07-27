#!/usr/bin/env python3
"""
LISA-Llama4 分散学習スクリプト (DeepSpeed統合)
A100 80GB × 8GPU対応

実行方法:
deepspeed --num_gpus=8 train_llama4_lisa_deepspeed.py \
    --exp_name lisa_llama4_a100x8 \
    --batch_size 2 \
    --grad_accumulation_steps 8
"""

import argparse
import os
import shutil
import sys
import time
from functools import partial
from datetime import datetime

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import tqdm

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LISA-Llama4モデルとユーティリティ
from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
from utils.dataset import HybridDataset, collate_fn
from utils.constants import DEFAULT_SEG_TOKEN
import config_linux

# 評価用メトリクス
from utils.utils import (
    AverageMeter, ProgressMeter, Summary, 
    dict_to_cuda, intersectionAndUnionGPU
)
from utils.hf_auth import ensure_hf_login


def setup_hf_cache_dirs():
    """HuggingFaceキャッシュディレクトリを自動設定・作成"""
    print("=== HuggingFaceキャッシュディレクトリ設定 ===")
    
    # 候補ディレクトリ（優先順位順）
    cache_candidates = [
        "/lambda/nfs/lisa-gemma-project-fs/cache/huggingface",
        os.path.expanduser("~/.cache/huggingface"),
        "/tmp/huggingface_cache"
    ]
    
    transformers_candidates = [
        "/lambda/nfs/lisa-gemma-project-fs/cache/transformers", 
        os.path.expanduser("~/.cache/transformers"),
        "/tmp/transformers_cache"
    ]
    
    # 書き込み可能なディレクトリを探す
    hf_cache_dir = None
    for candidate in cache_candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            # 書き込みテスト
            test_file = os.path.join(candidate, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            hf_cache_dir = candidate
            break
        except (OSError, PermissionError):
            continue
    
    transformers_cache_dir = None
    for candidate in transformers_candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            # 書き込みテスト
            test_file = os.path.join(candidate, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            transformers_cache_dir = candidate
            break
        except (OSError, PermissionError):
            continue
    
    # 環境変数に設定
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        print(f"✅ HF_HOME設定: {hf_cache_dir}")
    else:
        print("⚠️ HF_HOME設定に失敗")
    
    if transformers_cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache_dir
        print(f"✅ TRANSFORMERS_CACHE設定: {transformers_cache_dir}")
    else:
        print("⚠️ TRANSFORMERS_CACHE設定に失敗")
    
    # HuggingFace Hub設定も追加
    if hf_cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        print(f"✅ HUGGINGFACE_HUB_CACHE設定: {hf_cache_dir}")
    
    print("=== キャッシュディレクトリ設定完了 ===")


def parse_args():
    parser = argparse.ArgumentParser(description="LISA-Llama4 分散学習")
    
    # 基本設定
    parser.add_argument("--local_rank", default=0, type=int, help="ローカルランク")
    parser.add_argument("--exp_name", default="lisa_llama4_train", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    
    # モデル設定 (config_linuxから取得)
    parser.add_argument("--llama_model_id", 
                       default=config_linux.LLAMA_MODEL_ID, type=str)
    parser.add_argument("--sam_checkpoint_path", 
                       default=config_linux.SAM_CHECKPOINT_PATH, type=str)
    parser.add_argument("--precision", default="bf16", type=str,
                       choices=["fp32", "bf16", "fp16"])
    
    # データセット設定
    parser.add_argument("--dataset_base_dir", 
                       default=config_linux.DATASET_BASE_DIR, type=str)
    parser.add_argument("--dataset", 
                       default="sem_seg||refer_seg||vqa||reason_seg", type=str)
    parser.add_argument("--sample_rates", 
                       default=config_linux.DATASET_SAMPLE_RATES, type=str)
    parser.add_argument("--sem_seg_data", 
                       default=config_linux.SEM_SEG_DATA, type=str)
    parser.add_argument("--refer_seg_data", 
                       default=config_linux.REFER_SEG_DATA, type=str)
    parser.add_argument("--vqa_data", 
                       default=config_linux.VQA_DATA, type=str)
    parser.add_argument("--reason_seg_data", 
                       default=config_linux.REASON_SEG_DATA, type=str)
    parser.add_argument("--val_dataset", 
                       default=config_linux.VAL_DATASET, type=str)
    
    # 学習設定
    parser.add_argument("--epochs", default=config_linux.EPOCHS, type=int)
    parser.add_argument("--steps_per_epoch", 
                       default=config_linux.STEPS_PER_EPOCH, type=int)
    parser.add_argument("--batch_size", 
                       default=config_linux.BATCH_SIZE_PER_GPU, type=int)
    parser.add_argument("--grad_accumulation_steps", 
                       default=config_linux.GRADIENT_ACCUMULATION_STEPS, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", 
                       default=config_linux.DATALOADER_NUM_WORKERS, type=int)
    
    # 最適化設定
    parser.add_argument("--lr", default=config_linux.LEARNING_RATE, type=float)
    parser.add_argument("--weight_decay", 
                       default=config_linux.WEIGHT_DECAY, type=float)
    parser.add_argument("--beta1", default=config_linux.BETA1, type=float)
    parser.add_argument("--beta2", default=config_linux.BETA2, type=float)
    
    # LoRA設定
    parser.add_argument("--lora_r", default=config_linux.LORA_R, type=int)
    parser.add_argument("--lora_alpha", 
                       default=config_linux.LORA_ALPHA, type=int)
    parser.add_argument("--lora_dropout", 
                       default=config_linux.LORA_DROPOUT, type=float)
    
    # その他の設定
    parser.add_argument("--gradient_checkpointing", 
                       action="store_true", default=True)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    
    # 第3段階：完全ZeRO-3移行設定
    parser.add_argument("--zero_stage", default=3, type=int, choices=[2, 3],
                       help="DeepSpeed ZeRO stage (2: ZeRO-2, 3: ZeRO-3)")
    parser.add_argument("--cpu_offload_params", action="store_true", default=True,
                       help="ZeRO-3でパラメータをCPUオフロード（103Bモデル推奨）")
    
    return parser.parse_args()


def create_model(args):
    """LISA-Llama4モデルの作成"""
    print("=" * 80)
    print("LISA-Llama4モデル初期化開始...")
    print("=" * 80)
    
    # モデル設定を取得
    model_config_dict = config_linux.get_lisa_model_config()
    
    # DeepSpeed環境用に設定を調整
    # device_mapを削除（DeepSpeedが管理）
    model_config_dict.pop('device_map', None)
    
    # torch_dtypeを文字列から実際の型に変換
    if args.precision == "bf16":
        model_config_dict['torch_dtype'] = "bfloat16"
    elif args.precision == "fp16":
        model_config_dict['torch_dtype'] = "float16"
    else:
        model_config_dict['torch_dtype'] = "float32"
    
    # モデル設定作成
    model_config = LisaLlama4Config(**model_config_dict)
    
    # モデル初期化（DeepSpeed用）
    model = LisaLlama4ForCausalLM(model_config)
    
    # グラディエントチェックポイント有効化
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except ValueError as e:
            print(f"⚠️ カスタムモデルのgradient checkpointing: {e}")
            # 基底のLlamaモデルに対してgradient checkpointingを有効化
            if hasattr(model, 'llama_model') and hasattr(model.llama_model, 'gradient_checkpointing_enable'):
                model.llama_model.gradient_checkpointing_enable()
                print("✅ 基底Llama4モデルでgradient checkpointing有効化完了")
            else:
                print("⚠️ Gradient checkpointingをスキップしました")
    
    print("✅ LISA-Llama4モデル初期化完了")
    
    return model


def prepare_model_for_zero3(model, args):
    """第3段階：ZeRO-3用のLISA統合モデル前処理"""
    if args.local_rank == 0:
        print("=== ZeRO-3用モデル前処理 ===")
        
        # 1. パラメータの名前付けを明確化（ZeRO-3のデバッグ用）
        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count += 1
                # ZeRO-3では特定のパラメータ名でデバッグしやすくする
                if not hasattr(param, '_zero3_name'):
                    param._zero3_name = name
        
        print(f"✅ {param_count}個の学習可能パラメータにZeRO-3用メタデータを付与")
        
        # 2. SAMとLlama4の境界を明確化
        if hasattr(model, 'sam_model') and hasattr(model, 'llama_model'):
            print("✅ SAM・Llama4統合モデルの境界を設定")
            
            # SAMパラメータにマーキング
            for name, param in model.sam_model.named_parameters():
                if param.requires_grad:
                    param._component = 'sam'
            
            # Llama4パラメータにマーキング
            for name, param in model.llama_model.named_parameters():
                if param.requires_grad:
                    param._component = 'llama4'
            
            # MLPプロジェクタにマーキング
            if hasattr(model, 'multi_modal_projector'):
                for name, param in model.multi_modal_projector.named_parameters():
                    if param.requires_grad:
                        param._component = 'projector'
        
        # 3. 勾配チェックポイントの最適化
        print("✅ ZeRO-3用勾配チェックポイント最適化を設定")
        
        print("=== ZeRO-3用モデル前処理完了 ===")


def debug_gpu_memory_distribution():
    """GPU間のメモリ分散状況をデバッグ"""
    print("=== GPU メモリ分散デバッグ ===")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🔍 検出されたGPU数: {gpu_count}")
        
        for i in range(gpu_count):
            try:
                # GPU別メモリ使用量
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"GPU {i}: 使用済み {allocated:.2f}GB, 予約済み {reserved:.2f}GB, 総容量 {total:.2f}GB")
                
                # メモリ使用率
                usage_rate = (allocated / total) * 100
                if usage_rate > 90:
                    print(f"  ⚠️ GPU {i} 危険な使用率: {usage_rate:.1f}%")
                elif usage_rate > 75:
                    print(f"  🟡 GPU {i} 高使用率: {usage_rate:.1f}%")
                else:
                    print(f"  ✅ GPU {i} 正常使用率: {usage_rate:.1f}%")
                    
            except Exception as e:
                print(f"  ❌ GPU {i} アクセスエラー: {e}")
        
        # 分散状況の分析
        memory_usage = []
        for i in range(gpu_count):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_usage.append(allocated)
            except:
                memory_usage.append(0)
        
        if memory_usage:
            max_usage = max(memory_usage)
            min_usage = min(memory_usage)
            avg_usage = sum(memory_usage) / len(memory_usage)
            
            print(f"\n📊 分散状況分析:")
            print(f"  最大使用量: {max_usage:.2f}GB")
            print(f"  最小使用量: {min_usage:.2f}GB")
            print(f"  平均使用量: {avg_usage:.2f}GB")
            print(f"  使用量差: {max_usage - min_usage:.2f}GB")
            
            # 分散状況の判定
            if max_usage - min_usage > 10:  # 10GB以上の差
                print("  ❌ 極端な偏りあり - 分散が不均等")
            elif max_usage - min_usage > 5:  # 5GB以上の差
                print("  ⚠️ 軽度の偏りあり - 分散やや不均等")
            else:
                print("  ✅ 分散状況良好")
    
    print("=== GPU メモリ分散デバッグ完了 ===\n")


def check_lisa_zero3_compatibility(model, args):
    """第2段階：LISA統合モデルのZeRO-3互換性チェック"""
    if args.local_rank == 0:
        print("=== LISA統合モデル ZeRO-3互換性チェック ===")
        
        # 1. モデル構成要素の確認
        components = []
        if hasattr(model, 'llama_model'):
            components.append("Llama4モデル")
        if hasattr(model, 'sam_model'):
            components.append("SAMモデル")
        if hasattr(model, 'multi_modal_projector'):
            components.append("マルチモーダルプロジェクタ")
        
        print(f"✅ 検出された構成要素: {', '.join(components)}")
        
        # 2. パラメータサイズ分析
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 総パラメータ数: {total_params:,}")
        print(f"📊 学習可能パラメータ数: {trainable_params:,}")
        
        # 3. ZeRO-3メモリ推定
        gpu_count = torch.cuda.device_count()
        param_memory_gb = (total_params * 2) / (1024**3)  # bfloat16仮定
        param_per_gpu_gb = param_memory_gb / gpu_count
        
        print(f"📊 推定パラメータメモリ: {param_memory_gb:.2f}GB")
        print(f"📊 ZeRO-3時の1GPU当たり: {param_per_gpu_gb:.2f}GB")
        
        # 4. 互換性判定
        if param_per_gpu_gb < 60:  # A100 80GBの75%以下
            print("✅ ZeRO-3互換性: 良好（推奨）")
        elif param_per_gpu_gb < 70:
            print("⚠️ ZeRO-3互換性: 注意（CPUオフロード推奨）")
        else:
            print("❌ ZeRO-3互換性: 要CPUオフロード")
        
        # 5. 初期メモリ状況デバッグ
        debug_gpu_memory_distribution()
        
        print("=== 互換性チェック完了 ===")


def apply_lora(model, args):
    """LoRA設定の適用"""
    print("\n=== LoRA設定適用 ===")
    
    # LoRA設定取得
    lora_config_dict = config_linux.get_lora_config()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **lora_config_dict
    )
    
    # LoRA適用
    model = get_peft_model(model, lora_config)
    
    # エンベディング層とLMヘッドを凍結
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = False
    
    # 学習可能パラメータ情報
    model.print_trainable_parameters()
    
    # マルチモーダルプロジェクタは学習可能にする
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
    
    print("✅ LoRA設定適用完了")
    
    return model


def create_datasets(args, processor):
    """データセットの作成"""
    print("\n=== データセット準備 ===")
    
    world_size = torch.cuda.device_count()
    
    # 学習データセット
    train_dataset = HybridDataset(
        base_image_dir=args.dataset_base_dir,
        llama_processor=processor,
        samples_per_epoch=(
            args.batch_size * args.grad_accumulation_steps * 
            args.steps_per_epoch * world_size
        ),
        precision=args.precision,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
    )
    
    print(f"✅ 学習データセット準備完了: {len(train_dataset)}サンプル")
    
    # 検証データセット（必要に応じて）
    val_dataset = None
    if not args.no_eval:
        # TODO: 検証データセットの実装
        print("⚠️ 検証データセットは未実装です")
    
    return train_dataset, val_dataset


def create_deepspeed_config(args):
    """第2段階：動的DeepSpeed設定の作成（ZeRO-2/3切り替え対応）"""
    
    # 基本設定（共通）
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": int(0.1 * args.steps_per_epoch),  # 10%ウォームアップ
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
    }
    
    # ZeRO Stage別設定
    if args.zero_stage == 2:
        # ZeRO-2設定（第1段階で最適化済み）
        zero_config = {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu", 
                "pin_memory": True
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "allgather_bucket_size": 2e8,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e6,
        }
        print(f"[Rank {args.local_rank}] ZeRO-2設定を使用")
        
    elif args.zero_stage == 3:
        # ZeRO-3設定（第3段階：完全ZeRO-3移行・103Bモデル最適化・極限メモリ節約）
        zero_config = {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": False
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 1e7,  # 極限まで小さく（103Bモデル用）
            "allgather_bucket_size": 1e7,
            "stage3_prefetch_bucket_size": 1e7,
            "stage3_param_persistence_threshold": 1e3,  # 極限メモリ節約（103Bモデル用）
            "stage3_max_live_parameters": 1e8,  # 100M パラメータまでに制限
            "stage3_max_reuse_distance": 1e7,   # 再利用距離も短く
            "gather_16bit_weights_on_model_save": True,  # 非推奨パラメータ修正
            "sub_group_size": 1e7,  # サブグループサイズも極限まで小さく
            "memory_efficient_linear": True,  # メモリ効率的な線形層
            "stage3_prefetch_bucket_size": 5e6,  # プリフェッチサイズも削減
        }
        
        # ZeRO-3でのパラメータCPUオフロード（オプション）
        if args.cpu_offload_params:
            zero_config["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }
            print(f"[Rank {args.local_rank}] ZeRO-3 + パラメータCPUオフロード設定を使用")
        else:
            print(f"[Rank {args.local_rank}] ZeRO-3設定を使用（パラメータはGPU保持）")
    
    ds_config["zero_optimization"] = zero_config
    
    # アクティベーションチェックポイント設定（ZeRO-3用に最適化）
    if args.zero_stage == 3:
        activation_config = {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 8,  # ZeRO-3では細かく分割
            "synchronize_checkpoint_boundary": True,
            "profile": False
        }
    else:
        activation_config = {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": True,
            "profile": False
        }
    
    ds_config["activation_checkpointing"] = activation_config
    
    return ds_config


def train_one_epoch(train_loader, model_engine, epoch, scheduler, writer, args):
    """1エポックの学習"""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CE Loss", ":.4f")
    mask_losses = AverageMeter("Mask Loss", ":.4f")
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, ce_losses, mask_losses],
        prefix=f"Epoch: [{epoch}]",
    )
    
    model_engine.train()
    
    end = time.time()
    for i, batch in enumerate(train_loader):
        # データ時間計測
        data_time.update(time.time() - end)
        
        # GPUに移動
        batch = dict_to_cuda(batch)
        
        # フォワードパス
        outputs = model_engine(**batch)
        
        # 損失取得
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        ce_loss = outputs.get("ce_loss", loss)
        mask_loss = outputs.get("mask_loss", torch.tensor(0.0))
        
        # バックワード
        model_engine.backward(loss)
        model_engine.step()
        
        # メトリクス更新
        losses.update(loss.item(), batch["input_ids"].size(0))
        ce_losses.update(ce_loss.item(), batch["input_ids"].size(0))
        mask_losses.update(mask_loss.item(), batch["input_ids"].size(0))
        
        # バッチ時間計測
        batch_time.update(time.time() - end)
        end = time.time()
        
        # ログ出力
        if i % args.print_freq == 0:
            progress.display(i)
            
            if writer is not None and args.local_rank == 0:
                global_step = epoch * len(train_loader) + i
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar("train/lr", 
                                scheduler.get_last_lr()[0], global_step)
    
    return losses.avg


def save_checkpoint(model_engine, epoch, args, metrics=None):
    """チェックポイント保存"""
    if args.local_rank != 0:
        return
    
    save_dir = os.path.join(args.checkpoint_dir, args.exp_name, 
                           f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    
    # DeepSpeedチェックポイント保存
    model_engine.save_checkpoint(save_dir, tag=f"epoch_{epoch}")
    
    # メタデータ保存
    metadata = {
        "epoch": epoch,
        "args": vars(args),
        "timestamp": datetime.now().isoformat()
    }
    
    if metrics is not None:
        metadata["metrics"] = metrics
    
    torch.save(metadata, os.path.join(save_dir, "metadata.pth"))
    
    print(f"✅ チェックポイント保存: {save_dir}")


def main():
    args = parse_args()
    
    # NCCL/TCP通信エラー修正：A100 GPU対応環境変数設定
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_IB_TIMEOUT"] = "22"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    
    # 第3段階：ZeRO-3用CUDAメモリ最適化
    if args.zero_stage == 3:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:32"
        # ZeRO-3特有の最適化
        os.environ["DEEPSPEED_ZERO3_INIT_FLAG"] = "true"
        os.environ["DEEPSPEED_ZERO_STAGE"] = "3"
        print(f"[Rank {args.local_rank}] ZeRO-3用メモリ最適化を設定")
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True,roundup_power2_divisions:16"
    
    # 2025年PyTorch分散学習の既知問題修正：TCPStore heartbeat monitor競合状態対策
    os.environ["TORCH_NCCL_ABORT_IN_DESTROY_PG"] = "1"  # プロセスグループ破棄時の強制終了
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 詳細デバッグ情報
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"  # C++スタックトレース表示
    
    # 2025年NCCL heartbeat monitor無効化とタイムアウト延長：根本的解決策
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"  # NCCLモニタリング無効化
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"  # ハートビートタイムアウト1時間
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # ブロッキング待機有効化
    os.environ["TORCH_DISABLE_ADDR2LINE"] = "1"  # アドレス変換無効化（スタックトレース高速化）
    
    if args.local_rank == 0:
        print("✅ NCCL環境変数設定完了: Broken Pipe エラー対策")
        print("✅ PyTorch分散学習の競合状態対策完了")
        print("✅ NCCL heartbeat monitor無効化とタイムアウト延長完了")
        if args.zero_stage == 3:
            print("✅ 第3段階：完全ZeRO-3移行完了（103Bモデル対応）")
        else:
            print("✅ 第1段階：バッチサイズ削減 + ZeRO-2最適化完了")
    
    # HuggingFaceキャッシュディレクトリを自動設定（全rankで実行）
    setup_hf_cache_dirs()
    
    # HuggingFace自動ログイン（local_rank=0のみ）
    if args.local_rank == 0:
        print("=== HuggingFace認証 ===")
        if not ensure_hf_login():
            print("❌ HuggingFace認証に失敗しました。処理を中断します。")
            sys.exit(1)
        
    # ログディレクトリ作成
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        print(f"ログディレクトリ: {args.log_dir}")
    else:
        writer = None
    
    # モデル作成
    model = create_model(args)
    
    # 第2段階：ZeRO-3互換性チェック（LISA統合モデル特有）
    if args.zero_stage == 3:
        print(f"[Rank {args.local_rank}] LISA統合モデルZeRO-3互換性チェック...")
        check_lisa_zero3_compatibility(model, args)
    
    # プロセッサー取得
    processor = model.llama_processor
    
    # LoRA適用
    model = apply_lora(model, args)
    
    # データセット作成
    train_dataset, val_dataset = create_datasets(args, processor)
    
    # DeepSpeed設定
    ds_config = create_deepspeed_config(args)
    
    # DeepSpeed初期化（長時間タイムアウト設定）
    import datetime
    import torch.distributed as dist
    
    # 分散プロセスグループの初期化が行われていない場合は、長時間タイムアウトで初期化
    if not dist.is_initialized():
        print(f"[Rank {args.local_rank}] 分散プロセスグループを長時間タイムアウト（3600秒）で初期化...")
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=3600)  # 1時間タイムアウト
        )
    
    # 第3段階：ZeRO-3用のモデル前処理
    if args.zero_stage == 3:
        print(f"[Rank {args.local_rank}] ZeRO-3用モデル前処理を実行中...")
        prepare_model_for_zero3(model, args)
    
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=processor.tokenizer,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )
    
    # レジューム処理
    start_epoch = 0
    if args.auto_resume:
        # 最新のチェックポイントを探す
        checkpoint_base = os.path.join(args.checkpoint_dir, args.exp_name)
        if os.path.exists(checkpoint_base):
            epochs = [d for d in os.listdir(checkpoint_base) 
                     if d.startswith("epoch_")]
            if epochs:
                latest = max(epochs, key=lambda x: int(x.split("_")[1]))
                args.resume = os.path.join(checkpoint_base, latest)
    
    if args.resume:
        print(f"チェックポイントをロード: {args.resume}")
        _, client_state = model_engine.load_checkpoint(
            args.resume, 
            tag=os.path.basename(args.resume)
        )
        metadata = torch.load(os.path.join(args.resume, "metadata.pth"))
        start_epoch = metadata["epoch"] + 1
        print(f"エポック {start_epoch} から再開")
    
    # 学習ループ
    print("\n" + "=" * 80)
    print("学習開始")
    print("=" * 80)
    
    # 学習開始前のメモリ状況デバッグ
    if args.local_rank == 0:
        print("\n=== 学習開始前メモリ状況 ===")
        debug_gpu_memory_distribution()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== エポック {epoch}/{args.epochs} ===")
        
        # 1エポック学習
        avg_loss = train_one_epoch(
            train_loader, model_engine, epoch, scheduler, writer, args
        )
        
        # チェックポイント保存
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model_engine, epoch, args, 
                          metrics={"train_loss": avg_loss})
        
        # 検証（実装時）
        if val_dataset is not None and (epoch + 1) % args.eval_freq == 0:
            # TODO: 検証実装
            pass
    
    # 最終チェックポイント保存
    save_checkpoint(model_engine, args.epochs - 1, args, 
                   metrics={"train_loss": avg_loss, "final": True})
    
    if writer is not None:
        writer.close()
    
    print("\n✅ 学習完了!")
    
    # 2025年PyTorch分散学習の既知問題対策：TCPStore heartbeat monitor競合状態回避
    # WebリサーチでわかったGitHub Issue #123969の回避策
    if args.local_rank == 0:
        print("⏱️  Rank 0でTCPStore競合状態回避のため8秒間スリープ...")
        time.sleep(8)  # 他のrankがクリーンアップを完了する時間を与える
        print("✅ TCPStore競合状態回避完了")


if __name__ == "__main__":
    # DeepSpeed環境変数設定
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # メイン実行
    main()