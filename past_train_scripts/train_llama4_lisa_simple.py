#!/usr/bin/env python3
"""
LISA-Llama4 シンプル分散学習スクリプト（DeepSpeedなし）
verify_llama4_lisa_gradients.pyとoverfit_llama4_lisa_batch.pyの成功手法を参考

実行方法:
# 最小時間テスト（10ステップ、1エポック）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train_llama4_lisa_simple.py --exp_name quick_test --steps_per_epoch 10 --epochs 1 --batch_size 1

# 通常テスト
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train_llama4_lisa_simple.py --exp_name lisa_simple --epochs 3 --batch_size 1
"""

import argparse
import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image

# Disable warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def setup_logging(rank: int, log_dir: str):
    """ロギング設定"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, 'train.log'))
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    return logging.getLogger(__name__)

def setup_distributed():
    """分散学習環境のセットアップ（検証スクリプト成功手法）"""
    if not dist.is_available():
        raise RuntimeError("分散学習が利用できません")
    
    if not dist.is_initialized():
        # 環境変数から分散学習設定を取得
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # PyTorch分散処理初期化
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        # GPU設定
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return rank, world_size, local_rank

def cleanup_distributed():
    """分散学習のクリーンアップ"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_model_and_tokenizer(rank: int, world_size: int, logger):
    """モデルとトークナイザーの作成（HuggingFaceキャッシュ競合対策付き）"""
    logger.info("=== LISA-Llama4統合モデル初期化 ===")
    
    # 動的コンパイルを無効化してGPU分散エラーを回避（成功した検証スクリプトと同じ設定）
    torch.compiler.disable()
    logger.info("動的コンパイル無効化: GPU分散エラー回避のため")
    
    # HuggingFaceキャッシュ競合対策：Rank 0が先にダウンロード
    if rank == 0:
        logger.info("🔄 Rank 0: モデルファイルをダウンロード中...")
        # LISA統合モデル設定（config_linux統一設定を使用）
        lisa_config = LisaLlama4Config(**config_linux.get_lisa_model_config())
        
        # Rank 0がモデルをダウンロード・初期化
        model = LisaLlama4ForCausalLM(lisa_config)
        logger.info("✅ Rank 0: モデルダウンロード完了、他のプロセスに同期信号送信")
    
    # 全プロセス同期：Rank 0のダウンロード完了を待つ
    if dist.is_initialized():
        dist.barrier()
        logger.info(f"✅ Rank {rank}: 同期完了、ローカルキャッシュからモデルロード開始")
    
    # Rank 0以外はlocal_files_onlyでキャッシュから読み込み
    if rank != 0:
        lisa_config = LisaLlama4Config(**config_linux.get_lisa_model_config())
        
        # ローカルキャッシュのみ使用してモデル初期化
        try:
            # HuggingFace Transformersにlocal_files_onlyヒントを与える
            original_offline = os.environ.get('HF_DATASETS_OFFLINE', '0')
            os.environ['HF_DATASETS_OFFLINE'] = '1'  # オフラインモードヒント
            
            model = LisaLlama4ForCausalLM(lisa_config)
            
            # 元の設定を復元
            os.environ['HF_DATASETS_OFFLINE'] = original_offline
            
        except Exception as e:
            logger.warning(f"Rank {rank}: キャッシュからの読み込み失敗、通常モードで再試行: {e}")
            model = LisaLlama4ForCausalLM(lisa_config)
    
    logger.info(f"✓ Rank {rank}: LISA統合モデル初期化完了")
    logger.info(f"  - 総パラメータ: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def apply_lora_config(model, logger):
    """LoRA設定適用（成功した検証スクリプト手法）"""
    logger.info("=== LoRA設定適用 ===")
    
    # LoRA設定作成（config_linux統一設定を使用）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **config_linux.get_lora_config()
    )
    
    # LoRA適用
    model = get_peft_model(model, lora_config)
    
    # 学習可能パラメータ統計
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"✓ LoRA適用完了")
    logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
    logger.info(f"  - 全パラメータ: {total_params:,}")
    logger.info(f"  - 学習可能割合: {100 * trainable_params / total_params:.3f}%")
    
    return model

def create_dataset_and_dataloader(rank: int, world_size: int, args, logger):
    """データセットとデータローダー作成（キャッシュ競合対策付き）"""
    logger.info("=== データセット初期化 ===")
    
    # データセットキャッシュ競合対策：Rank 0が先に初期化
    if rank == 0:
        logger.info("🔄 Rank 0: データセットを初期化中...")
        dataset = HybridDataset()
        logger.info("✅ Rank 0: データセット初期化完了、他のプロセスに同期信号送信")
    
    # 全プロセス同期：Rank 0のデータセット初期化完了を待つ
    if dist.is_initialized():
        dist.barrier()
        logger.info(f"✅ Rank {rank}: データセット同期完了")
    
    # Rank 0以外はキャッシュ済みデータセットを読み込み
    if rank != 0:
        dataset = HybridDataset()
    
    logger.info(f"✓ Rank {rank}: データセット初期化完了: {len(dataset)} サンプル")
    
    # DistributedSampler作成
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # DataLoader作成
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"✓ DataLoader作成完了: バッチサイズ={args.batch_size}, ワーカー数={args.workers}")
    
    return dataset, dataloader, sampler

def train_epoch(model, dataloader, optimizer, scheduler, epoch, args, rank, logger, writer=None):
    """1エポックの学習実行"""
    model.train()
    
    # メトリクス初期化
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(dataloader) if args.steps_per_epoch is None else args.steps_per_epoch,
        [losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # ステップ数制限チェック
        if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
            break
        
        # データをGPUに移動
        batch = dict_to_cuda(batch)
        
        # フォワードパス（検証スクリプト成功手法）
        optimizer.zero_grad()
        
        # LISA統合モデルの完全フォワードパス（SAM機能付き）
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            pixel_values=batch.get('pixel_values'),
            labels=batch['input_ids'],  # 言語モデリング用
            generate_mask=True  # SAM機能を有効化
        )
        
        # 損失計算（検証スクリプト成功手法）
        if isinstance(outputs, dict):
            if 'text_loss' in outputs:
                loss = outputs['text_loss']
            elif 'loss' in outputs:
                loss = outputs['loss']
            else:
                # フォールバック：手動計算
                logits = outputs.get('logits')
                if logits is None:
                    raise ValueError("logitsが見つかりません")
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["input_ids"][..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
        else:
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                raise ValueError("損失が見つかりません")
        
        # 逆伝播
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        
        # パラメータ更新
        optimizer.step()
        scheduler.step()
        
        # メトリクス更新
        losses.update(loss.item(), batch['input_ids'].size(0))
        
        # ログ出力
        if step % args.print_freq == 0 and rank == 0:
            progress.display(step, logger)
            
            # TensorBoard記録
            if writer is not None:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('Train/Loss', losses.val, global_step)
                writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
        
        # メモリクリーンアップ
        del outputs, loss
        torch.cuda.empty_cache()
    
    epoch_time = time.time() - start_time
    if rank == 0:
        logger.info(f"✓ エポック {epoch} 完了: 平均損失={losses.avg:.4f}, 時間={epoch_time:.1f}秒")
    
    return losses.avg

def main():
    parser = argparse.ArgumentParser(description="LISA-Llama4 シンプル分散学習")
    
    # 基本設定
    parser.add_argument("--exp_name", type=str, required=True, help="実験名")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ（GPU毎）")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学習率")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="勾配クリッピング")
    parser.add_argument("--workers", type=int, default=4, help="データローダーワーカー数")
    parser.add_argument("--print_freq", type=int, default=10, help="ログ出力頻度")
    parser.add_argument("--save_freq", type=int, default=1, help="チェックポイント保存頻度")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="エポックあたりのステップ数（制限）")
    
    # PyTorch分散学習用の引数（torch.distributed.launchが自動的に渡す）
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, help="ローカルランク（分散学習用）")
    
    args = parser.parse_args()
    
    # 分散学習環境セットアップ
    rank, world_size, local_rank = setup_distributed()
    
    # ログディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}_{timestamp}"
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    
    # ロギング設定
    logger = setup_logging(rank, log_dir)
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("LISA-Llama4 シンプル分散学習開始")
        logger.info("=" * 80)
        logger.info(f"実験名: {args.exp_name}")
        logger.info(f"Rank: {rank}/{world_size}")
        logger.info(f"Local Rank: {local_rank}")
        logger.info(f"バッチサイズ: {args.batch_size}")
        logger.info(f"エポック数: {args.epochs}")
        logger.info(f"学習率: {args.lr}")
        if args.steps_per_epoch:
            logger.info(f"ステップ制限: {args.steps_per_epoch}/エポック")
    
    try:
        # HuggingFace認証確認
        if rank == 0:
            ensure_hf_login()
        dist.barrier()  # 全プロセス同期
        
        # モデル初期化（Model Parallelism対応）
        model = create_model_and_tokenizer(rank, world_size, logger)
        model = apply_lora_config(model, logger)
        
        # GPU移動とModel Parallelism設定
        if rank == 0:
            logger.info("=== Model Parallelism設定 ===")
            logger.info(f"103Bモデルを{world_size}GPUに分散配置")
        
        # モデル分散配置（DDP前に実行）
        try:
            # device_mapが既に設定されている場合はそのまま使用
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                if rank == 0:
                    logger.info("✅ モデルは既にdevice_mapで分散配置済み")
                    logger.info(f"Device map: {dict(list(model.hf_device_map.items())[:5])}...")
            else:
                # 手動でGPUに移動（小さなモデルの場合）
                model = model.cuda(local_rank)
                if rank == 0:
                    logger.info(f"✅ モデルをGPU {local_rank}に配置")
                    
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"❌ Rank {rank}: GPU {local_rank}でOOMエラー: {e}")
            raise RuntimeError(f"モデルが大きすぎます。DeepSpeedまたはより小さなモデルを使用してください") from e
        
        # DDP ラッピング（Model Parallelismと互換性確保）
        try:
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                # Device mapが設定されている場合はDDPをスキップ
                if rank == 0:
                    logger.warning("⚠️ Device mapが設定されているため、DDPをスキップします")
                    logger.info("Model Parallelismモードで実行")
            else:
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                if rank == 0:
                    logger.info("✅ DDP初期化完了")
        except Exception as e:
            logger.error(f"❌ Rank {rank}: DDP初期化エラー: {e}")
            if rank == 0:
                logger.info("Model Parallelismモードで継続")
            # DDPなしで継続
        
        # データセットとデータローダー
        dataset, dataloader, sampler = create_dataset_and_dataloader(rank, world_size, args, logger)
        
        # オプティマイザーとスケジューラー
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        total_steps = len(dataloader) * args.epochs
        if args.steps_per_epoch:
            total_steps = args.steps_per_epoch * args.epochs
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
        
        # TensorBoard
        writer = None
        if rank == 0:
            writer = SummaryWriter(f"./runs/{args.exp_name}_{timestamp}")
        
        # 学習ループ
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            
            avg_loss = train_epoch(
                model, dataloader, optimizer, scheduler, 
                epoch, args, rank, logger, writer
            )
            
            # チェックポイント保存
            if rank == 0 and epoch % args.save_freq == 0:
                checkpoint_dir = f"./checkpoints/{args.exp_name}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
                
                logger.info(f"✓ チェックポイント保存: epoch_{epoch}.pt")
        
        if rank == 0:
            logger.info("=" * 80)
            logger.info("✅ 学習完了!")
            logger.info(f"ログディレクトリ: {log_dir}")
            logger.info(f"チェックポイント: ./checkpoints/{args.exp_name}")
            if writer:
                logger.info(f"TensorBoard: tensorboard --logdir=./runs/{args.exp_name}_{timestamp}")
            logger.info("=" * 80)
        
        if writer:
            writer.close()
    
    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()