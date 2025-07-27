#!/usr/bin/env python3
"""
Enhanced Phase 3B QFormerSegmentationBridge 訓練スクリプト
o3-modification20250727.mdに基づく改修版

改修内容:
1. Enhanced Q-Former (BLIP-2準拠テキスト入力対応)
2. 視覚・言語特徴分離機構
3. LoRAエキスパート統合
4. マルチスケールセグメンテーションヘッド

実行方法:
# 最小時間テスト（10ステップ）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip> "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u train_phase3b_enhanced.py --exp_name quick_test --steps_per_epoch 10 --epochs 1 2>&1"

# 通常訓練
ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip> "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u train_phase3b_enhanced.py --exp_name phase3b_enhanced --epochs 3 --steps_per_epoch 50 2>&1"
"""

# PyTorchインポート前の環境準備
import sys
import os

# 環境変数設定
print("🔧 CUDA環境設定...")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

# メモリ最適化設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8,roundup_power2_divisions:8'
print("✅ 環境変数設定完了")

# 基本インポート
import argparse
import time
import json
import logging
import gc
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, Any, List, Tuple, Optional

# PyTorchインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import transformers
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# メモリ最適化設定
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)
    torch.cuda.empty_cache()
    print("🔧 CUDA memory fraction set to 0.7")

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available. Using standard AdamW optimizer.")

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 警告抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# メモリモニタリング
from utils.memory_monitor import GPUMemoryMonitor, memory_monitor_section, log_memory_status, emergency_cleanup

# 改修版モデルとユーティリティ
from model.enhanced_llama4_qformer_sam2 import EnhancedQFormerSegmentationBridge, EnhancedLlamaQFormerSAM2Config
from model.dataset_adapter import adapt_dataset_for_qformer
from utils.dataset import HybridDataset, collate_fn, preprocess_sam_image, build_correct_labels_for_llama4
from utils.constants import DEFAULT_SEG_TOKEN
from utils.segmentation_eval import evaluate_segmentation_metrics
import config_linux


def setup_logging(exp_name: str):
    """ロギング設定（既存と同じ）"""
    # ログディレクトリ作成
    log_dir = Path(f"lambda_results/runs/{exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # タイムスタンプ付きログファイル
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Enhanced Phase 3B訓練開始 ===")
    logger.info(f"実験名: {exp_name}")
    logger.info(f"ログファイル: {log_file}")
    
    return logger, log_dir


def initialize_enhanced_model(args, logger):
    """改修版モデルの初期化"""
    logger.info("=== 改修版統合モデル初期化 ===")
    
    try:
        # 1. Llama-4初期化
        logger.info("🧠 Llama-4-Scout初期化...")
        from transformers import AutoModelForCausalLM
        
        llama4_model = AutoModelForCausalLM.from_pretrained(
            config_linux.LLAMA_MODEL_ID,
            device_map="balanced_low_0",  # GPU負荷分散
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        llama4_processor = AutoProcessor.from_pretrained(
            config_linux.LLAMA_MODEL_ID,
            trust_remote_code=True
        )
        
        logger.info("✅ Llama-4初期化完了")
        
        # 2. 改修版統合モデル初期化
        logger.info("🔧 Enhanced QFormerSegmentationBridge初期化...")
        
        config = EnhancedLlamaQFormerSAM2Config()
        config.training_stage = args.training_stage
        
        # LoRAとマルチスケール設定
        enable_lora = not args.disable_lora
        enable_multiscale = not args.disable_multiscale
        
        model = EnhancedQFormerSegmentationBridge(
            config=config,
            shared_llama_model=llama4_model,
            shared_llama_processor=llama4_processor,
            enable_lora=enable_lora,
            enable_multiscale=enable_multiscale,
            training_stage=args.training_stage
        )
        
        logger.info("✅ 改修版モデル初期化完了")
        logger.info(f"  - テキスト入力: 対応")
        logger.info(f"  - 視覚・言語分離: 有効")
        logger.info(f"  - LoRA: {'有効' if enable_lora else '無効'}")
        logger.info(f"  - マルチスケール: {'有効' if enable_multiscale else '無効'}")
        
        return {
            'model': model,
            'llama4_model': llama4_model,
            'llama4_processor': llama4_processor
        }
        
    except Exception as e:
        logger.error(f"❌ モデル初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        raise


def setup_data_loaders(args, logger):
    """データローダー設定（テキスト入力対応版）"""
    logger.info("=== データローダー設定 ===")
    
    # データセット作成
    train_dataset = HybridDataset(
        dataset_dirs=config_linux.DATASET_DIRS,
        dataset_types=config_linux.DATASET_TYPES,
        dataset_samples=config_linux.DATASET_SAMPLES,
        split='train',
        max_samples_per_dataset=1000,
        resize_size=(448, 448),
        clip_processor=None,  # Q-Formerが処理
        include_text=True  # テキスト入力を含める
    )
    
    # Q-Former用に適応
    train_dataset = adapt_dataset_for_qformer(
        train_dataset, 
        qformer_type='enhanced',  # 改修版指定
        qformer_config={'max_txt_len': 64}
    )
    
    # データローダー
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"✅ データローダー設定完了")
    logger.info(f"  - 訓練サンプル数: {len(train_dataset)}")
    logger.info(f"  - バッチサイズ: {args.batch_size}")
    logger.info(f"  - テキスト入力: 有効")
    
    return train_loader


def train_one_epoch(
    model: EnhancedQFormerSegmentationBridge,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    args: Any,
    logger: logging.Logger,
    writer: SummaryWriter,
    memory_monitor: GPUMemoryMonitor
) -> Dict[str, float]:
    """1エポックの訓練（改修版対応）"""
    model.train()
    
    epoch_losses = []
    epoch_metrics = {
        'total_loss': 0.0,
        'seg_loss': 0.0,
        'text_loss': 0.0,
        'iou': 0.0,
        'dice': 0.0
    }
    
    # 学習モードの選択（エポックごとに切り替え）
    modes = ['itc', 'itm', 'itg']
    current_mode = modes[epoch % len(modes)]
    logger.info(f"📚 エポック {epoch} - Q-Former学習モード: {current_mode}")
    
    # プログレスバー設定
    from tqdm import tqdm
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # ステップ制限（テスト時）
        if args.steps_per_epoch and batch_idx >= args.steps_per_epoch:
            break
        
        # メモリ監視
        memory_monitor.start_monitoring("batch_forward")
        
        try:
            # デバイス転送
            images = batch['images'].cuda()
            labels = batch['labels'].cuda()
            
            # テキスト入力（ある場合）
            text_input = batch.get('text_input', None)
            if text_input is None:
                # デフォルトテキスト生成
                text_input = [f"Segment the objects in this image." for _ in range(images.size(0))]
            
            # Mixed Precision Training
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Forward処理
                outputs = model(
                    images=images,
                    text_input=text_input,
                    labels=labels,
                    mode=current_mode
                )
                
                loss = outputs['loss']
                loss_dict = outputs.get('loss_dict', {})
            
            # Backward処理
            scaler.scale(loss).backward()
            
            # 勾配クリッピング
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 最適化ステップ
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # メトリクス更新
            epoch_losses.append(loss.item())
            epoch_metrics['total_loss'] += loss.item()
            
            if 'seg_loss' in loss_dict:
                epoch_metrics['seg_loss'] += loss_dict['seg_loss'].item()
            if 'text_loss' in loss_dict:
                epoch_metrics['text_loss'] += loss_dict['text_loss'].item()
            
            # IoU/Dice計算（マスク出力がある場合）
            if 'masks' in outputs:
                masks = outputs['masks']
                # 最良のマスクを選択
                if masks.size(1) > 1:
                    iou_scores = outputs.get('iou_scores', None)
                    if iou_scores is not None:
                        best_idx = iou_scores.argmax(dim=1)
                        masks = masks[torch.arange(masks.size(0)), best_idx].unsqueeze(1)
                    else:
                        masks = masks[:, 0:1]
                
                # メトリクス計算
                metrics = evaluate_segmentation_metrics(
                    masks > 0.5,
                    labels.unsqueeze(1),
                    num_classes=2
                )
                epoch_metrics['iou'] += metrics['mIoU']
                epoch_metrics['dice'] += metrics['mDice']
            
            # プログレスバー更新
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mode': current_mode,
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # TensorBoard記録
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
            
        except Exception as e:
            logger.error(f"❌ バッチ処理エラー: {e}")
            memory_monitor.print_summary()
            emergency_cleanup()
            continue
        
        finally:
            memory_monitor.stop_monitoring("batch_forward")
    
    # エポック平均計算
    num_batches = min(len(train_loader), args.steps_per_epoch or len(train_loader))
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    # エポック終了時のログ
    logger.info(f"📊 エポック {epoch} 完了:")
    logger.info(f"  - 平均損失: {epoch_metrics['total_loss']:.4f}")
    logger.info(f"  - セグメンテーション損失: {epoch_metrics['seg_loss']:.4f}")
    logger.info(f"  - テキスト損失: {epoch_metrics['text_loss']:.4f}")
    logger.info(f"  - mIoU: {epoch_metrics['iou']:.4f}")
    logger.info(f"  - mDice: {epoch_metrics['dice']:.4f}")
    
    return epoch_metrics


def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser(description='Enhanced Phase 3B Training')
    parser.add_argument('--exp_name', type=str, default='phase3b_enhanced',
                        help='実験名')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=3,
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学習率')
    parser.add_argument('--workers', type=int, default=4,
                        help='データローダーワーカー数')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='エポックあたりのステップ数（テスト用）')
    parser.add_argument('--training_stage', type=int, default=1,
                        help='訓練ステージ (1-3)')
    parser.add_argument('--disable_lora', action='store_true',
                        help='LoRAを無効化')
    parser.add_argument('--disable_multiscale', action='store_true',
                        help='マルチスケールを無効化')
    
    args = parser.parse_args()
    
    # ロギング設定
    logger, log_dir = setup_logging(args.exp_name)
    
    # TensorBoard設定
    writer = SummaryWriter(log_dir / 'tensorboard')
    
    # メモリモニター
    memory_monitor = GPUMemoryMonitor()
    
    try:
        # モデル初期化
        model_components = initialize_enhanced_model(args, logger)
        model = model_components['model']
        
        # データローダー設定
        train_loader = setup_data_loaders(args, logger)
        
        # オプティマイザー設定
        logger.info("📊 オプティマイザー設定...")
        if BITSANDBYTES_AVAILABLE:
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=args.lr,
                weight_decay=0.01
            )
            logger.info("✅ 8bit AdamW使用")
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=0.01
            )
            logger.info("✅ 標準AdamW使用")
        
        # スケジューラー設定
        num_training_steps = len(train_loader) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        # GradScaler設定
        scaler = GradScaler()
        
        # 訓練ループ
        logger.info("\n🚀 訓練開始")
        logger.info(f"  - エポック数: {args.epochs}")
        logger.info(f"  - バッチサイズ: {args.batch_size}")
        logger.info(f"  - 学習率: {args.lr}")
        
        for epoch in range(1, args.epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"エポック {epoch}/{args.epochs}")
            logger.info(f"{'='*50}")
            
            # 1エポックの訓練
            epoch_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                args=args,
                logger=logger,
                writer=writer,
                memory_monitor=memory_monitor
            )
            
            # スケジューラー更新
            scheduler.step()
            
            # チェックポイント保存
            if epoch % 1 == 0:  # 毎エポック保存
                checkpoint_path = log_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'metrics': epoch_metrics,
                    'args': vars(args)
                }, checkpoint_path)
                logger.info(f"💾 チェックポイント保存: {checkpoint_path}")
            
            # メモリクリーンアップ
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("\n✅ 訓練完了!")
        
    except Exception as e:
        logger.error(f"❌ 訓練エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # クリーンアップ
        writer.close()
        memory_monitor.print_summary()
        logger.info("🧹 クリーンアップ完了")


if __name__ == "__main__":
    main()