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

# メモリ最適化設定（🔧 修正方針T: 極限メモリ管理）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5,roundup_power2_divisions:32'
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
# from torch.cuda.amp import GradScaler  # 🔧 修正方針E: GradScaler不使用
import transformers
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# メモリ最適化設定（🔧 修正方針V: 最終手段メモリ解決）
if torch.cuda.is_available():
    # 初期化段階は完全無制限（caching_allocator_warmup OOM回避）
    torch.cuda.set_per_process_memory_fraction(1.0)  # 完全無制限
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    print("🔧 CUDA memory fraction UNLIMITED (修正方針V: 緊急対策)")

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
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
# from utils.segmentation_eval import evaluate_segmentation_metrics  # 存在しないため削除
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
        from transformers import Llama4ForConditionalGeneration
        
        # 🔧 修正方針W: BitsAndBytesConfig 4-bit量子化でメモリ削減（旧モデル成功パターン＋Webリサーチ）
        logger.info("🔧 緊急メモリクリーンアップ実行...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        
        # 4-bit量子化設定（train_phase3b_qformer_bridge.py成功パターン準拠）
        if BITSANDBYTES_AVAILABLE:
            logger.info("🔧 BitsAndBytesConfig 4-bit量子化でLlama-4読み込み...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # ダブル量子化でさらなるメモリ削減
            )
            
            llama4_model = Llama4ForConditionalGeneration.from_pretrained(
                config_linux.LLAMA_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",  # 自動分散配置
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=config_linux.ATTN_IMPLEMENTATION,  # config統一
                low_cpu_mem_usage=True,
                offload_state_dict=True,  # チェックポイント読み込み最適化
            )
            logger.info("✅ 4-bit量子化読み込み完了（約75%メモリ削減）")
        else:
            logger.warning("⚠️ BitsAndBytesConfig未利用: 従来方式でフォールバック")
            llama4_model = Llama4ForConditionalGeneration.from_pretrained(
                config_linux.LLAMA_MODEL_ID,
                device_map="auto",  # 2025年推奨: 自動分散配置
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=config_linux.ATTN_IMPLEMENTATION,  # config統一
                low_cpu_mem_usage=True,
            )
            logger.info("✅ 従来方式読み込み完了")
        
        llama4_processor = AutoProcessor.from_pretrained(
            config_linux.LLAMA_MODEL_ID,
            trust_remote_code=True
        )
        
        logger.info("✅ Llama-4初期化完了")
        
        # 2. 改修版統合モデル初期化
        logger.info("🔧 Enhanced QFormerSegmentationBridge初期化...")
        
        config = EnhancedLlamaQFormerSAM2Config()
        config.training_stage = args.training_stage
        
        # 🔧 修正方針W: test_phase3b_enhanced_multiscale_lora.py準拠のSAM2最適化
        config.sam2_config['compile_model'] = False  # メモリ効率化
        logger.info("✅ SAM2 compileモード無効化（メモリ効率化）")
        
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
        
        # 🔧 修正方針O: Webリサーチ結果に基づく確定的gradient checkpointing設定
        if args.gradient_checkpointing:
            # Llama-4: Hugging Face transformersの標準API
            if hasattr(model.llama_model, 'gradient_checkpointing_enable'):
                model.llama_model.gradient_checkpointing_enable()
                logger.info("✅ Llama-4勾配チェックポイント有効化")
            else:
                logger.warning("⚠️ Llama-4: gradient_checkpointing_enable未対応")
            
            # SAM2: PyTorchの標準torch.utils.checkpoint APIを使用
            # SAM2（facebook/sam2）には独自のgradient checkpointing APIは存在しない
            # Webリサーチ確認済み: PyTorchネイティブのcheckpoint機能を使用する必要がある
            logger.info("✅ SAM2勾配チェックポイント: PyTorchネイティブAPI使用予定")
            logger.info("  - torch.utils.checkpoint.checkpoint()をモデル内で適用")
            logger.info("  - メモリ削減効果: 約60%, 計算時間増加: 約20-30%")
        
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



def setup_data_loaders(args, logger, llama_processor):
    """データローダー設定（既存HybridDataset使用）"""
    logger.info("=== データローダー設定 ===")
    
    # データセット設定（train_phase3b_qformer_bridge.py参考）
    if hasattr(args, 'dataset') and args.dataset:
        datasets = [ds.strip() for ds in args.dataset.split(',')]
        dataset_string = "||".join(datasets)
    else:
        dataset_string = "reason_seg"  # デフォルト
    
    # サンプル数設定
    if args.steps_per_epoch:
        samples_per_epoch = args.steps_per_epoch * args.batch_size
    else:
        samples_per_epoch = 500  # デフォルト
    
    # HybridDataset初期化
    logger.info("🔍 HybridDataset初期化前デバッグ情報:")
    logger.info(f"  - base_image_dir: {config_linux.DATASET_BASE_DIR}")
    logger.info(f"  - base_image_dir存在確認: {os.path.exists(config_linux.DATASET_BASE_DIR)}")
    logger.info(f"  - dataset_string: {dataset_string}")
    
    dataset = HybridDataset(
        base_image_dir=config_linux.DATASET_BASE_DIR,
        llama_processor=llama_processor,
        samples_per_epoch=samples_per_epoch,
        precision="bf16",
        llama_image_size=config_linux.LLAMA_IMAGE_SIZE,
        sam_image_size=config_linux.SAM_IMAGE_SIZE,
        num_classes_per_sample=3,
        exclude_val=True,
        dataset=dataset_string,
        sample_rate=[9, 3, 3, 1],  # reason_seg, refer_seg, vqa, sem_seg の比率
        sem_seg_data=config_linux.SEM_SEG_DATA,
        refer_seg_data=config_linux.REFER_SEG_DATA,
        vqa_data=config_linux.VQA_DATA,
        reason_seg_data=config_linux.REASON_SEG_DATA,
        explanatory=0.1
    )
    
    logger.info(f"HybridDataset初期化完了: {len(dataset)} サンプル")
    
    # データローダー
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=False,  # 🔧 修正方針M: メモリ使用量削減
        drop_last=True,
        persistent_workers=False  # 🔧 修正方針M: ワーカープロセス再利用無効化でメモリリーク対策
    )
    
    logger.info(f"✅ データローダー設定完了")
    logger.info(f"  - 訓練サンプル数: {len(dataset)}")
    logger.info(f"  - バッチサイズ: {args.batch_size}")
    logger.info(f"  - データセット: {dataset_string}")
    
    return train_loader


def train_one_epoch(
    model: EnhancedQFormerSegmentationBridge,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: None,  # 🔧 修正方針E: GradScaler無効化
    epoch: int,
    args: Any,
    logger: logging.Logger,
    writer: SummaryWriter,
    memory_monitor: GPUMemoryMonitor
) -> Dict[str, float]:
    """1エポックの訓練（改修版対応）"""
    model.train()
    
    epoch_losses = []
    # 🔧 修正方針J: md_files仕様準拠の損失追跡
    epoch_metrics = {
        'total_loss': 0.0,
        'focal_tversky_loss': 0.0,  # セグメンテーション：高性能Focal Tversky
        'lovasz_loss': 0.0,         # セグメンテーション：IoU直接最適化
        'dice_loss': 0.0,           # セグメンテーション：BCE、バランス調整
        'qformer_total': 0.0,       # Q-Former：統合マルチモーダル損失
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
        
        # 🔧 修正方針X: outputs変数スコープ修正（例外ハンドリング対応）
        outputs = None
        loss = None
        loss_dict = {}
        
        try:
            # HybridDatasetの出力をデュアルエンコーダー対応に変換
            adapted_batch = adapt_dataset_for_qformer(batch)
            
            # デバイス転送
            images = adapted_batch['images'].cuda()
            sam_images = adapted_batch.get('sam_images', None)  # SAM2用画像
            if sam_images is not None:
                sam_images = sam_images.cuda()
            labels = adapted_batch['labels'].cuda()
            
            # テキスト入力
            text_input = adapted_batch.get('text_input', None)
            if text_input is None:
                # デフォルトテキスト生成
                text_input = [f"Segment the objects in this image." for _ in range(images.size(0))]
            
            # 🔧 修正方針E: BFloat16ネイティブ学習のためautocast範囲を推論のみに限定
            # 🔧 修正方針M: メモリ最適化
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Forward処理（推論部分のみautocast）
                outputs = model(
                    images=images,
                    sam_images=sam_images,  # SAM2用画像を追加
                    text_input=text_input,
                    labels=labels,
                    mode=current_mode
                )
            
            # 🔧 修正方針X: outputs検証 - None判定を追加
            if outputs is None:
                logger.error("❌ モデル出力がNone - スキップ")
                continue
            
            # Loss計算はautocast外で実行（BFloat16ネイティブ）
            loss = outputs['loss']
            loss_dict = outputs.get('loss_dict', {})
            
            # 🔧 修正方針E: BFloat16ネイティブ学習（GradScaler不使用）
            # BFloat16は数値安定性が高いため、直接backward()を呼び出し
            loss.backward()
            
            # 勾配クリッピング（BFloat16対応）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 最適化ステップ（スケーリング無し）
            optimizer.step()
            optimizer.zero_grad()
            
            # メトリクス更新
            epoch_losses.append(loss.item())
            epoch_metrics['total_loss'] += loss.item()
            
            # 🔧 修正方針R: loss_dict変数スコープ修正 - メモリクリーンアップ前に処理
            # セグメンテーション損失（個別追跡）
            for seg_key in ['focal_tversky_loss', 'lovasz_loss', 'dice_loss']:
                if seg_key in loss_dict:
                    epoch_metrics[seg_key] += loss_dict[seg_key].item()
            
            # Q-Former損失（統合追跡）
            qformer_total = 0.0
            for qformer_key in ['qformer_itc_loss', 'qformer_itm_loss', 'qformer_itg_loss']:
                if qformer_key in loss_dict:
                    qformer_total += loss_dict[qformer_key].item()
            epoch_metrics['qformer_total'] += qformer_total
            
            # IoU/Dice計算（マスク出力がある場合）- メモリクリーンアップ前に実行
            if outputs is not None and 'masks' in outputs:
                masks = outputs['masks']
                # 最良のマスクを選択
                if masks.size(1) > 1:
                    iou_scores = outputs.get('iou_scores', None)
                    if iou_scores is not None:
                        best_idx = iou_scores.argmax(dim=1)
                        masks = masks[torch.arange(masks.size(0)), best_idx].unsqueeze(1)
                    else:
                        masks = masks[:, 0:1]
                
                # 簡易メトリクス計算
                pred_binary = (masks > 0.5).float()
                target_binary = labels.unsqueeze(1).float()
                
                # IoU計算
                intersection = (pred_binary * target_binary).sum(dim=(2, 3))
                union = pred_binary.sum(dim=(2, 3)) + target_binary.sum(dim=(2, 3)) - intersection
                iou = (intersection / (union + 1e-6)).mean().item()
                
                # Dice計算
                dice = (2 * intersection / (pred_binary.sum(dim=(2, 3)) + target_binary.sum(dim=(2, 3)) + 1e-6)).mean().item()
                
                epoch_metrics['iou'] += iou
                epoch_metrics['dice'] += dice
            
            # プログレスバー更新 - loss安全性チェック
            if loss is not None:
                current_lr = optimizer.param_groups[0]['lr']
                
                # 🔧 修正方針Z: 学習率デバッグログ追加
                if batch_idx == 0:  # 各エポックの最初のバッチでログ
                    logger.info(f"🔍 エポック{epoch} バッチ{batch_idx} 学習率: {current_lr}")
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mode': current_mode,
                    'lr': f"{current_lr:.2e}"
                })
                
                # TensorBoard記録
                global_step = epoch * len(train_loader) + batch_idx
                if batch_idx % 10 == 0:
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    writer.add_scalar('Train/LR', current_lr, global_step)
            
            # 🔧 修正方針M: 積極的メモリクリーンアップ
            if outputs is not None:
                del outputs
            if loss is not None:
                del loss
            if loss_dict:
                del loss_dict
            # バッチ毎のメモリクリーンアップ（OOM対策）
            del images, sam_images, labels, adapted_batch
            if 'text_input' in locals():
                del text_input
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # 5バッチ毎により徹底的なクリーンアップ
            if batch_idx > 0 and batch_idx % 5 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            logger.error(f"❌ バッチ処理エラー: {e}")
            memory_summary = memory_monitor.get_memory_summary()
            logger.info(f"📊 エラー時メモリ状況: {memory_summary}")
            emergency_cleanup()
            continue
        
        finally:
            pass  # メモリ監視完了
    
    # エポック平均計算
    num_batches = min(len(train_loader), args.steps_per_epoch or len(train_loader))
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    # エポック終了時のログ（md_files仕様準拠）
    logger.info(f"📊 エポック {epoch} 完了:")
    logger.info(f"  - 平均損失: {epoch_metrics['total_loss']:.4f}")
    logger.info(f"  - Focal Tversky Loss: {epoch_metrics['focal_tversky_loss']:.4f}")
    logger.info(f"  - Lovász Loss: {epoch_metrics['lovasz_loss']:.4f}")
    logger.info(f"  - Dice Loss: {epoch_metrics['dice_loss']:.4f}")
    logger.info(f"  - Q-Former損失: {epoch_metrics['qformer_total']:.4f}")
    logger.info(f"  - mIoU: {epoch_metrics['iou']:.4f}")
    logger.info(f"  - mDice: {epoch_metrics['dice']:.4f}")
    
    return epoch_metrics


def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser(description='Enhanced Phase 3B Training')
    parser.add_argument('--exp_name', type=str, default='phase3b_enhanced',
                        help='実験名')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='バッチサイズ（🔧 修正方針M: OOM対策で1に縮小）')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='勾配チェックポイント有効化（メモリ削減）')
    parser.add_argument('--epochs', type=int, default=3,
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学習率')
    parser.add_argument('--workers', type=int, default=1,
                        help='データローダーワーカー数（🔧 修正方針M: メモリリーク対策で1に縮小）')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='エポックあたりのステップ数（テスト用）')
    parser.add_argument('--training_stage', type=int, default=1,
                        help='訓練ステージ (1-3)')
    parser.add_argument('--disable_lora', action='store_true',
                        help='LoRAを無効化')
    parser.add_argument('--disable_multiscale', action='store_true',
                        help='マルチスケールを無効化')
    parser.add_argument('--dataset', type=str, default='reason_seg',
                        help='使用するデータセット（カンマ区切り）')
    
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
        train_loader = setup_data_loaders(args, logger, model_components['llama4_processor'])
        
        # オプティマイザー設定
        logger.info("📊 オプティマイザー設定...")
        # 🔧 修正方針D: BFloat16+GradScaler互換性のため標準AdamWを使用
        logger.info("🔧 BFloat16最適化: 8bit AdamW → 標準AdamW (GradScaler互換性)")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),  # BFloat16推奨設定
            eps=1e-8
        )
        logger.info("✅ 標準AdamW使用 (BFloat16最適化)")
        
        # 🔧 bitsandbytes 8bit AdamWは一時的に無効化
        # if BITSANDBYTES_AVAILABLE:
        #     optimizer = bnb.optim.AdamW8bit(
        #         model.parameters(),
        #         lr=args.lr,
        #         weight_decay=0.01
        #     )
        #     logger.info("✅ 8bit AdamW使用")
        
        # スケジューラー設定
        num_training_steps = len(train_loader) * args.epochs
        
        # 🔧 修正方針Z: 学習率スケジューラーデバッグと修正
        logger.info(f"🔍 スケジューラー設定デバッグ:")
        logger.info(f"  - len(train_loader): {len(train_loader)}")
        logger.info(f"  - args.epochs: {args.epochs}")
        logger.info(f"  - num_training_steps: {num_training_steps}")
        logger.info(f"  - 初期学習率: {args.lr}")
        
        # 短いステップ数での問題回避: warmup_stepsを1に固定
        warmup_steps = 1  # 最小値に設定（10ステップなので0.1*10=1でも多い）
        logger.info(f"  - warmup_steps: {warmup_steps}")
        
        # 短期訓練用: より緩やかなコサインスケジュール
        if num_training_steps <= 20:  # 短期訓練の場合
            logger.info("🔧 短期訓練検出: 固定学習率に変更")
            scheduler = None  # スケジューラー無効化
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # 初期学習率確認
        initial_lr = optimizer.param_groups[0]['lr']
        logger.info(f"🔍 初期学習率確認: {initial_lr}")
        if initial_lr == 0.0:
            logger.error("❌ 初期学習率が0です！オプティマイザー設定を確認してください。")
            raise ValueError("初期学習率が0に設定されています")
        
        # 🔧 修正方針E: BFloat16ネイティブ学習のためGradScaler完全無効化
        # BFloat16は数値安定性が高いため、勾配スケーリング不要
        scaler = None  # GradScaler完全無効化
        logger.info("🔧 GradScaler無効化: BFloat16ネイティブ学習モード")
        
        # 🔧 修正方針W: 旧モデル成功パターン準拠のメモリ制限適用
        logger.info("🔧 訓練段階: メモリ制限適用 (無制限 → 70% 旧モデル成功パターン準拠)")
        torch.cuda.set_per_process_memory_fraction(0.7)  # train_phase3b_qformer_bridge.py成功値
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 訓練ループ
        logger.info("\n🚀 訓練開始")
        logger.info(f"  - エポック数: {args.epochs}")
        logger.info(f"  - バッチサイズ: {args.batch_size}")
        logger.info(f"  - 学習率: {args.lr}")
        logger.info(f"  - メモリ制限: 70% (旧モデル成功パターン準拠)")
        
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
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"🔍 エポック{epoch}後の学習率: {current_lr}")
            else:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"🔍 固定学習率: {current_lr}")
            
            # チェックポイント保存
            if epoch % 1 == 0:  # 毎エポック保存
                checkpoint_path = log_dir / f"checkpoint_epoch_{epoch}.pt"
                
                # 🔧 修正方針I: BFloat16ネイティブ学習対応チェックポイント保存
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': epoch_metrics,
                    'args': vars(args),
                    'training_mode': 'bfloat16_native'  # 学習モード識別
                }
                
                # スケジューラー状態保存（存在する場合のみ）
                if scheduler is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                    logger.info("💾 スケジューラー状態も保存")
                else:
                    logger.info("💾 固定学習率モード: スケジューラー状態なし")
                
                # GradScaler状態保存（存在する場合のみ）
                if scaler is not None:
                    checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                    logger.info("💾 GradScaler状態も保存")
                else:
                    logger.info("💾 BFloat16ネイティブモード: GradScaler状態なし")
                
                torch.save(checkpoint_data, checkpoint_path)
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
        memory_summary = memory_monitor.get_memory_summary()
        logger.info(f"📊 メモリ使用量サマリ: {memory_summary}")
        logger.info("🧹 クリーンアップ完了")


if __name__ == "__main__":
    main()