#!/usr/bin/env python3
"""
LISA-Llama4 シングルプロセス学習スクリプト（overfit_llama4_lisa_batch.py成功パターン移植）
103Bモデルを1プロセスでModel Parallelism使用し、HybridDatasetで学習

実行方法:
# 最小時間テスト（10ステップ）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_llama4_lisa_single_process.py --exp_name quick_test --steps_per_epoch 10 --epochs 1

# 通常テスト
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_llama4_lisa_single_process.py --exp_name lisa_single --epochs 3
"""

import argparse
import os
import sys
import time
import json
import logging
import gc
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available. Using standard AdamW optimizer.")
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話モードに設定
import matplotlib.pyplot as plt
import numpy as np

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

def create_loss_plots(loss_history, log_dir, exp_name):
    """学習曲線の可視化（シングルエンコーダー構成対応）"""
    if not loss_history:
        return
    
    # 全エポックのデータを整理
    epochs = list(range(1, len(loss_history) + 1))
    total_losses = [epoch_data['total_loss'] for epoch_data in loss_history]
    lm_losses = [epoch_data['lm_loss'] for epoch_data in loss_history]
    seg_losses = [epoch_data['seg_loss'] for epoch_data in loss_history]
    dice_losses = [epoch_data['dice_loss'] for epoch_data in loss_history]
    bce_losses = [epoch_data['bce_loss'] for epoch_data in loss_history]
    
    # 統合チャート（全損失を1つのグラフに）- メイン目的
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, total_losses, 'b-', linewidth=2, marker='o', label='総損失')
    plt.plot(epochs, lm_losses, 'r-', linewidth=2, marker='s', label='LM損失')
    plt.plot(epochs, seg_losses, 'orange', linewidth=2, marker='v', label='Seg損失')
    plt.plot(epochs, dice_losses, 'g-', linewidth=2, marker='^', label='DICE損失')
    plt.plot(epochs, bce_losses, 'm-', linewidth=2, marker='d', label='BCE損失')
    
    plt.title(f'LISA-Llama4 学習曲線検証 - {exp_name}', fontsize=14)
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    plot_file = os.path.join(log_dir, f'loss_curves_{exp_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def setup_logging(log_dir: str):
    """ロギング設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train.log'))
        ]
    )
    return logging.getLogger(__name__)

def setup_environment():
    """環境変数の設定（overfit成功パターン）"""
    # GPU表示順序を固定
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # メモリ最適化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
    
    # HuggingFaceキャッシュ設定
    setup_hf_cache_dirs()

def setup_hf_cache_dirs():
    """HuggingFaceキャッシュディレクトリを自動設定・作成"""
    print("=== HuggingFaceキャッシュディレクトリ設定 ===")
    
    # 候補ディレクトリ（優先順位順）
    cache_candidates = [
        "/lambda/nfs/lisa-gemma-project-fs/cache/huggingface",
        os.path.expanduser("~/.cache/huggingface"),
        "/tmp/huggingface_cache"
    ]
    
    # 書き込み可能なディレクトリを探す
    hf_cache_dir = None
    for candidate in cache_candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            os.makedirs(os.path.join(candidate, "models"), exist_ok=True)
            os.makedirs(os.path.join(candidate, "datasets"), exist_ok=True)
            
            # 書き込みテスト
            test_file = os.path.join(candidate, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            hf_cache_dir = candidate
            break
        except (OSError, PermissionError):
            continue
    
    # 環境変数に設定
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
        os.environ["TOKENIZERS_CACHE"] = os.path.join(hf_cache_dir, "tokenizers")
        print(f"✅ HuggingFaceキャッシュ設定: {hf_cache_dir}")
    else:
        print("⚠️ キャッシュディレクトリ設定に失敗")
    
    print("=== キャッシュディレクトリ設定完了 ===")

def create_model_and_tokenizer(logger):
    """モデルとトークナイザーの作成（overfit成功パターン完全移植）"""
    logger.info("=== LISA-Llama4統合モデル初期化 ===")
    
    try:
        # 動的コンパイルを無効化してGPU分散エラーを回避（成功した検証スクリプトと同じ設定）
        torch.compiler.disable()
        logger.info("動的コンパイル無効化: GPU分散エラー回避のため")
        
        # LISA統合モデル設定（config_linux統一設定を使用）
        lisa_config = LisaLlama4Config(**config_linux.get_lisa_model_config())
        
        # LISA統合モデル初期化（成功したoverfit手法）
        model = LisaLlama4ForCausalLM(lisa_config)
        
        logger.info(f"✓ LISA統合モデル初期化完了")
        logger.info(f"  - 総パラメータ: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        logger.error(f"LISA統合モデル初期化エラー: {e}")
        raise

def apply_lora_config(model, args, logger):
    """Web調査結果に基づくLoRA設定適用（device_map preservation対応）完全移植版"""
    logger.info("=== LoRA設定適用 ===")
    
    try:
        # PEFT適用前にdevice_mapを保存（Web調査：既知の問題対策）
        original_device_map = None
        original_device_map_location = None
        
        # device_mapの場所を特定して保存（成功パターン完全移植）
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            original_device_map = model.hf_device_map.copy()
            original_device_map_location = "direct"
            logger.info(f"✓ 元のdevice_map保存（直接アクセス）: {len(original_device_map)} エントリ")
        elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'hf_device_map') and model.llama_model.hf_device_map:
            original_device_map = model.llama_model.hf_device_map.copy()
            original_device_map_location = "llama_model"
            logger.info(f"✓ 元のdevice_map保存（llama_model経由）: {len(original_device_map)} エントリ")
        else:
            logger.warning("⚠️ device_mapが見つかりません。Model Parallelismが未設定の可能性があります。")
        
        # LoRA設定作成（config_linux統一設定を使用）
        lora_config_dict = config_linux.get_lora_config()
        # task_typeを文字列から除外（後でenumとして設定）
        lora_config_dict.pop('task_type', None)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **lora_config_dict
        )
        
        # LoRA適用
        model = get_peft_model(model, lora_config)
        
        # Gradient Checkpointing有効化（メモリ30%削減）
        if args.gradient_checkpointing:
            logging.info("✅ Gradient Checkpointing有効化試行（メモリ30%削減）")
            try:
                # カスタムモデルの場合、内部のllama_modelで有効化を試行
                if hasattr(model, 'llama_model') and hasattr(model.llama_model, 'gradient_checkpointing_enable'):
                    model.llama_model.gradient_checkpointing_enable()
                    logging.info("✅ llama_modelでGradient Checkpointing有効化成功")
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model'):
                    if hasattr(model.base_model.llama_model, 'gradient_checkpointing_enable'):
                        model.base_model.llama_model.gradient_checkpointing_enable()
                        logging.info("✅ base_model.llama_modelでGradient Checkpointing有効化成功")
                    else:
                        logging.warning("⚠️ base_model.llama_modelはGradient Checkpointingをサポートしていません")
                elif hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logging.info("✅ 直接Gradient Checkpointing有効化成功")
                else:
                    logging.warning("⚠️ Gradient Checkpointingがサポートされていません - スキップ")
            except Exception as e:
                logging.warning(f"⚠️ Gradient Checkpointing有効化失敗（スキップ）: {e}")
        
        # device_mapの復元試行（Web調査：PEFT既知問題の対策）成功パターン完全移植
        if original_device_map and original_device_map_location:
            # 複数のアクセス方法を試行
            restoration_success = False
            
            # 方法1: 直接アクセス確認
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                logger.info("✓ 直接device_mapアクセス確認済み")
                restoration_success = True
            
            # 方法2: base_model経由のアクセス
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
                logger.info("✓ base_model経由device_mapアクセス確認済み")
                restoration_success = True
            
            # 方法3: llama_model経由のアクセス（LISA統合モデル特有）
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
                logger.info("✓ base_model.llama_model経由device_mapアクセス確認済み")
                restoration_success = True
            
            # 方法4: 手動復元
            if not restoration_success:
                logger.warning("⚠️ device_mapが失われました。手動復元を試行...")
                
                # 元の場所に基づいて復元
                if original_device_map_location == "llama_model":
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model'):
                        model.base_model.llama_model.hf_device_map = original_device_map
                        logger.info("✓ base_model.llama_model.hf_device_mapを手動復元")
                        restoration_success = True
                elif original_device_map_location == "direct":
                    if hasattr(model, 'base_model'):
                        model.base_model.hf_device_map = original_device_map
                        logger.info("✓ base_model.hf_device_mapを手動復元")
                        restoration_success = True
            
            if not restoration_success:
                logger.error("❌ device_mapの復元に失敗しました")
        
        # 学習可能パラメータ統計
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"✓ LoRA適用完了")
        logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
        logger.info(f"  - 全パラメータ: {total_params:,}")
        logger.info(f"  - 学習可能割合: {100 * trainable_params / total_params:.3f}%")
        
        # config_linux.ADDITIONAL_TRAINABLE_PARAMSで指定されたパラメータを学習可能に
        logger.info("=== 追加学習可能パラメータの凍結解除（config_linux設定） ===")
        unfrozen_count = 0
        additional_params = config_linux.ADDITIONAL_TRAINABLE_PARAMS
        for name, param in model.named_parameters():
            # config_linuxで指定された追加学習対象
            if any([x in name for x in additional_params]):
                if not param.requires_grad:
                    # 浮動小数点型のテンソルのみrequires_gradを設定可能
                    if param.dtype.is_floating_point or param.dtype.is_complex:
                        param.requires_grad = True
                        unfrozen_count += 1
                        logger.info(f"  ✅ {name} を学習可能に設定 (shape: {param.shape})")
                    else:
                        logger.info(f"  ⚠️ {name} はdtype={param.dtype}のためスキップ")
        
        logger.info(f"✓ 追加で{unfrozen_count}個のパラメータを学習可能に設定")
        
        # 最終的な学習可能パラメータ数を再計算
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        final_total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ 最終学習可能パラメータ: {final_trainable_params:,}")
        logger.info(f"✓ 最終学習可能割合: {100 * final_trainable_params / final_total_params:.3f}%")
        
        # 最終device_map確認
        verify_device_map_after_lora(model, logger)
        
        return model
        
    except Exception as e:
        logger.error(f"LoRA適用エラー: {e}")
        raise

def create_dataset_and_dataloader(model, args, logger):
    """データセットとデータローダー作成（段階的検証対応）"""
    logger.info("=== データセット初期化 ===")
    
    # データセット種類の解析（既存のHybridDatasetインターフェースに合わせて変換）
    if hasattr(args, 'dataset') and args.dataset:
        datasets = [ds.strip() for ds in args.dataset.split(',')]
        # HybridDatasetの期待する形式（"||"区切り）に変換
        dataset_string = "||".join(datasets)
    else:
        dataset_string = "reason_seg"  # デフォルト
    
    logger.info(f"対象データセット: {datasets if 'datasets' in locals() else [dataset_string]}")
    
    # 各データセットのサンプル数制御
    samples_per_dataset = getattr(args, 'samples_per_dataset', None)
    if samples_per_dataset:
        logger.info(f"サンプル数制限: {samples_per_dataset}")
        # samples_per_datasetからsamples_per_epochを計算
        # データセット数 × サンプル数で概算
        dataset_count = len(datasets) if 'datasets' in locals() else 1
        samples_per_epoch = samples_per_dataset * dataset_count
    else:
        # デフォルトは小さめに設定（段階的検証用）
        samples_per_epoch = 500  # 既存のデフォルトから大幅削減
    
    logger.info(f"エポックあたりのサンプル数: {samples_per_epoch}")
    
    # プロセッサを取得（LoRA適用後のモデル対応）
    if hasattr(model, 'base_model'):
        llama_processor = model.base_model.model.llama_processor if hasattr(model.base_model, 'model') else model.base_model.llama_processor
    else:
        llama_processor = model.llama_processor
    
    # HybridDatasetの初期化（verify_dataset_integrity.pyと同じパラメータ構成）
    dataset = HybridDataset(
        base_image_dir=config_linux.DATASET_BASE_DIR,
        llama_processor=llama_processor,
        samples_per_epoch=samples_per_epoch,
        precision="bf16",
        llama_image_size=config_linux.LLAMA_IMAGE_SIZE,
        sam_image_size=config_linux.SAM_IMAGE_SIZE,
        num_classes_per_sample=3,
        exclude_val=False,
        dataset=dataset_string,
        sample_rate=[9, 3, 3, 1],  # reason_seg, refer_seg, vqa, sem_seg の比率
        sem_seg_data=config_linux.SEM_SEG_DATA,
        refer_seg_data=config_linux.REFER_SEG_DATA,
        vqa_data=config_linux.VQA_DATA,
        reason_seg_data=config_linux.REASON_SEG_DATA,
        explanatory=0.1
    )
    
    logger.info(f"✓ データセット初期化完了: {len(dataset)} サンプル")
    
    # データセット構成の詳細表示
    if hasattr(dataset, 'datasets'):
        logger.info(f"データセット構成: {dataset.datasets}")
    
    # DataLoader作成
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"✓ DataLoader作成完了: バッチサイズ={args.batch_size}, ワーカー数={args.workers}")
    
    return dataset, dataloader

def verify_device_map_after_lora(model, logger):
    """LoRA適用後のdevice_map確認（LISA統合モデル対応）完全移植版"""
    logger.info("=== LoRA適用後device_map確認 ===")
    
    device_map = None
    access_path = None
    
    # 複数のアクセス方法を試行（成功パターン完全移植）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        access_path = "直接アクセス"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
        device_map = model.base_model.hf_device_map
        access_path = "base_model経由"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
        device_map = model.base_model.llama_model.hf_device_map
        access_path = "base_model.llama_model経由"
    
    if device_map:
        logger.info(f"✓ {access_path}でhf_device_mapアクセス成功")
        logger.info(f"  - デバイスマップ: {dict(list(device_map.items())[:5])}...")
        logger.info(f"  - 使用GPU数: {len(set(device_map.values()))}")
        return True
    else:
        logger.error("❌ device_mapが見つかりません")
        logger.error("Model Parallelismが設定されていません。103Bモデルには必須です。")
        return False

def get_model_device(model):
    """モデルの適切なデバイスを取得（Model Parallelism対応）完全移植版"""
    # Model Parallelismのデバイスマップを探す（LISA統合モデル対応）
    device_map = None
    access_path = None
    
    # 複数のアクセス方法を試行（成功パターン完全移植）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        access_path = "直接アクセス"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
        device_map = model.base_model.hf_device_map
        access_path = "base_model経由"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
        device_map = model.base_model.llama_model.hf_device_map
        access_path = "base_model.llama_model経由"
    
    if device_map:
        # Model Parallelismの場合、最初のデバイスを返す
        return next(iter(device_map.values()))
    else:
        raise RuntimeError("Model Parallelismが設定されていません。103Bモデルには必須です。")

def train_epoch(model, dataloader, optimizer, scheduler, epoch, args, logger, writer=None):
    """1エポックの学習実行（個別損失トラッキング付き）"""
    model.train()
    
    # 個別損失メトリクス初期化
    total_losses = AverageMeter('Total', ':.4e')
    lm_losses = AverageMeter('LM', ':.4e')
    seg_losses = AverageMeter('Seg', ':.4e')
    dice_losses = AverageMeter('DICE', ':.4e')
    bce_losses = AverageMeter('BCE', ':.4e')
    
    progress = ProgressMeter(
        len(dataloader) if args.steps_per_epoch is None else args.steps_per_epoch,
        [total_losses],  # 表示は総損失のみでシンプルに
        prefix=f"Epoch: [{epoch}]"
    )
    
    # 個別損失履歴（エポック内）
    epoch_loss_history = {
        'total_loss': [],
        'lm_loss': [],
        'seg_loss': [],
        'dice_loss': [],
        'bce_loss': []
    }
    
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # ステップ数制限チェック
        if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
            break
        
        # データを適切なデバイスに移動（Model Parallelism対応）完全移植版
        # Model Parallelismのdevice_mapを厳密にチェック（LISA統合モデル対応）
        device_map = None
        access_path = None
        
        # 複数のアクセス方法を試行（成功パターン完全移植）
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device_map = model.hf_device_map
            access_path = "直接アクセス"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
            device_map = model.base_model.hf_device_map
            access_path = "base_model経由"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
            device_map = model.base_model.llama_model.hf_device_map
            access_path = "base_model.llama_model経由"
        
        if not device_map:
            raise RuntimeError("Model Parallelismが設定されていません。103Bモデルには必須です。")
        
        if step == 0:
            logger.info(f"Model Parallelismデバイスマップ確認: {access_path}")
        
        first_device = next(iter(device_map.values()))
        batch = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        # フォワードパス（overfit成功手法）
        optimizer.zero_grad()
        
        # バッチ内容をデバッグ出力
        if step == 0:
            logger.info(f"バッチ内容: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        # 実際のLISA統合モデル使用：完全なフォワードパス（シングルエンコーダー構成）
        # HybridDatasetの実際の出力形式に合わせて修正
        model_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch.get('attention_mask'),
            # シングルエンコーダー構成: pixel_valuesを削除、sam_pixel_valuesのみ使用
            'sam_pixel_values': batch.get('sam_pixel_values'),     # SAM画像入力
            'labels': batch.get('labels'),                         # 実際のラベル使用
            'ground_truth_masks': batch.get('ground_truth_mask'),  # マスク損失計算用（HybridDatasetは単数形）
            'seg_token_mask': batch.get('seg_token_mask'),         # SEGトークン位置
            'original_sizes': batch.get('original_sizes'),         # 元画像サイズ
            'generate_mask': True  # SAM機能を有効化
        }
        
        # Noneの値を除去（引数重複エラー対策）
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        model_outputs = model(**model_inputs)
        
        # 個別損失を取得（CompositeLoss統合対応）
        individual_losses = {'total_loss': 0, 'lm_loss': 0, 'seg_loss': 0, 'dice_loss': 0, 'bce_loss': 0}
        
        # デバッグ: モデル出力のキーを確認（初回のみ）
        if step == 0:
            if isinstance(model_outputs, dict):
                logger.info(f"  🔍 モデル出力キー: {list(model_outputs.keys())}")
            else:
                logger.info(f"  🔍 モデル出力タイプ: {type(model_outputs)}")
        
        if isinstance(model_outputs, dict):
            # CompositeLossからの総損失を取得（フォールバック処理なし）
            
            # 1. 直接total_lossを確認
            if 'total_loss' in model_outputs:
                loss = model_outputs['total_loss']
                individual_losses['total_loss'] = loss.item()
                if step == 0:
                    logger.info(f"  ✓ total_loss取得成功: {individual_losses['total_loss']:.4f}")
            
            # 2. losses辞書内のtotal_lossを確認
            elif 'losses' in model_outputs and isinstance(model_outputs['losses'], dict):
                losses_dict = model_outputs['losses']
                if 'total_loss' in losses_dict:
                    loss = losses_dict['total_loss']
                    individual_losses['total_loss'] = loss.item()
                    if step == 0:
                        logger.info(f"  ✓ losses辞書内total_loss取得成功: {individual_losses['total_loss']:.4f}")
                else:
                    raise ValueError(f"losses辞書にtotal_lossが見つかりません。利用可能なキー: {list(losses_dict.keys())}")
            
            # 3. エラー：total_lossが見つからない
            else:
                available_keys = list(model_outputs.keys())
                raise ValueError(f"total_lossが見つかりません。モデル出力キー: {available_keys}")
            
            # 個別損失の確認も同様に厳密化
            
            # 個別損失の詳細を取得（厳密チェック）
            if 'losses' in model_outputs and isinstance(model_outputs['losses'], dict):
                losses_dict = model_outputs['losses']
                if step == 0:
                    logger.info(f"  🔍 losses辞書キー: {list(losses_dict.keys())}")
                
                # lm_loss（言語モデリング損失）
                if 'lm_loss' in losses_dict and losses_dict['lm_loss'] is not None:
                    individual_losses['lm_loss'] = losses_dict['lm_loss'].item()
                    if step == 0:
                        logger.info(f"  ✓ lm_loss取得成功: {individual_losses['lm_loss']:.4f}")
                else:
                    # lm_lossがない場合は0とする（VQAタスクなどの場合）
                    individual_losses['lm_loss'] = 0.0
                    if step == 0:
                        logger.info(f"  ℹ️ lm_loss: N/A (セグメンテーションタスクなし)")
                
                # seg_loss（セグメンテーション損失）
                if 'seg_loss' in losses_dict and losses_dict['seg_loss'] is not None:
                    individual_losses['seg_loss'] = losses_dict['seg_loss'].item()
                    if step == 0:
                        logger.info(f"  ✓ seg_loss取得成功: {individual_losses['seg_loss']:.4f}")
                else:
                    # seg_lossがない場合は0とする
                    individual_losses['seg_loss'] = 0.0
                    if step == 0:
                        logger.info(f"  ℹ️ seg_loss: N/A")
                
                # dice_loss
                if 'dice_loss' in losses_dict and losses_dict['dice_loss'] is not None:
                    individual_losses['dice_loss'] = losses_dict['dice_loss'].item()
                    if step == 0:
                        logger.info(f"  ✓ dice_loss取得成功: {individual_losses['dice_loss']:.4f}")
                else:
                    individual_losses['dice_loss'] = 0.0
                    if step == 0:
                        logger.info(f"  ℹ️ dice_loss: N/A")
                
                # bce_loss
                if 'bce_loss' in losses_dict and losses_dict['bce_loss'] is not None:
                    individual_losses['bce_loss'] = losses_dict['bce_loss'].item()
                    if step == 0:
                        logger.info(f"  ✓ bce_loss取得成功: {individual_losses['bce_loss']:.4f}")
                else:
                    individual_losses['bce_loss'] = 0.0
                    if step == 0:
                        logger.info(f"  ℹ️ bce_loss: N/A")
            else:
                logger.warning("losses辞書が見つかりません")
        else:
            # 非辞書型出力の場合
            if hasattr(model_outputs, 'loss'):
                loss = model_outputs.loss
                individual_losses['total_loss'] = loss.item()
            else:
                raise ValueError(f"モデル出力から損失が見つかりません。タイプ: {type(model_outputs)}")
        
        # メモリクリアしてから逆伝播（CUBLAS_STATUS_ALLOC_FAILED対策）
        torch.cuda.empty_cache()
        
        # 逆伝播
        try:
            loss.backward()
        except RuntimeError as e:
            if "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                logger.error("❌ CUDA/CUBLAS メモリ不足エラー")
                logger.error("💡 対策:")
                logger.error("  1. --steps_per_epoch を小さくする（例: --steps_per_epoch 10）")
                logger.error("  2. gradient_accumulation_steps の使用を検討")
                logger.error("  3. より小さなデータセットで検証")
                raise
            else:
                raise
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        
        # パラメータ更新
        optimizer.step()
        scheduler.step()
        
        # 個別メトリクス更新
        batch_size = batch['input_ids'].size(0)
        total_losses.update(individual_losses['total_loss'], batch_size)
        lm_losses.update(individual_losses['lm_loss'], batch_size)
        seg_losses.update(individual_losses['seg_loss'], batch_size)
        dice_losses.update(individual_losses['dice_loss'], batch_size)
        bce_losses.update(individual_losses['bce_loss'], batch_size)
        
        # エポック内損失履歴に追加
        epoch_loss_history['total_loss'].append(individual_losses['total_loss'])
        epoch_loss_history['lm_loss'].append(individual_losses['lm_loss'])
        epoch_loss_history['seg_loss'].append(individual_losses['seg_loss'])
        epoch_loss_history['dice_loss'].append(individual_losses['dice_loss'])
        epoch_loss_history['bce_loss'].append(individual_losses['bce_loss'])
        
        # 学習進行度とログ出力
        if step % args.print_freq == 0:
            # 一般的な進行度表示
            total_steps = args.steps_per_epoch if args.steps_per_epoch else len(dataloader)
            progress_pct = (step + 1) / total_steps * 100
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (step + 1) * (total_steps - step - 1) if step > 0 else 0
            
            logger.info(f"🚀 Epoch [{epoch+1}] Step [{step+1:3d}/{total_steps}] ({progress_pct:5.1f}%) "
                       f"ETA: {eta/60:.1f}min | "
                       f"Total: {individual_losses['total_loss']:6.3f} | "
                       f"LM: {individual_losses['lm_loss']:6.3f} | "
                       f"Seg: {individual_losses['seg_loss']:6.3f} | "
                       f"DICE: {individual_losses['dice_loss']:6.3f} | "
                       f"BCE: {individual_losses['bce_loss']:6.3f}")
            
            # TensorBoard記録
            if writer is not None:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('Train/TotalLoss', total_losses.val, global_step)
                writer.add_scalar('Train/LMLoss', lm_losses.val, global_step)
                writer.add_scalar('Train/SegLoss', seg_losses.val, global_step)
                writer.add_scalar('Train/DiceLoss', dice_losses.val, global_step)
                writer.add_scalar('Train/BCELoss', bce_losses.val, global_step)
                writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
        
        # メモリクリーンアップ
        del model_outputs, loss
        torch.cuda.empty_cache()
        gc.collect()
    
    epoch_time = time.time() - start_time
    logger.info(f"✓ エポック {epoch} 完了:")
    logger.info(f"  - 総損失: {total_losses.avg:.4f}")
    logger.info(f"  - LM損失: {lm_losses.avg:.4f}")
    logger.info(f"  - Seg損失: {seg_losses.avg:.4f}")
    logger.info(f"  - DICE損失: {dice_losses.avg:.4f}")
    logger.info(f"  - BCE損失: {bce_losses.avg:.4f}")
    logger.info(f"  - 時間: {epoch_time:.1f}秒")
    
    return {
        'total_loss': total_losses.avg,
        'lm_loss': lm_losses.avg,
        'seg_loss': seg_losses.avg,
        'dice_loss': dice_losses.avg,
        'bce_loss': bce_losses.avg,
        'epoch_loss_history': epoch_loss_history
    }

def main():
    parser = argparse.ArgumentParser(description="LISA-Llama4 シングルプロセス学習（overfit成功パターン移植）")
    
    # 基本設定
    parser.add_argument("--exp_name", type=str, required=True, help="実験名")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    
    # メモリ最適化オプション（新規追加）
    parser.add_argument("--use_8bit_adam", action="store_true", default=True,
                        help="8bit AdamWオプティマイザを使用（メモリ25%%削減）")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Gradient Checkpointingを有効化（メモリ30%%削減）")
    parser.add_argument("--lr", type=float, default=config_linux.LEARNING_RATE, help="学習率（edit_config.md推奨: 2e-4）")
    parser.add_argument("--clip_grad_norm", type=float, default=config_linux.GRADIENT_CLIP_NORM, help="勾配クリッピング（edit_config.md推奨: 1.0）")
    parser.add_argument("--workers", type=int, default=config_linux.DATALOADER_NUM_WORKERS, help="データローダーワーカー数")
    parser.add_argument("--print_freq", type=int, default=10, help="ログ出力頻度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config_linux.GRADIENT_ACCUMULATION_STEPS,
                        help="勾配累積ステップ数（edit_config.md推奨: 16で実効バッチサイズ64-128）")
    parser.add_argument("--weight_decay", type=float, default=config_linux.WEIGHT_DECAY, help="重み減衰（edit_config.md推奨: 0.05）")
    parser.add_argument("--save_freq", type=int, default=1, help="チェックポイント保存頻度")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="エポックあたりのステップ数（制限）")
    
    # 段階的検証用追加設定
    parser.add_argument("--dataset", type=str, default="reason_seg", 
                       help="データセット種類（カンマ区切り）: reason_seg,refer_seg,vqa,sem_seg")
    parser.add_argument("--samples_per_dataset", type=int, default=None,
                       help="各データセットのサンプル数制限")
    parser.add_argument("--visualize_losses", action="store_true", default=True,
                       help="学習曲線可視化を有効化")
    parser.add_argument("--plot_interval", type=int, default=1,
                       help="学習曲線プロット更新間隔（エポック）")
    
    args = parser.parse_args()
    
    # 環境設定
    setup_environment()
    
    # ログディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # ロギング設定
    logger = setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("LISA-Llama4 シングルプロセス学習開始（overfit成功パターン移植）")
    logger.info("=" * 80)
    logger.info(f"実験名: {args.exp_name}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"学習率: {args.lr}")
    if args.steps_per_epoch:
        logger.info(f"ステップ制限: {args.steps_per_epoch}/エポック")
    
    try:
        # HuggingFace認証確認
        ensure_hf_login()
        
        # モデル初期化
        model = create_model_and_tokenizer(logger)
        model = apply_lora_config(model, args, logger)
        
        # データセットとデータローダー
        dataset, dataloader = create_dataset_and_dataloader(model, args, logger)
        
        # オプティマイザーとスケジューラー
        # 学習可能パラメータのみ対象
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        # Model Parallelism環境では8bit Adamは使用不可（GPU間でパラメータが分散するため）
        is_model_parallel = hasattr(model, 'hf_device_map') or \
                           (hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map')) or \
                           (hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and 
                            hasattr(model.base_model.llama_model, 'hf_device_map'))
        
        # 8bit optimizer使用（メモリ効率化）
        if BITSANDBYTES_AVAILABLE and args.use_8bit_adam and not is_model_parallel:
            logging.info("✅ 8bit AdamWオプティマイザーを使用（メモリ25%削減）")
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999)
            )
        else:
            if args.use_8bit_adam and is_model_parallel:
                logging.info("📌 Model Parallelism検出: 標準AdamWを使用（8bit Adamは非互換）")
            elif args.use_8bit_adam:
                logging.warning("⚠️ bitsandbytesが利用できません。標準AdamWを使用します。")
            optimizer = optim.AdamW(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999)
            )
        
        total_steps = len(dataloader) * args.epochs
        if args.steps_per_epoch:
            total_steps = args.steps_per_epoch * args.epochs
            
        # 学習率スケジューラー（config_linux設定使用）
        training_config = config_linux.get_training_config()
        warmup_steps = int(training_config["warmup_ratio"] * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logging.info(f"✅ {training_config['lr_scheduler_type']}スケジューラー設定（warmup: {warmup_steps}ステップ = 全体の{training_config['warmup_ratio']*100:.0f}%）")
        
        
        logger.info(f"✓ オプティマイザー設定完了: 学習可能パラメータ={len(trainable_params):,}")
        
        # TensorBoard
        writer = SummaryWriter(f"./runs/{args.exp_name}_{timestamp}")
        
        # 学習履歴記録用
        loss_history = []
        
        # 学習ループ
        for epoch in range(args.epochs):
            epoch_results = train_epoch(
                model, dataloader, optimizer, scheduler, 
                epoch, args, logger, writer
            )
            
            # 学習履歴に追加
            loss_history.append(epoch_results)
            
            # 学習曲線の可視化（指定間隔で実行）
            if args.visualize_losses and (epoch + 1) % args.plot_interval == 0:
                logger.info(f"学習曲線を更新中... (エポック {epoch + 1})")
                try:
                    plot_file = create_loss_plots(loss_history, log_dir, args.exp_name)
                    if plot_file:
                        logger.info(f"✓ 学習曲線を保存: {plot_file}")
                except Exception as e:
                    logger.warning(f"学習曲線の生成に失敗: {e}")
            
            # 損失改善状況の報告
            if len(loss_history) >= 2:
                prev_total = loss_history[-2]['total_loss']
                curr_total = loss_history[-1]['total_loss']
                improvement = prev_total - curr_total
                improvement_pct = (improvement / prev_total) * 100 if prev_total > 0 else 0
                
                logger.info(f"📊 損失改善状況:")
                change = curr_total - prev_total
                change_pct = (change / prev_total) * 100 if prev_total > 0 else 0
                logger.info(f"  - 総損失: {prev_total:.4f} → {curr_total:.4f} ({change:+.4f}, {change_pct:+.1f}%)")
                
                # 個別損失の改善も報告
                for loss_type in ['text_loss', 'dice_loss', 'bce_loss']:
                    if loss_type in loss_history[-1] and loss_type in loss_history[-2]:
                        prev_val = loss_history[-2][loss_type]
                        curr_val = loss_history[-1][loss_type]
                        if prev_val > 0:
                            change = curr_val - prev_val
                            change_pct = (change / prev_val) * 100
                            logger.info(f"  - {loss_type}: {prev_val:.4f} → {curr_val:.4f} ({change:+.4f}, {change_pct:+.1f}%)")
            
            # 最終可視化
            if epoch == args.epochs - 1:
                logger.info("最終学習曲線を生成中...")
                try:
                    plot_file = create_loss_plots(loss_history, log_dir, args.exp_name)
                    if plot_file:
                        logger.info(f"✓ 最終学習曲線を保存: {plot_file}")
                except Exception as e:
                    logger.warning(f"最終学習曲線の生成に失敗: {e}")
            
            # チェックポイント保存
            if epoch % args.save_freq == 0:
                checkpoint_dir = f"./checkpoints/{args.exp_name}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Model Parallelism対応でチェックポイント保存
                device_map = None
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    device_map = model.hf_device_map
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
                    device_map = model.base_model.hf_device_map
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
                    device_map = model.base_model.llama_model.hf_device_map
                
                if not device_map:
                    logger.warning("⚠️ Model Parallelismのdevice_mapが見つかりません。チェックポイント保存をスキップします。")
                    logger.info("✓ 学習完了（チェックポイント保存なし）")
                    continue
                
                # base_modelを通してアクセス
                if hasattr(model, 'base_model'):
                    model_state_dict = model.base_model.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'current_epoch_loss': epoch_results,
                }, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
                
                logger.info(f"✓ チェックポイント保存: epoch_{epoch}.pt")
        
        # 最終学習レポート
        logger.info("=" * 80)
        logger.info("✅ LISA-Llama4 段階的検証学習完了!")
        logger.info("=" * 80)
        
        # 学習結果サマリー
        if loss_history:
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            
            logger.info("📊 学習結果サマリー:")
            logger.info(f"  対象データセット: {args.dataset}")
            if args.samples_per_dataset:
                logger.info(f"  サンプル数制限: {args.samples_per_dataset}/データセット")
            logger.info(f"  総エポック数: {args.epochs}")
            
            logger.info("📈 損失変化:")
            for loss_type in ['total_loss', 'text_loss', 'dice_loss', 'bce_loss']:
                if loss_type in initial_loss and loss_type in final_loss:
                    init_val = initial_loss[loss_type]
                    final_val = final_loss[loss_type]
                    change = final_val - init_val
                    change_pct = (change / init_val) * 100 if init_val > 0 else 0
                    trend = "📉" if change < 0 else "📈" if change > 0 else "➡️"
                    logger.info(f"  - {loss_type}: {init_val:.4f} → {final_val:.4f} {trend} ({change:+.4f}, {change_pct:+.1f}%)")
            
            # 学習成功判定
            total_improvement = initial_loss['total_loss'] - final_loss['total_loss']
            improvement_pct = (total_improvement / initial_loss['total_loss']) * 100 if initial_loss['total_loss'] > 0 else 0
            
            if improvement_pct > 10:
                logger.info("🎉 優秀な学習結果 - 総損失が10%以上改善!")
            elif improvement_pct > 5:
                logger.info("✅ 良好な学習結果 - 総損失が5%以上改善")
            elif improvement_pct > 0:
                logger.info("📊 学習進行中 - 総損失が改善傾向")
            else:
                logger.info("⚠️ 学習要調整 - 損失改善が見られない可能性")
        
        logger.info("📁 出力ファイル:")
        logger.info(f"  - ログディレクトリ: {log_dir}")
        logger.info(f"  - チェックポイント: ./checkpoints/{args.exp_name}")
        logger.info(f"  - TensorBoard: tensorboard --logdir=./runs/{args.exp_name}_{timestamp}")
        if args.visualize_losses:
            logger.info(f"  - 学習曲線: {log_dir}/loss_curves_{args.exp_name}.png")
        
        logger.info("=" * 80)
        
        writer.close()
    
    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        raise

if __name__ == "__main__":
    main()