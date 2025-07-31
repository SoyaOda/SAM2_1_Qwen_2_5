#!/usr/bin/env python3
"""
Llama4-LISA DeepSpeed分散学習スクリプト
==================================

DeepSpeed ZeRO Stage 2 + Expert Parallelismによる効率的分散学習
- Llama4-Scout-17B-16E-Instruct + SAM統合
- A100×8環境でのMoE最適化
- LoRAファインチューニング対応
"""

import argparse
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, AutoTokenizer, AutoConfig,
    Llama4ForConditionalGeneration
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import psutil
import wandb

# DeepSpeed統合
import deepspeed
from deepspeed.utils import logger as ds_logger

# プロジェクトのパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 既存のutilsとmodelコンポーネントを活用
from model.segment_anything import sam_model_registry
from utils.dataset import HybridDataset, collate_fn
import config_linux

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama4LisaDeepSpeedConfig:
    """DeepSpeed用Llama4-LISA設定"""
    
    def __init__(self):
        # モデル設定
        self.llama4_model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.sam_model_type = "vit_h"
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        
        # LoRA設定（Gemma設定を継承）
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # SAM統合設定
        self.sam_projector_hidden_size = 5120  # Llama4の隠れ層サイズ
        self.sam_projector_output_size = 256   # SAMプロンプト埋め込み
        
        # DeepSpeed設定
        self.enable_expert_parallelism = True
        self.expert_parallel_size = 8  # A100×8
        self.enable_gradient_checkpointing = True
        
        # 特殊トークン
        self.seg_token = "[SEG]"

class SAMProjector(nn.Module):
    """SAMプロンプト生成用MLP射影層"""
    
    def __init__(self, config: Llama4LisaDeepSpeedConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.sam_projector_hidden_size, config.sam_projector_hidden_size),
            nn.GELU(),
            nn.Linear(config.sam_projector_hidden_size, config.sam_projector_output_size)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

class Llama4LisaDeepSpeedModel(nn.Module):
    """DeepSpeed分散学習専用Llama4-LISAモデル"""
    
    def __init__(self, config: Llama4LisaDeepSpeedConfig):
        super().__init__()
        self.config = config
        
        logger.info("=" * 60)
        logger.info("🚀 Llama4-LISA DeepSpeed統合モデル初期化")
        logger.info("  - ZeRO Stage 2 + Expert Parallelism")
        logger.info("  - LoRAファインチューニング対応")
        logger.info("=" * 60)
        
        # 1. Llama4モデルの初期化
        self._init_llama4_model()
        
        # 2. SAMモデルの初期化
        self._init_sam_model()
        
        # 3. SAMプロジェクターの初期化
        self.sam_projector = SAMProjector(config)
        
        # 4. LoRA設定の適用
        self._apply_lora_config()
        
        # 5. 学習可能パラメータの設定
        self._configure_trainable_parameters()
        
        # 6. Processorの初期化
        self._init_processor()
        
        logger.info("✅ Llama4-LISA DeepSpeedモデル初期化完了")
        
    def _init_llama4_model(self):
        """Llama4モデルの初期化"""
        logger.info("📥 Llama4-Scout-17B-16E-Instructモデル読み込み中...")
        
        try:
            # DeepSpeed分散学習用の設定
            self.llama4_model = Llama4ForConditionalGeneration.from_pretrained(
                self.config.llama4_model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Gradient Checkpointingの有効化
            if self.config.enable_gradient_checkpointing:
                self.llama4_model.gradient_checkpointing_enable()
                logger.info("✅ Gradient Checkpointing有効化")
            
            logger.info("✅ Llama4モデル読み込み完了")
            
        except Exception as e:
            logger.error(f"❌ Llama4モデル読み込みエラー: {e}")
            raise
    
    def _init_sam_model(self):
        """SAMモデルの初期化"""
        logger.info("📥 SAMモデル読み込み中...")
        
        try:
            # SAMモデルの読み込み（簡略化）
            self.sam_model = sam_model_registry[self.config.sam_model_type](
                checkpoint=None  # 自動ダウンロード
            )
            
            # SAMコンポーネントの凍結
            for param in self.sam_model.image_encoder.parameters():
                param.requires_grad = False
            for param in self.sam_model.prompt_encoder.parameters():
                param.requires_grad = False
            # mask_decoderは学習可能
            
            logger.info("✅ SAMモデル読み込み完了")
            
        except Exception as e:
            logger.error(f"❌ SAMモデル読み込みエラー: {e}")
            raise
    
    def _apply_lora_config(self):
        """LoRA設定の適用"""
        logger.info("🔧 LoRA設定適用中...")
        
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none"
            )
            
            self.llama4_model = get_peft_model(self.llama4_model, lora_config)
            logger.info("✅ LoRA設定適用完了")
            
        except Exception as e:
            logger.error(f"❌ LoRA適用エラー: {e}")
            raise
    
    def _configure_trainable_parameters(self):
        """学習可能パラメータの設定"""
        logger.info("⚙️ 学習可能パラメータ設定中...")
        
        # Llama4ベースモデルを凍結（LoRAアダプタのみ学習可能）
        for name, param in self.llama4_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # SAMプロジェクターを学習可能に設定
        for param in self.sam_projector.parameters():
            param.requires_grad = True
        
        # SAM Mask Decoderを学習可能に設定
        for param in self.sam_model.mask_decoder.parameters():
            param.requires_grad = True
        
        # パラメータ統計
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        
        logger.info("✅ 学習可能パラメータ設定完了")
        logger.info(f"  - 総パラメータ数: {total_params:,}")
        logger.info(f"  - 学習可能パラメータ数: {trainable_params:,}")
        logger.info(f"  - 学習可能率: {trainable_percentage:.2f}%")
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_percentage
        }
    
    def _init_processor(self):
        """Processorの初期化"""
        logger.info("🔧 Processor初期化中...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.llama4_model_id,
                trust_remote_code=True
            )
            
            # [SEG]トークンの追加
            if self.config.seg_token not in self.processor.tokenizer.get_vocab():
                self.processor.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.config.seg_token]
                })
                self.llama4_model.resize_token_embeddings(len(self.processor.tokenizer))
                logger.info("✅ [SEG]トークン追加完了")
            
            logger.info("✅ Processor初期化完了")
            
        except Exception as e:
            logger.error(f"❌ Processor初期化エラー: {e}")
            raise
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """前向き計算"""
        
        # Llama4モデルによる前向き計算
        outputs = self.llama4_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )
        
        return outputs

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="Llama4-LISA DeepSpeed分散学習")
    
    # 学習設定
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="エポックあたりステップ数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学習率")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    
    # データセット設定
    parser.add_argument("--dataset_type", type=str, default="hybrid", 
                       choices=["refer_seg", "vqa", "reason_seg", "hybrid"],
                       help="データセットタイプ")
    
    # DeepSpeed設定
    parser.add_argument("--ds_config", type=str, default="ds_config_llama4_moe.json",
                       help="DeepSpeed設定ファイル")
    
    # 出力設定
    parser.add_argument("--output_dir", type=str, default="./outputs/llama4_deepspeed",
                       help="出力ディレクトリ")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                       help="チェックポイント保存間隔")
    
    # WandB設定
    parser.add_argument("--use_wandb", action="store_true", help="WandB使用")
    parser.add_argument("--wandb_project", type=str, default="llama4-lisa-deepspeed",
                       help="WandBプロジェクト名")
    
    return parser.parse_args()

def create_dataloader(args, config, model, is_main_process=True):
    """データローダーの作成"""
    
    if is_main_process:
        logger.info("📊 データセット準備中...")
    
    # 既存のHybridDatasetを使用
    dataset = HybridDataset(
        dataset_type=args.dataset_type,
        tokenizer=model.processor.tokenizer,
        samples_per_epoch=args.steps_per_epoch,
        max_length=512,
        is_train=True
    )
    
    # DataLoaderの作成
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer=model.processor.tokenizer,
            conv_type="llava_v1",
            use_mm_start_end=True,
            local_rank=int(os.environ.get('LOCAL_RANK', 0))
        )
    )
    
    if is_main_process:
        logger.info(f"✅ データセット準備完了")
        logger.info(f"  - サンプル数: {len(dataset)}")
        logger.info(f"  - バッチサイズ: {args.batch_size}")
    
    return dataloader

def main():
    """メイン実行関数"""
    args = parse_args()
    
    # DeepSpeed分散環境の初期化
    deepspeed.init_distributed()
    
    # ランク情報の取得
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info("=" * 80)
        logger.info("🚀 Llama4-LISA DeepSpeed分散学習開始")
        logger.info(f"  - ワールドサイズ: {world_size}")
        logger.info(f"  - ローカルランク: {local_rank}")
        logger.info(f"  - DeepSpeed設定: {args.ds_config}")
        logger.info("=" * 80)
    
    # 出力ディレクトリの作成
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # WandB初期化
    if args.use_wandb and is_main_process:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"llama4_lisa_deepspeed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info("✅ WandB初期化完了")
    
    # モデル設定とモデルの作成
    model_config = Llama4LisaDeepSpeedConfig()
    model = Llama4LisaDeepSpeedModel(model_config)
    
    # パラメータ情報の表示
    if is_main_process:
        param_info = model._configure_trainable_parameters()
        logger.info("📊 モデル統計:")
        for key, value in param_info.items():
            if isinstance(value, float):
                logger.info(f"  - {key}: {value:.2f}%")
            else:
                logger.info(f"  - {key}: {value:,}")
    
    # データローダーの作成
    dataloader = create_dataloader(args, model_config, model, is_main_process)
    
    # DeepSpeed設定の確認
    if not os.path.exists(args.ds_config):
        raise FileNotFoundError(f"DeepSpeed設定ファイルが見つかりません: {args.ds_config}")
    
    # DeepSpeedでモデルとオプティマイザーを初期化
    if is_main_process:
        logger.info("🚀 DeepSpeed初期化中...")
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.ds_config,
        dist_init_required=False  # 既に初期化済み
    )
    
    if is_main_process:
        logger.info("✅ DeepSpeed初期化完了")
    
    # 学習ループ
    if is_main_process:
        logger.info("🎯 学習開始...")
    
    step = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        if is_main_process:
            logger.info(f"\n📈 エポック {epoch + 1}/{args.epochs}")
        
        model_engine.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            if step >= args.epochs * args.steps_per_epoch:
                break
            
            # バッチをGPUに転送
            batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向き計算
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            # 後向き計算
            model_engine.backward(loss)
            model_engine.step()
            
            # 統計更新
            epoch_loss += loss.item()
            step += 1
            
            # ログ出力
            if step % 10 == 0 and is_main_process:
                elapsed_time = time.time() - start_time
                logger.info(f"ステップ {step}: 損失={loss.item():.4f}, "
                           f"経過時間={elapsed_time:.1f}s")
            
            # WandBログ
            if args.use_wandb and is_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": step,
                    "train/epoch": epoch
                })
            
            # チェックポイント保存
            if step % args.checkpoint_interval == 0 and is_main_process:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_step_{step}")
                model_engine.save_checkpoint(checkpoint_dir)
                logger.info(f"💾 チェックポイント保存: {checkpoint_dir}")
        
        if is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"エポック {epoch + 1} 完了: 平均損失={avg_loss:.4f}")
    
    # 最終モデルの保存
    if is_main_process:
        final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
        model_engine.save_checkpoint(final_checkpoint_dir)
        logger.info(f"💾 最終モデル保存: {final_checkpoint_dir}")
        
        logger.info("🎉 学習完了!")
    
    # WandB終了
    if args.use_wandb and is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main() 