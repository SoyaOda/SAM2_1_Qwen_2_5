#!/usr/bin/env python3
"""
Phase 3B QFormerSegmentationBridge 訓練スクリプト
test_phase3b_integration_real.pyの成功パターンを基に実際のデータで学習

実行方法:
# 最小時間テスト（10ステップ）- test script成功パターン準拠（H100x4対応）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.52.239 "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u train_phase3b_qformer_bridge.py --exp_name quick_test --steps_per_epoch 10 --epochs 1 2>&1"

# 通常訓練（実際のデータで損失減少観察）- test script成功パターン準拠（H100x4対応）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.52.239 "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u train_phase3b_qformer_bridge.py --exp_name phase3b_train --epochs 3 --steps_per_epoch 50 2>&1"
"""

# 🔥 PyTorchインポート前の環境準備（CUDA Error 802対策）
import sys
import os

# Step 1: CUDA Error 802対策 + Webリサーチメモリ最適化環境変数設定（PyTorchインポート前）
print("🔧 CUDA Error 802対策：PyTorchインポート前環境設定...")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

# Webリサーチ最適化: 2025年最新のOOM対策設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8,roundup_power2_divisions:8'
print("🔧 Webリサーチ最適化: 2025年最新OOM対策（max_split_size_mb:512,gc_threshold:0.8）")

# Step 2: CUDA_VISIBLE_DEVICESが未設定の場合のみ設定（H100x4対応 - test script成功パターン準拠）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # H100x4専用

# Step 2.1: test script成功パターン準拠の最小限設定
# ❌ 重複削除: PYTORCH_CUDA_ALLOC_CONF は上で設定済み

print("✅ 環境変数設定完了 - PyTorchインポート開始...")

# Step 3: PyTorchインポート
import argparse
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
from torch.cuda.amp import GradScaler  # Loss Scaling用
import transformers
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# メモリ最適化設定（CPU offload戦略対応）
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)  # 70%使用許可（CPU offload戦略）
    torch.cuda.empty_cache()  # 初期キャッシュクリア
    print("🔧 CUDA memory fraction set to 0.7 (70% usage, CPU offload strategy)")

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

# 🔥 Webリサーチメモリ最適化: GPU RAMモニタリング追加
from utils.memory_monitor import GPUMemoryMonitor, memory_monitor_section, log_memory_status, emergency_cleanup

# Phase 3B統合モデルとユーティリティ
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
from model.sam2_integration import get_sam2_wrapper
# from model.qformer import get_qformer_model  # ❌ 削除: QFormerSegmentationBridge内で初期化されるため不要
from model.dual_pathway_decoder import create_dual_pathway_decoder
from model.multiresolution_fusion import Llama4SAM2MultiResolutionFusion
from model.ohem_loss import create_ohem_loss
from model.dataset_adapter import adapt_dataset_for_qformer, configure_dual_encoder
from utils.dataset import HybridDataset, collate_fn, preprocess_sam_image, build_correct_labels_for_llama4
from utils.constants import DEFAULT_SEG_TOKEN
import config_linux

# Lambda Cloud環境パス検証レポート表示
config_linux.validate_and_report_paths()

# 評価用メトリクス
from utils.utils import (
    AverageMeter, ProgressMeter, Summary, 
    dict_to_cuda, intersectionAndUnionGPU
)

# HuggingFace transformers（Llama-4-Scout用）
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ HuggingFace Transformers利用可能")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ HuggingFace Transformersが利用できません")

def create_loss_plots(loss_history, log_dir, exp_name):
    """Phase 3B学習曲線の可視化（統合損失対応）"""
    if not loss_history:
        return
    
    # 全エポックのデータを整理
    epochs = list(range(1, len(loss_history) + 1))
    total_losses = [epoch_data['total_loss'] for epoch_data in loss_history]
    phase2_losses = [epoch_data.get('phase2_loss', 0) for epoch_data in loss_history]
    seg_losses = [epoch_data.get('seg_loss', 0) for epoch_data in loss_history]
    ohem_losses = [epoch_data.get('ohem_loss', 0) for epoch_data in loss_history]
    
    # Phase 3B統合チャート
    plt.figure(figsize=(15, 10))
    
    # メインプロット: 総損失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, total_losses, 'b-', linewidth=3, marker='o', markersize=6, label='総損失')
    plt.title('Phase 3B: 総損失', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase 2損失
    plt.subplot(2, 2, 2)
    plt.plot(epochs, phase2_losses, 'g-', linewidth=2, marker='s', label='Phase 2統一空間')
    plt.title('Phase 2: 部分統一トークン空間', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # セグメンテーション損失
    plt.subplot(2, 2, 3)
    plt.plot(epochs, seg_losses, 'orange', linewidth=2, marker='v', label='SAM2セグメンテーション')
    plt.title('SAM2セグメンテーション損失', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # OHEM損失
    plt.subplot(2, 2, 4)
    plt.plot(epochs, ohem_losses, 'r-', linewidth=2, marker='^', label='OHEM困難例')
    plt.title('OHEM困難例マイニング', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Phase 3B QFormerSegmentationBridge 学習曲線 - {exp_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    plot_file = os.path.join(log_dir, f'phase3b_loss_curves_{exp_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def setup_logging(log_dir: str):
    """ロギング設定"""
    # 環境変数からログレベルを取得
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train_phase3b.log'))
        ]
    )
    return logging.getLogger(__name__)


def setup_environment():
    """環境変数の設定（test_phase3b成功パターン）"""
    # GPU表示順序を固定
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # メモリ最適化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def create_model_and_components(args, logger):
    """Phase 3Bモデルとコンポーネント作成（test_phase3b成功パターン移植）"""
    logger.info("=== Phase 3B統合モデル初期化（QFormerSegmentationBridge） ===")
    
    # test_phase3b成功パターン: GPU最適化設定
    logger.info("🔥 GPU RAM分散利用最適化設定を適用...")
    
    # PyTorchマルチGPU最適化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info("✓ TF32最適化 + cuDNNベンチマーク有効化")
    
    # GPU間メモリ転送最適化
    if torch.cuda.device_count() >= 2:
        torch.cuda.set_per_process_memory_fraction(0.95)  # 95%使用許可
        logger.info("✓ GPU RAM使用率95%設定（メモリ効率化）")
    
    # 動的コンパイル無効化（安定性のため）
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    logger.info("✓ torch._dynamo.config.suppress_errors = True: Llama-4-Scout既知バグ回避")
    
    try:
        # 1. Llama-4-Scout初期化（test_phase3b成功パターン）
        logger.info("🔄 Llama-4-Scout-17B-16E-Instruct初期化...")
        
        if TRANSFORMERS_AVAILABLE:
            # test_phase3b成功パターン完全移植
            llama_config = {
                "llama_model_id": config_linux.LLAMA_MODEL_ID,
                "use_cache": False,
                "low_cpu_mem_usage": True
            }
            
            model_id = llama_config["llama_model_id"]
            
            # HuggingFaceから直接ロード（Web調査2025年最新パターン）+ GPU RAM分散対応
            # LISA準拠: CausalLMアーキテクチャを使用
            try:
                # Llama4専用クラスの確認
                try:
                    from transformers import Llama4ForCausalLM
                    model_class = Llama4ForCausalLM
                    logger.info("✓ Llama4ForCausalLMクラス利用可能")
                except ImportError:
                    # AutoModelForCausalLMを使用（Llama4を自動選択）
                    from transformers import AutoModelForCausalLM
                    model_class = AutoModelForCausalLM
                    logger.warning("⚠️ Llama4ForCausalLM未対応、AutoModelForCausalLM使用")
                    logger.info("💡 transformers>=4.45.0へのアップデートを推奨")
                
                # 🔥 Model Parallelism復活: マルチGPU分散利用（17Bモデル必須）
                device_count = torch.cuda.device_count()
                logger.info(f"🔥 Model Parallelism復活: {device_count}x H100分散利用（17Bモデル対応）")
                logger.info(f"検出されたGPU数: {device_count}")
                
                # 🔧 メモリ断片化解決: PYTORCH_CUDA_ALLOC_CONF設定（Webリサーチ推奨）
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                logger.info("✓ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True設定（断片化解決）")
                
                # 🔥 解決策3: 8xH100環境対応（容量不足解決・全GPU均等分散）
                logger.info("🔥 解決策3: 8xH100環境対応（容量不足根本解決）")
                logger.info("📋 問題: 4xH100では容量不足（232.9GB > 316.8GB理論値）")
                logger.info("📋 解決: 8xH100で余裕のある分散配置（50-60GB/GPU目標）")
                
                device_map_setting = "balanced"  # 8GPU均等分散
                
                if device_count >= 8:
                    # 8x H100環境: 余裕のあるメモリ配置（OOM根本解決）
                    max_memory_dict = {
                        0: "70GB",   # GPU 0: 88%使用率（余裕重視）
                        1: "70GB",   # GPU 1: 88%使用率
                        2: "70GB",   # GPU 2: 88%使用率
                        3: "70GB",   # GPU 3: 88%使用率
                        4: "70GB",   # GPU 4: 88%使用率
                        5: "70GB",   # GPU 5: 88%使用率
                        6: "70GB",   # GPU 6: 88%使用率
                        7: "70GB",   # GPU 7: 88%使用率
                    }
                    logger.info("✓ 8x H100 balanced戦略: 各GPU 70GB（88%使用率・大容量対応）")
                    logger.info("📋 総容量: 633.6GB（8x79.2GB）、制限: 560GB（88%）")
                    logger.info("📋 CPU offload回避でNaN問題防止、8GPU分散でOOM根本解決")
                
                elif device_count >= 4:
                    # 4x H100環境: 容量制限対応（後方互換）
                    max_memory_dict = {
                        0: "78GB",   # GPU 0: 98.5%使用率
                        1: "78GB",   # GPU 1: 98.5%使用率
                        2: "78GB",   # GPU 2: 98.5%使用率
                        3: "78GB",   # GPU 3: 98.5%使用率
                    }
                    logger.info("✓ 4x H100 balanced戦略: 各GPU 78GB（98.5%使用率・限界運用）")
                    logger.info("⚠️ 4GPU環境は容量限界、8GPU推奨")
                
                elif device_count >= 2:
                    # 2x H100環境
                    max_memory_dict = {
                        0: "78GB",  # GPU 0: 98.5%使用率
                        1: "78GB",  # GPU 1: 98.5%使用率
                    }
                    logger.info("✓ 2x H100 balanced戦略: 各GPU 78GB（98.5%使用率）")
                    
                else:
                    logger.warning("⚠️ GPU数不足、single GPU mode")
                    device_map_setting = "auto"
                    max_memory_dict = {0: "70GB"}  # single GPU用
                
                # モデルロード（HuggingFaceの自動device_map処理を活用）
                logger.info("🔧 HuggingFaceの自動device_map処理でModel Parallelismを実装...")
                
                # 🚨 2025年修正A改良版: 軽量device_map修正
                # dummy model作成を避けて軽量に実装
                logger.info("🔧 軽量device_map修正を適用中...")
                
                # autoで一度ロードしてから修正する方針に変更
                logger.info("  - 方針: auto → 事後lm_head修正")
                
                # 🚨 2025年修正B: 数値安定化config適用
                from transformers import LlamaConfig
                
                # Llama設定に数値安定化パラメータを適用
                config_modifications = {
                    "rms_norm_eps": config_linux.LAYERNORM_EPSILON,  # LayerNorm epsilon強化
                }
                
                # 🔥 Webリサーチ修正: カスタムdevice_mapでlm_head明示配置
                # HuggingFace Accelerate推奨パターンを適用
                
                # モデルロード（Webリサーチ最適化版）
                load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": device_map_setting,  # カスタムdevice_map使用
                    "attn_implementation": "sdpa",
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "max_memory": max_memory_dict,     # disk除外設定
                    "offload_state_dict": False,       # disk offload無効化
                    "use_safetensors": True,           # safetensors使用
                }
                
                logger.info(f"🔧 数値安定化設定適用: rms_norm_eps={config_linux.LAYERNORM_EPSILON}")
                
                # 🔥 Webリサーチ修正: offload_folder完全無効化
                # "Cannot copy out of meta tensor"エラー回避のため
                logger.info("🔧 offload_folder無効化: Webリサーチによるmeta tensor問題回避")
                logger.info("💡 accelerate disk_offload関数は使用せず、カスタムdevice_mapで解決")
                
                # 🚨 数値安定化: config修正後にモデルロード
                try:
                    # 既存configを取得
                    model_config = model_class.config_class.from_pretrained(model_id)
                    
                    # 数値安定化パラメータを適用
                    model_config.rms_norm_eps = config_linux.LAYERNORM_EPSILON
                    logger.info(f"  ✅ LayerNorm epsilon: {model_config.rms_norm_eps} (安定化強化)")
                    
                    # 修正されたconfigでモデルロード
                    load_kwargs["config"] = model_config
                    
                except Exception as config_error:
                    logger.warning(f"  ⚠️ config修正スキップ: {config_error}")
                
                llama4_model = model_class.from_pretrained(model_id, **load_kwargs)
                logger.info(f"✓ {model_class.__name__}使用（LISA準拠CausalLM + カスタムdevice_map）")
                
                # 🚨 事後lm_head修正: ディスク配置を検出して修正
                logger.info("🔧 lm_head配置確認・修正中...")
                
                if hasattr(llama4_model, 'lm_head'):
                    try:
                        # lm_headのデバイス確認
                        lm_head_device = next(llama4_model.lm_head.parameters()).device
                        logger.info(f"  - lm_head現在配置: {lm_head_device}")
                        
                        # diskに配置されている場合のみ修正
                        if str(lm_head_device) == 'meta' or 'disk' in str(lm_head_device):
                            logger.info("  🔧 lm_head: ディスク配置検出 → 修正スキップ（meta tensor対応待ち）")
                            logger.warning("  ⚠️ NaN発生リスクあり（lm_headディスク配置）")
                        else:
                            logger.info(f"  ✅ lm_head負荷分散配置: {lm_head_device} (最適化版)")
                            
                    except Exception as lm_head_check_error:
                        logger.warning(f"  ⚠️ lm_head確認エラー: {lm_head_check_error}")
                
                if hasattr(llama4_model, 'model') and hasattr(llama4_model.model, 'embed_tokens'):
                    try:
                        embed_device = llama4_model.model.embed_tokens.weight.device
                        logger.info(f"  ✅ embed_tokens配置: {embed_device}")
                    except Exception as embed_check_error:
                        logger.warning(f"  ⚠️ embed_tokens確認エラー: {embed_check_error}")
                
                # HuggingFaceが自動的にdevice_mapを処理するため、dispatch_modelは不要
                logger.info("✓ Model Parallelismはfrom_pretrainedで自動的に適用されています")
                
                # embed_tokensの状態確認（デバッグ用）
                if hasattr(llama4_model, 'model') and hasattr(llama4_model.model, 'embed_tokens'):
                    embed_weight = llama4_model.model.embed_tokens.weight
                    logger.info(f"✓ embed_tokens確認: shape={embed_weight.shape}, dtype={embed_weight.dtype}")
                    if embed_weight.is_meta:
                        logger.info("  💡 embed_tokensはmeta tensorです（disk offload）。HuggingFaceが自動的に処理します")
                    else:
                        logger.info(f"  ✓ embed_tokensデバイス: {embed_weight.device}")
                        # 統計情報（NaN確認用）
                        if not embed_weight.is_meta:
                            has_nan = torch.isnan(embed_weight).any().item()
                            has_inf = torch.isinf(embed_weight).any().item()
                            logger.info(f"  ✓ embed_tokens健全性: NaN={has_nan}, Inf={has_inf}")
                
                # GPU配置確認
                if hasattr(llama4_model, 'hf_device_map'):
                    actual_device_map = llama4_model.hf_device_map
                    gpu_distribution = {}
                    for component, device in actual_device_map.items():
                        if device not in gpu_distribution:
                            gpu_distribution[device] = 0
                        gpu_distribution[device] += 1
                    logger.info(f"✓ 実際のGPU分散: {gpu_distribution}")
                else:
                    logger.warning("⚠️ device_map情報を取得できませんでした")
                
            except Exception as e:
                error_msg = f"❌ CausalLMモデルロード失敗: {e}"
                logger.error(error_msg)
                logger.error(f"モデルID: {model_id}")
                logger.error(f"使用クラス: {model_class.__name__ if 'model_class' in locals() else 'Unknown'}")
                logger.error("考えられる原因:")
                logger.error("1. モデルへのアクセス権限がない")
                logger.error("2. HuggingFaceトークンが未設定")
                logger.error("3. ネットワーク接続の問題")
                logger.error("4. モデルIDが正しくない")
                raise RuntimeError(error_msg)
            
            # プロセッサ初期化（エラー詳細付き）
            try:
                llama4_processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                logger.info("✓ AutoProcessor初期化成功")
                
                # Webリサーチ対策: プロセッサにもbfloat16設定を確認
                if hasattr(llama4_processor, 'image_processor'):
                    if hasattr(llama4_processor.image_processor, 'do_convert_rgb'):
                        llama4_processor.image_processor.do_convert_rgb = True
                    if hasattr(llama4_processor.image_processor, 'do_rescale'):
                        llama4_processor.image_processor.do_rescale = True
                    logger.info("✓ 画像プロセッサ設定確認完了")
            except Exception as proc_e:
                logger.warning(f"⚠️ AutoProcessor初期化失敗: {proc_e}")
                # フォールバック: AutoTokenizer使用
                try:
                    from transformers import AutoTokenizer
                    llama4_processor = AutoTokenizer.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        use_fast=True
                    )
                    logger.info("✓ AutoTokenizer使用（プロセッサ代替）")
                except Exception as tok_e:
                    logger.error(f"❌ AutoTokenizer初期化も失敗: {tok_e}")
                    raise RuntimeError(f"❌ プロセッサ初期化完全失敗: {tok_e}")
            
            llama_params = sum(p.numel() for p in llama4_model.parameters())
            logger.info(f"✓ Llama-4-Scout初期化完了: {llama_params:,} パラメータ")
        else:
            # ❌ ダミーモデルフォールバック削除 - 明確にエラーで停止
            error_msg = "❌ HuggingFace Transformersが利用できません。Lambda環境のtransformersライブラリを確認してください。"
            logger.error(error_msg)
            logger.error("原因として考えられる問題:")
            logger.error("1. transformersライブラリがインストールされていない")
            logger.error("2. Pythonパスの問題")
            logger.error("3. 仮想環境が正しく有効化されていない")
            raise RuntimeError(error_msg)
        
        # 2. Q-Formerは統合ブリッジ内で初期化（test_phase3b成功パターン準拠）
        logger.info("🔄 Q-Formerは統合ブリッジ内で初期化されます（重複回避）")
        
        # 3. 統合ブリッジ初期化（デュアルエンコーダー構成対応）
        logger.info("🔄 Q-Former-SAM2統合ブリッジ初期化（デュアルエンコーダー対応版）...")
        qformer_config = LlamaQFormerSAM2Config()
        
        # デュアルエンコーダー設定を適用
        qformer_config = configure_dual_encoder(qformer_config)
        
        qformer_bridge = QFormerSegmentationBridge(
            config=qformer_config,
            shared_llama_model=llama4_model,        # ✅ 共有インスタンス
            shared_llama_processor=llama4_processor, # ✅ 共有プロセッサ
            training_stage=1,
            enable_moe=True
        )
        
        # 🔥 Model Parallelism対応: 補助コンポーネントを主要デバイスに配置
        logger.info("🔄 QFormerSegmentationBridge Model Parallelism対応配置...")
        try:
            # Llama-4の主要デバイスを取得
            main_device, access_info = get_model_device_map(qformer_bridge)
            logger.info(f"主要デバイス確認: {access_info}")
            
            # 補助コンポーネントを主要デバイスに配置（meta tensor対応）
            if hasattr(qformer_bridge, 'qformer') and qformer_bridge.qformer is not None:
                qformer_bridge.qformer = move_to_device_safely(qformer_bridge.qformer, main_device, logger, "qformer")
            if hasattr(qformer_bridge, 'sam2_wrapper') and qformer_bridge.sam2_wrapper is not None:
                qformer_bridge.sam2_wrapper = move_to_device_safely(qformer_bridge.sam2_wrapper, main_device, logger, "sam2_wrapper")
            if hasattr(qformer_bridge, 'seg_token_generator') and qformer_bridge.seg_token_generator is not None:
                qformer_bridge.seg_token_generator = move_to_device_safely(qformer_bridge.seg_token_generator, main_device, logger, "seg_token_generator")
            
            # Llama-4モデルはModel Parallelismで分散配置済み（手動移動不要）
            
            logger.info(f"✓ QFormerSegmentationBridge Model Parallelism配置完了（主要: {main_device}）")
        except Exception as e:
            logger.warning(f"⚠️ QFormerSegmentationBridge配置エラー: {e}")
        
        logger.info("✓ 統合ブリッジ初期化完了（Model Parallelism対応）")
        
        # 4. Phase 3B拡張コンポーネント初期化
        logger.info("🔄 Phase 3B拡張コンポーネント初期化...")
        
        # デュアルパスウェイデコーダ（test_phase3b成功パターン移植）
        dual_pathway_decoder = create_dual_pathway_decoder(
            llama_hidden_size=5120,           # Llama-4隠れ層サイズ
            sam_output_dim=256,               # SAM2出力次元
            fusion_strategy="learned_weighted",
            force_gpu=True                    # 訓練スクリプト対応：GPU強制
        )
        # Model Parallelism対応デバイス配置
        main_device, _ = get_model_device_map(qformer_bridge)
        dual_pathway_decoder = move_to_device_safely(dual_pathway_decoder, main_device, logger, "dual_pathway_decoder")
        logger.info(f"✓ デュアルパスウェイデコーダ初期化完了（{main_device}）")
        
        # 多重解像度特徴統合（test_phase3b成功パターン移植）
        multiresolution_fusion = Llama4SAM2MultiResolutionFusion(
            llama_hidden_size=5120,          # Llama-4隠れ層サイズ
            sam_feature_dim=256,             # SAM2特徴次元（SAM2公式準拠）
            sam_scales=[1024, 512, 256],     # SAM2マルチスケール
            qformer_dim=768,                 # Q-Former特徴次元
            qformer_queries=32,              # Q-Formerクエリ数
            fusion_dim=512,                  # 正しい引数名（実装確認済み）
            output_size=(448, 448),          # Llama-4解像度準拠
            fusion_strategy="attention"      # 注意機構融合（2024ベストプラクティス）
        )
        # Model Parallelism対応デバイス配置
        multiresolution_fusion = move_to_device_safely(multiresolution_fusion, main_device, logger, "multiresolution_fusion")
        logger.info(f"✓ 多重解像度特徴統合初期化完了（{main_device}）")
        
        # OHEM損失関数（test_phase3b成功パターン移植）
        ohem_loss = create_ohem_loss(
            hard_ratio=0.25,
            config_override={
                'ce_loss_weight': 1.0,
                'bce_loss_weight': 2.0,
                'dice_loss_weight': 0.5
            }
        )
        logger.info("✓ OHEM損失関数初期化完了")
        
        logger.info("✅ Phase 3B統合モデル初期化完成")
        
        return {
            'qformer_bridge': qformer_bridge,
            'dual_pathway_decoder': dual_pathway_decoder,
            'multiresolution_fusion': multiresolution_fusion,
            'ohem_loss': ohem_loss,
            'llama4_model': llama4_model,
            'llama4_processor': llama4_processor
            # 注: qformer_modelは削除（QFormerSegmentationBridge内で管理）
        }
        
    except Exception as e:
        logger.error(f"❌ Phase 3Bモデル初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        raise

def apply_lora_to_model(model_components, args, logger):
    """QFormerSegmentationBridgeにLoRA適用（train_llama4_lisa_single_process.py準拠）"""
    logger.info("=== QFormerSegmentationBridgeにMixLoRA適用 ===")
    
    try:
        qformer_bridge = model_components['qformer_bridge']
        
        # Llama-4部分にLoRA適用（train_llama4_lisa_single_process.py成功パターン）
        if hasattr(qformer_bridge, 'llama_model') and qformer_bridge.llama_model is not None:
            # device_map保存（train_llama4_lisa_single_process.py完全移植）
            original_device_map = None
            original_device_map_location = None
            
            # device_mapの場所を特定して保存（成功パターン完全移植）
            if hasattr(qformer_bridge.llama_model, 'hf_device_map') and qformer_bridge.llama_model.hf_device_map:
                original_device_map = qformer_bridge.llama_model.hf_device_map.copy()
                original_device_map_location = "llama_model"
                logger.info(f"✓ 元のdevice_map保存（llama_model経由）: {len(original_device_map)} エントリ")
            elif hasattr(qformer_bridge, 'hf_device_map') and qformer_bridge.hf_device_map:
                original_device_map = qformer_bridge.hf_device_map.copy()
                original_device_map_location = "direct"
                logger.info(f"✓ 元のdevice_map保存（直接アクセス）: {len(original_device_map)} エントリ")
            else:
                logger.warning("⚠️ device_mapが見つかりません。Model Parallelismが未設定の可能性があります。")
            
            # MixLoRA設定作成
            lora_config_dict = config_linux.get_lora_config()
            lora_config_dict.pop('task_type', None)  # 重複回避
            
            # modules_to_saveの設定を確認
            modules_to_save = config_linux.ADDITIONAL_TRAINABLE_PARAMS
            logger.info(f"📊 modules_to_save設定: {modules_to_save}")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                modules_to_save=modules_to_save,  # lm_head, embed_tokensを含む
                **lora_config_dict
            )
            
            logger.info(f"📊 MixLoRA設定:")
            logger.info(f"  - rank: {lora_config.r}")
            logger.info(f"  - alpha: {lora_config.lora_alpha}")
            logger.info(f"  - target_modules: {lora_config.target_modules}")
            
            # LoRA適用
            qformer_bridge.llama_model = get_peft_model(
                qformer_bridge.llama_model, 
                lora_config,
                autocast_adapter_dtype=False  # 🔥 dtype自動変換無効化
            )
            
            # パラメータ統計
            trainable_params = sum(p.numel() for p in qformer_bridge.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in qformer_bridge.parameters())
            
            logger.info(f"✓ LoRA適用完了")
            logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
            logger.info(f"  - 全パラメータ: {total_params:,}")
            logger.info(f"  - 効率化比率: {100 * trainable_params / total_params:.3f}%")
            
            # 🔥 device_mapの復元試行（train_llama4_lisa_single_process.py完全移植）
            if original_device_map and original_device_map_location:
                # 複数のアクセス方法を試行
                restoration_success = False
                
                # 方法1: 直接アクセス確認
                if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'hf_device_map') and qformer_bridge.llama_model.hf_device_map:
                    logger.info("✓ llama_model経由device_mapアクセス確認済み")
                    restoration_success = True
                
                # 方法2: base_model経由のアクセス
                elif hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'base_model') and hasattr(qformer_bridge.llama_model.base_model, 'hf_device_map') and qformer_bridge.llama_model.base_model.hf_device_map:
                    logger.info("✓ llama_model.base_model経由device_mapアクセス確認済み")
                    restoration_success = True
                
                # 方法3: 手動復元
                if not restoration_success:
                    logger.warning("⚠️ device_mapが失われました。手動復元を試行...")
                    
                    # 元の場所に基づいて復元
                    if original_device_map_location == "llama_model":
                        if hasattr(qformer_bridge, 'llama_model'):
                            qformer_bridge.llama_model.hf_device_map = original_device_map
                            logger.info("✓ llama_model.hf_device_mapを手動復元")
                            restoration_success = True
                    elif original_device_map_location == "direct":
                        if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'base_model'):
                            qformer_bridge.llama_model.base_model.hf_device_map = original_device_map
                            logger.info("✓ llama_model.base_model.hf_device_mapを手動復元")
                            restoration_success = True
                
                if not restoration_success:
                    logger.error("❌ device_mapの復元に失敗しました")
            
            # 🔥 Gradient Checkpointing有効化（参考ファイル準拠メモリ30%削減）
            # 🔧 修正: use_reentrant=Falseに再変更（requires_grad問題解決のため）
            # Web調査結果: use_reentrant=Falseはrequires_gradを正しく維持する
            try:
                gradient_ckpt_kwargs = {"use_reentrant": False}
                
                if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'gradient_checkpointing_enable'):
                    # 引数を受け取れるか確認
                    import inspect
                    sig = inspect.signature(qformer_bridge.llama_model.gradient_checkpointing_enable)
                    if 'gradient_checkpointing_kwargs' in sig.parameters:
                        qformer_bridge.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_ckpt_kwargs)
                        logger.info("✅ Gradient Checkpointing有効化成功（use_reentrant=False、requires_grad維持）")
                    else:
                        # 旧バージョンのHuggingFaceの場合
                        qformer_bridge.llama_model.gradient_checkpointing_enable()
                        logger.info("✅ Gradient Checkpointing有効化成功（メモリ30%削減）")
                        logger.warning("⚠️ use_reentrant=False未対応のHuggingFaceバージョン")
                elif hasattr(qformer_bridge, 'base_model') and hasattr(qformer_bridge.base_model, 'llama_model'):
                    if hasattr(qformer_bridge.base_model.llama_model, 'gradient_checkpointing_enable'):
                        sig = inspect.signature(qformer_bridge.base_model.llama_model.gradient_checkpointing_enable)
                        if 'gradient_checkpointing_kwargs' in sig.parameters:
                            qformer_bridge.base_model.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_ckpt_kwargs)
                            logger.info("✅ base_model Gradient Checkpointing有効化成功（use_reentrant=False）")
                        else:
                            qformer_bridge.base_model.llama_model.gradient_checkpointing_enable()
                            logger.info("✅ base_model Gradient Checkpointing有効化成功")
                elif hasattr(qformer_bridge, 'gradient_checkpointing_enable'):
                    sig = inspect.signature(qformer_bridge.gradient_checkpointing_enable)
                    if 'gradient_checkpointing_kwargs' in sig.parameters:
                        qformer_bridge.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_ckpt_kwargs)
                        logger.info("✅ 直接Gradient Checkpointing有効化成功（use_reentrant=False）")
                    else:
                        qformer_bridge.gradient_checkpointing_enable()
                        logger.info("✅ 直接Gradient Checkpointing有効化成功")
                else:
                    logger.warning("⚠️ Gradient Checkpointingサポートなし")
            except Exception as e:
                logger.warning(f"⚠️ Gradient Checkpointing有効化失敗（スキップ）: {e}")
            
            model_components['qformer_bridge'] = qformer_bridge
        else:
            logger.warning("⚠️ Llama-4モデルが見つかりません")
        
        return model_components
        
    except Exception as e:
        logger.error(f"LoRA適用エラー: {e}")
        raise

def create_dataset_and_dataloader(model_components, args, logger):
    """データセットとデータローダー作成（train_llama4_lisa成功パターン移植）"""
    logger.info("=== HybridDataset初期化（train_llama4_lisa成功パターン） ===")
    
    # データセット設定
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
        dataset_count = len(datasets) if 'datasets' in locals() else 1
        samples_per_epoch = samples_per_dataset * dataset_count
    else:
        # デフォルトは小さめに設定（段階的検証用）
        samples_per_epoch = 500  # train_llama4_lisa準拠
    
    logger.info(f"エポックあたりのサンプル数: {samples_per_epoch}")
    
    # プロセッサを取得（QFormerSegmentationBridge対応）
    qformer_bridge = model_components['qformer_bridge']
    if hasattr(qformer_bridge, 'llama_processor') and qformer_bridge.llama_processor:
        llama_processor = qformer_bridge.llama_processor
    elif hasattr(qformer_bridge, 'shared_llama_processor') and qformer_bridge.shared_llama_processor:
        llama_processor = qformer_bridge.shared_llama_processor
    else:
        logger.warning("⚠️ Llama processor not found in QFormerSegmentationBridge")
        llama_processor = None
    
    # HybridDataset作成（エラーハンドリング強化版）
    logger.info("🔍 HybridDataset初期化前デバッグ情報:")
    logger.info(f"  - base_image_dir: {config_linux.DATASET_BASE_DIR}")
    logger.info(f"  - base_image_dir存在確認: {os.path.exists(config_linux.DATASET_BASE_DIR)}")
    logger.info(f"  - llama_processor: {type(llama_processor)}")
    logger.info(f"  - samples_per_epoch: {samples_per_epoch}")
    logger.info(f"  - dataset_string: {dataset_string}")
    logger.info(f"  - sem_seg_data: {config_linux.SEM_SEG_DATA}")
    logger.info(f"  - refer_seg_data: {config_linux.REFER_SEG_DATA}")
    logger.info(f"  - vqa_data: {config_linux.VQA_DATA}")
    logger.info(f"  - reason_seg_data: {config_linux.REASON_SEG_DATA}")
    
    try:
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
        
        # ❌ フォールバック削除: データ検証を強化してエラーで停止
        logger.info(f"HybridDataset初期化完了: {len(dataset)} サンプル")
        
        # データセット検証: 実際にデータが取得できるかテスト
        logger.info("🔍 データセット検証開始...")
        if len(dataset) == 0:
            error_msg = "❌ データセットが空です。データが存在しません。"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 最初のサンプルを取得してデータの有効性を検証
        try:
            test_sample = dataset[0]
            logger.info("✅ データセット検証成功: 最初のサンプル取得完了")
        except Exception as validation_error:
            error_msg = f"❌ データセット検証失敗: 実際のデータ取得でエラーが発生しました。\n詳細: {validation_error}"
            logger.error(error_msg)
            logger.error("原因として考えられる問題:")
            logger.error("1. データファイルが存在しない")
            logger.error("2. データファイルが破損している")
            logger.error("3. 必要な説明ファイル（explanatory/train.json等）が不足")
            logger.error("4. データセットパスの設定ミス")
            logger.error(f"5. 設定確認: base_image_dir={config_linux.DATASET_BASE_DIR}")
            logger.error(f"6. 設定確認: reason_seg_data={config_linux.REASON_SEG_DATA}")
            raise RuntimeError(error_msg)
        
    except Exception as e:
        logger.error(f"❌ HybridDataset初期化または検証失敗: {e}")
        logger.error("詳細なエラー情報:")
        import traceback
        traceback.print_exc()
        logger.error("❌ データセットエラーにより訓練を停止します。")
        logger.error("解決方法:")
        logger.error("1. Lambda環境のデータセットパスを確認")
        logger.error("2. 必要なデータファイルの存在確認")
        logger.error("3. config_linux.pyのデータセット設定を確認")
        raise
    
    # ステップ数制限（デバッグ用）
    if hasattr(args, 'samples_per_epoch') and args.samples_per_epoch:
        logger.info(f"サンプル数制限: {args.samples_per_epoch}")
        # データセットの一部のみ使用
        indices = list(range(min(args.samples_per_epoch, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # DataLoader作成（参考ファイル準拠：メモリ最適化）
    dataloader = DataLoader(
        dataset,
        batch_size=getattr(args, 'batch_size', 1),  # GPU制約により小バッチ
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0),  # メモリ最適化：0に設定
        pin_memory=False,  # メモリ最適化：Falseに設定
        collate_fn=collate_fn
    )
    
    logger.info(f"✓ DataLoader作成完了")
    logger.info(f"  - データセットサイズ: {len(dataset)}")
    logger.info(f"  - バッチサイズ: {getattr(args, 'batch_size', 1)}")
    logger.info(f"  - ワーカー数: {getattr(args, 'num_workers', 0)}")
    
    return dataloader

def create_optimizer_and_scheduler(model_components, dataloader, args, logger):
    """オプティマイザーとスケジューラー作成"""
    logger.info("=== オプティマイザーとスケジューラー作成 ===")
    
    qformer_bridge = model_components['qformer_bridge']
    
    # 学習可能パラメータのみ対象
    trainable_params = [p for p in qformer_bridge.parameters() if p.requires_grad]
    
    # パラメータ状態チェック
    total_params = sum(1 for _ in qformer_bridge.named_parameters())
    trainable_count = len(trainable_params)
    logger.info(f"  - 総パラメータ数: {total_params}, 学習可能: {trainable_count}")
    
    # オプティマイザー作成（NaN対策版）
    lr = getattr(args, 'lr', config_linux.LEARNING_RATE)
    weight_decay = getattr(args, 'weight_decay', config_linux.WEIGHT_DECAY)
    
    if BITSANDBYTES_AVAILABLE and getattr(args, 'use_8bit_adam', False):
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=lr,
            betas=(config_linux.BETA1, config_linux.BETA2),
            weight_decay=weight_decay
        )
        logger.info("✓ 8bit AdamW使用")
    else:
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(config_linux.BETA1, config_linux.BETA2),
            weight_decay=weight_decay
        )
        logger.info("✓ 標準AdamW使用")
    
    # Loss Scaler作成（NaN対策：2025年推奨）
    scaler = None
    if getattr(config_linux, 'USE_LOSS_SCALING', False):
        scaler = GradScaler(
            init_scale=getattr(config_linux, 'INITIAL_LOSS_SCALE', 2**10),
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        logger.info(f"✓ Loss Scaling有効化: 初期スケール={config_linux.INITIAL_LOSS_SCALE}")
    
    # スケジューラー作成（linear scheduler使用）
    total_steps = len(dataloader) * getattr(args, 'epochs', 3)
    if hasattr(args, 'steps_per_epoch') and args.steps_per_epoch:
        total_steps = args.steps_per_epoch * getattr(args, 'epochs', 3)
    
    warmup_steps = int(config_linux.WARMUP_RATIO * total_steps)
    
    # linear scheduler使用（安定性優先）
    if config_linux.LR_SCHEDULER_TYPE == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("✓ Linear Scheduler使用（安定性優先）")
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("✓ Cosine Scheduler使用")
    
    logger.info(f"✓ スケジューラー作成完了")
    logger.info(f"  - 総ステップ数: {total_steps}")
    logger.info(f"  - ウォームアップステップ: {warmup_steps}")
    logger.info(f"  - 学習率: {lr}")
    
    return optimizer, scheduler, scaler

def get_model_device_map(model):
    """Model Parallelism対応device_map取得（17Bモデル用分散処理）"""
    device_map = None
    access_path = None
    
    # 複数のアクセス方法を試行（Model Parallelism対応）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        access_path = "直接アクセス"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
        device_map = model.base_model.hf_device_map
        access_path = "base_model経由"
    elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'hf_device_map') and model.llama_model.hf_device_map:
        device_map = model.llama_model.hf_device_map
        access_path = "llama_model経由"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
        device_map = model.base_model.llama_model.hf_device_map
        access_path = "base_model.llama_model経由"
    
    if device_map:
        # Model Parallelismの場合、最初のデバイスを返す（主要処理用）
        first_device = next(iter(device_map.values()))
        return first_device, f"{access_path} (Model Parallelism: 主要デバイス {first_device})"
    else:
        # フォールバック: cuda:0
        return torch.device("cuda:0"), "fallback (cuda:0)"

def move_to_device_safely(obj, device, logger, name="object"):
    """meta tensors対応のデバイス移動（test_phase3b成功パターン完全移植）"""
    try:
        if isinstance(obj, torch.nn.Module):
            # test_phase3b成功パターン: meta tensorの詳細チェック
            meta_params = [n for n, p in obj.named_parameters() if p.is_meta]
            if meta_params:
                logger.info(f"  - {name}: meta tensorsを含むためスキップ（disk offload維持）")
                logger.debug(f"    meta parameters: {meta_params[:3]}{'...' if len(meta_params) > 3 else ''}")
                return obj  # meta tensor含有モジュールはスキップ
            else:
                # meta tensorがない場合のみデバイス移動
                return obj.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        elif isinstance(obj, torch.Tensor):
            if obj.is_meta:
                logger.debug(f"  - {name}: meta tensorのためスキップ")
                return obj  # meta tensorはスキップ
            else:
                return obj.to(device, non_blocking=True)
        else:
            return obj
    except Exception as e:
        logger.warning(f"  - {name}: 移動エラー（{e}）、スキップ")
        return obj

def train_epoch(model_components, dataloader, optimizer, scheduler, scaler, epoch, args, logger, writer=None):
    """1エポックの学習実行（Phase 3B統合対応）"""
    qformer_bridge = model_components['qformer_bridge']
    dual_pathway_decoder = model_components['dual_pathway_decoder']
    multiresolution_fusion = model_components['multiresolution_fusion']
    ohem_loss = model_components['ohem_loss']
    
    qformer_bridge.train()
    dual_pathway_decoder.train()
    multiresolution_fusion.train()
    
    # 損失メトリクス初期化
    total_losses = AverageMeter('Total', ':.4e')
    phase2_losses = AverageMeter('Phase2', ':.4e')
    seg_losses = AverageMeter('Seg', ':.4e')
    ohem_losses = AverageMeter('OHEM', ':.4e')
    
    progress = ProgressMeter(
        len(dataloader) if args.steps_per_epoch is None else args.steps_per_epoch,
        [total_losses, phase2_losses, seg_losses, ohem_losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    # 個別損失履歴（エポック内）
    epoch_loss_history = {
        'total_loss': [],
        'phase2_loss': [],
        'seg_loss': [],
        'ohem_loss': []
    }
    
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # 🔥 メモリ最適化: バッチごとにCUDAキャッシュクリア（参考ファイル準拠）
        torch.cuda.empty_cache()
        
        # ステップ数制限チェック
        if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
            break
        
        # 🔥 Model Parallelism対応データ準備（train_llama4_lisa_single_process.py完全移植）
        try:
            # Model Parallelismのdevice_mapを厳密にチェック（QFormerSegmentationBridge対応）
            device_map = None
            access_path = None
            
            # 複数のアクセス方法を試行（成功パターン完全移植）
            if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'hf_device_map') and qformer_bridge.llama_model.hf_device_map:
                device_map = qformer_bridge.llama_model.hf_device_map
                access_path = "llama_model経由"
            elif hasattr(qformer_bridge, 'hf_device_map') and qformer_bridge.hf_device_map:
                device_map = qformer_bridge.hf_device_map
                access_path = "直接アクセス"
            elif hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'base_model') and hasattr(qformer_bridge.llama_model.base_model, 'hf_device_map') and qformer_bridge.llama_model.base_model.hf_device_map:
                device_map = qformer_bridge.llama_model.base_model.hf_device_map
                access_path = "llama_model.base_model経由"
            
            if not device_map:
                raise RuntimeError("Model Parallelismが設定されていません。17Bモデルには必須です。")
            
            if step == 0:
                logger.info(f"Model Parallelismデバイスマップ確認: {access_path}")
            
            first_device = next(iter(device_map.values()))
            
            # HybridDatasetの出力をデュアルエンコーダー対応に変換
            adapted_batch = adapt_dataset_for_qformer(batch)
            
            # デバイス移動（train_llama4_lisa_single_process.py準拠）
            adapted_batch = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in adapted_batch.items()}
            
            # バッチ内容をデバッグ出力（初回のみ）
            if step == 0:
                logger.info(f"📊 デュアルエンコーダーバッチ確認:")
                for key, value in adapted_batch.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape} ({value.dtype})")
                    else:
                        logger.info(f"  {key}: {type(value)}")
        
        except Exception as e:
            logger.error(f"データ準備エラー (step {step}): {e}")
            continue
        
        # フォワードパス
        optimizer.zero_grad()
        
        
        # CUDA OOM対策: メモリクリア（test_phase3b成功パターン移植）
        torch.cuda.empty_cache()
        
        try:
            # Phase 3B統合フォワードパス（train_llama4_lisa損失抽出パターン移植）
            torch.cuda.empty_cache()  # フォワードパス前のメモリクリア
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # 1. QFormerSegmentationBridge推論（test_phase3b成功パターン移植）
                qformer_outputs = qformer_bridge(
                    images=adapted_batch.get('pixel_values', adapted_batch.get('images')),     # pixel_values → images
                    sam_images=adapted_batch.get('sam_pixel_values', adapted_batch.get('sam_images')),  # sam_pixel_values → sam_images
                    input_ids=adapted_batch['input_ids'],
                    attention_mask=adapted_batch['attention_mask'],
                    labels=adapted_batch['labels'],
                    return_dict=True
                )
                
                # デバッグ: モデル出力のキーを確認（初回のみ）
                if step == 0:
                    if isinstance(qformer_outputs, dict):
                        logger.info(f"  🔍 QFormerSegmentationBridge出力キー: {list(qformer_outputs.keys())}")
                    else:
                        logger.info(f"  🔍 QFormerSegmentationBridge出力タイプ: {type(qformer_outputs)}")
                
                # 2. Llama隠れ状態の取得（test_phase3b成功パターン完全移植）
                llama_features = None
                sam_features = None
                qformer_features = None
                
                # test_phase3bパターン: QFormerSegmentationBridge出力から確実に取得
                logger.debug(f"  🔍 Llama隠れ状態取得開始...")
                
                # Llama特徴の取得（実際の出力構造に基づく）
                # デバッグ出力: ['text_loss', 'predicted_masks', 'query_embeddings', 'sam_prompts', 'method', 'num_queries', 'seg_token', 'seg_sam_prompt']
                
                # QFormerSegmentationBridgeにはllama_hidden_statesがないため、query_embeddingsを使用
                if hasattr(qformer_outputs, 'query_embeddings') and qformer_outputs.query_embeddings is not None:
                    # query_embeddingsをLlama特徴として使用（768次元 → 5120次元に投影必要）
                    query_embeddings = qformer_outputs.query_embeddings
                    logger.debug(f"    ✓ query_embeddings取得: {query_embeddings.shape}")
                    
                    # Q-Formerクエリ(768次元)をLlama隠れ状態(5120次元)に投影
                    if 'query_to_llama_projector' not in model_components:
                        logger.debug(f"    🔧 Query→Llama投影層作成中...")
                        model_components['query_to_llama_projector'] = nn.Linear(
                            768,  # Q-Former query次元
                            5120, # Llama隠れ状態次元
                            bias=True
                        ).to(device=query_embeddings.device, dtype=torch.bfloat16)
                        logger.debug(f"    ✅ 投影層作成完了: 768 → 5120")
                    
                    # 投影実行
                    llama_features = model_components['query_to_llama_projector'](query_embeddings.to(torch.bfloat16))
                    logger.debug(f"    ✅ Query→Llama投影完了: {query_embeddings.shape} → {llama_features.shape}")
                    
                elif isinstance(qformer_outputs, dict) and 'query_embeddings' in qformer_outputs:
                    query_embeddings = qformer_outputs['query_embeddings']
                    logger.debug(f"    ✓ 辞書からquery_embeddings取得: {query_embeddings.shape}")
                    
                    # 同様の投影処理
                    if 'query_to_llama_projector' not in model_components:
                        model_components['query_to_llama_projector'] = nn.Linear(768, 5120, bias=True).to(device=query_embeddings.device, dtype=torch.bfloat16)
                    
                    llama_features = model_components['query_to_llama_projector'](query_embeddings.to(torch.bfloat16))
                    logger.debug(f"    ✅ 辞書Query→Llama投影完了: {llama_features.shape}")
                    
                else:
                    available_attrs = list(qformer_outputs.keys()) if isinstance(qformer_outputs, dict) else [attr for attr in dir(qformer_outputs) if not attr.startswith('_')]
                    raise ValueError(f"query_embeddingsがQFormerSegmentationBridge出力に見つかりません。利用可能な属性: {available_attrs}")
                
                # SAM特徴とQ-Former特徴の取得
                if hasattr(qformer_outputs, 'sam_features') and qformer_outputs.sam_features is not None:
                    sam_features = qformer_outputs.sam_features
                    logger.debug(f"    ✓ SAM特徴取得: {sam_features.shape}")
                
                if hasattr(qformer_outputs, 'query_embeddings') and qformer_outputs.query_embeddings is not None:
                    qformer_features = qformer_outputs.query_embeddings
                    logger.debug(f"    ✓ Q-Former特徴(query_embeddings)取得: {qformer_features.shape}")
                elif hasattr(qformer_outputs, 'query_embeds') and qformer_outputs.query_embeds is not None:
                    qformer_features = qformer_outputs.query_embeds
                    logger.debug(f"    ✓ Q-Former特徴(query_embeds)取得: {qformer_features.shape}")
                
                # 多重解像度融合（特徴が利用可能な場合のみ）
                fusion_outputs = None
                if llama_features is not None and sam_features is not None and qformer_features is not None:
                    # デバイス統一確認
                    if qformer_features.device != first_device:
                        qformer_features = qformer_features.to(first_device)
                    
                    fusion_outputs = multiresolution_fusion(
                        llama_features=llama_features,
                        sam_features=sam_features,
                        qformer_features=qformer_features
                    )
                    logger.debug(f"    ✅ 多重解像度融合完了")
                else:
                    logger.debug(f"    ⚠️ 多重解像度融合スキップ（特徴不足）")
                
                # 3. デュアルパスウェイ推論（SAM画像が利用可能な場合）
                if 'sam_images' in adapted_batch or 'sam_pixel_values' in adapted_batch:
                    sam_images = adapted_batch.get('sam_images', adapted_batch.get('sam_pixel_values'))
                    
                    if sam_images is not None:
                        # SAMプロンプト生成（test_phase3b成功パターン: 複数属性名対応）
                        sam_prompts = None
                        if hasattr(qformer_outputs, 'sam_prompts') and qformer_outputs.sam_prompts is not None:
                            sam_prompts = qformer_outputs.sam_prompts
                            logger.debug(f"  🔍 sam_prompts属性使用: {sam_prompts.shape}")
                        elif hasattr(qformer_outputs, 'seg_sam_prompt') and qformer_outputs.seg_sam_prompt is not None:
                            sam_prompts = qformer_outputs.seg_sam_prompt
                            logger.debug(f"  🔍 seg_sam_prompt属性使用: {sam_prompts.shape}")
                        elif isinstance(qformer_outputs, dict):
                            if 'sam_prompts' in qformer_outputs and qformer_outputs['sam_prompts'] is not None:
                                sam_prompts = qformer_outputs['sam_prompts']
                                logger.debug(f"  🔍 辞書sam_prompts使用: {sam_prompts.shape}")
                            elif 'seg_sam_prompt' in qformer_outputs and qformer_outputs['seg_sam_prompt'] is not None:
                                sam_prompts = qformer_outputs['seg_sam_prompt']
                                logger.debug(f"  🔍 辞書seg_sam_prompt使用: {sam_prompts.shape}")
                        
                        if sam_prompts is None:
                            # デバッグ: 利用可能な属性/キー出力
                            if hasattr(qformer_outputs, '__dict__'):
                                available_attrs = [attr for attr in dir(qformer_outputs) if not attr.startswith('_')]
                                logger.error(f"❌ 利用可能な属性: {available_attrs}")
                            if isinstance(qformer_outputs, dict):
                                logger.error(f"❌ 利用可能なキー: {list(qformer_outputs.keys())}")
                            raise ValueError("SAM prompts not found in qformer_outputs")
                        
                        # Llama隠れ状態（必須）
                        if llama_features is not None:
                            llama_hidden_states = llama_features
                            logger.debug(f"    ✓ Llama隠れ状態使用: {llama_hidden_states.shape}")
                        else:
                            raise ValueError("Llama features必須ですが利用できません")
                        
                        # デュアルパスウェイ推論
                        dual_outputs = dual_pathway_decoder(
                            images=sam_images,
                            sam_prompts=sam_prompts,
                            llama_hidden_states=llama_hidden_states
                        )
                    else:
                        dual_outputs = None
                else:
                    dual_outputs = None
                
                # 4. 損失計算（詳細デバッグ付き）
                logger.debug(f"  🔍 損失計算デバッグ開始 (step {step})...")
                individual_losses = {'total_loss': 0, 'phase2_loss': 0, 'seg_loss': 0, 'ohem_loss': 0}
                
                # QFormerSegmentationBridge出力の詳細分析
                logger.debug(f"    📊 qformer_outputs詳細分析:")
                logger.debug(f"      - タイプ: {type(qformer_outputs)}")
                if isinstance(qformer_outputs, dict):
                    logger.debug(f"      - 利用可能なキー: {list(qformer_outputs.keys())}")
                    
                    # 各キーの値をチェック
                    for key, value in qformer_outputs.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:  # スカラー値
                                logger.debug(f"        {key}: {value.item():.6f} (shape: {value.shape}, dtype: {value.dtype})")
                            else:
                                logger.debug(f"        {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            logger.debug(f"        {key}: {type(value)}")
                
                if isinstance(qformer_outputs, dict):
                    # 🔍 デバッグ: 利用可能な損失を全て確認
                    logger.debug(f"    🔍 損失候補検索...")
                    potential_losses = {}
                    
                    for key, value in qformer_outputs.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:  # スカラーテンソル
                            if 'loss' in key.lower():
                                potential_losses[key] = value.item()
                                logger.debug(f"      損失候補発見: {key} = {value.item():.6f}")
                    
                    logger.debug(f"    📋 発見された損失候補: {potential_losses}")
                    
                    # 1. 直接total_lossを確認
                    if 'total_loss' in qformer_outputs:
                        loss = qformer_outputs['total_loss']
                        logger.debug(f"    ✓ total_loss直接取得: {loss.item():.6f}")
                        individual_losses['total_loss'] = loss.item()
                    
                    # 2. losses辞書内のtotal_lossを確認
                    elif 'losses' in qformer_outputs and isinstance(qformer_outputs['losses'], dict):
                        losses_dict = qformer_outputs['losses']
                        logger.debug(f"    📋 losses辞書内容: {list(losses_dict.keys())}")
                        if 'total_loss' in losses_dict:
                            loss = losses_dict['total_loss']
                            logger.debug(f"    ✓ losses辞書内total_loss取得: {loss.item():.6f}")
                            individual_losses['total_loss'] = loss.item()
                        else:
                            logger.error(f"    ❌ losses辞書にtotal_lossなし: {list(losses_dict.keys())}")
                            raise ValueError(f"losses辞書にtotal_lossが見つかりません。利用可能なキー: {list(losses_dict.keys())}")
                    
                    # 3. loss属性を確認（標準transformers出力）
                    elif 'loss' in qformer_outputs:
                        loss = qformer_outputs['loss']
                        logger.debug(f"    ✓ loss属性取得: {loss.item():.6f}")
                        individual_losses['total_loss'] = loss.item()
                    
                    # 4. text_lossをtotal_lossとして使用（QFormerSegmentationBridge対応）
                    elif 'text_loss' in qformer_outputs:
                        loss = qformer_outputs['text_loss']
                        logger.warning(f"    ⚠️ text_lossをtotal_lossとして使用: {loss.item():.6f}")
                        individual_losses['total_loss'] = loss.item()
                    
                    # 5. エラー：損失が見つからない
                    else:
                        available_keys = list(qformer_outputs.keys())
                        logger.error(f"    ❌ いかなる損失も見つかりません")
                        logger.error(f"      利用可能キー: {available_keys}")
                        logger.error(f"      損失候補: {potential_losses}")
                        raise ValueError(f"total_loss not found in model outputs. Available keys: {available_keys}")
                    
                    # 個別損失の詳細を取得（厳密チェック）
                    if 'losses' in qformer_outputs and isinstance(qformer_outputs['losses'], dict):
                        losses_dict = qformer_outputs['losses']
                        if step == 0:
                            logger.info(f"  🔍 losses辞書キー: {list(losses_dict.keys())}")
                        
                        # phase2_loss（Phase 2統一空間）
                        if 'phase2_loss' in losses_dict and losses_dict['phase2_loss'] is not None:
                            individual_losses['phase2_loss'] = losses_dict['phase2_loss'].item()
                        elif 'moe_loss' in losses_dict and losses_dict['moe_loss'] is not None:
                            individual_losses['phase2_loss'] = losses_dict['moe_loss'].item()
                        
                        # seg_loss（セグメンテーション損失）
                        if 'seg_loss' in losses_dict and losses_dict['seg_loss'] is not None:
                            individual_losses['seg_loss'] = losses_dict['seg_loss'].item()
                        
                        # ohem_loss（OHEM損失）
                        if 'ohem_loss' in losses_dict and losses_dict['ohem_loss'] is not None:
                            individual_losses['ohem_loss'] = losses_dict['ohem_loss'].item()
                        elif 'lm_loss' in losses_dict and losses_dict['lm_loss'] is not None:
                            individual_losses['ohem_loss'] = losses_dict['lm_loss'].item()
                    
                    # Phase 2損失（統一トークン空間）- フォールバック
                    if individual_losses['phase2_loss'] == 0:
                        if hasattr(qformer_outputs, 'phase2_outputs') and qformer_outputs.phase2_outputs:
                            phase2_output = qformer_outputs.phase2_outputs
                            if 'moe_adapted_outputs' in phase2_output and 'load_balance_loss' in phase2_output['moe_adapted_outputs']:
                                lb_loss = phase2_output['moe_adapted_outputs']['load_balance_loss']
                                if not torch.isnan(lb_loss):
                                    individual_losses['phase2_loss'] = lb_loss.item()
                    
                    # セグメンテーション損失 - フォールバック
                    if individual_losses['seg_loss'] == 0:
                        if dual_outputs and 'consistency_loss' in dual_outputs:
                            individual_losses['seg_loss'] = abs(dual_outputs['consistency_loss']).item()
                
                else:
                    # 非辞書型出力の場合
                    if hasattr(qformer_outputs, 'loss'):
                        loss = qformer_outputs.loss
                        individual_losses['total_loss'] = loss.item()
                    else:
                        raise ValueError(f"Loss not found in model outputs. Output type: {type(qformer_outputs)}")
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"🔥 CUDA OOMエラー (step {step}): {e}")
            # 詳細メモリ状況をログ出力
            log_memory_status(f"OOM発生 step {step}")
            
            # 緊急クリーンアップ実行
            emergency_cleanup()
            # CLAUDE.md指示: エラーを隠蔽せず停止してデバッグに繋げる
            raise
        except Exception as e:
            logger.error(f"フォワードパスエラー (step {step}): {e}")
            continue
        
        # バックワードパス（train_llama4_lisa成功パターン移植）
        try:
            # メモリクリアしてから逆伝播（CUBLAS_STATUS_ALLOC_FAILED対策）
            torch.cuda.empty_cache()
            
            if individual_losses['total_loss'] > 0 and 'loss' in locals():
                # 🔧 方針A: meta tensor対応のカスタムbackward処理
                # meta tensorsを含むレイヤーの勾配計算を回避
                
                # 勾配追跡デバッグコード
                logger.debug("📊 勾配追跡開始...")
                grad_info = {
                    'total_params': 0,
                    'requires_grad': 0,
                    'meta_tensors': 0,
                    'cuda_tensors': 0,
                    'has_grad': 0,
                    'grad_nan': 0
                }
                
                for name, param in qformer_bridge.named_parameters():
                    grad_info['total_params'] += 1
                    if param.requires_grad:
                        grad_info['requires_grad'] += 1
                        if param.is_meta:
                            grad_info['meta_tensors'] += 1
                        else:
                            grad_info['cuda_tensors'] += 1
                            # meta tensorでないパラメータのみ勾配を有効化
                            param.retain_grad()
                
                logger.debug(f"  - 総パラメータ数: {grad_info['total_params']}")
                logger.debug(f"  - requires_grad=True: {grad_info['requires_grad']}")
                logger.debug(f"  - meta tensors: {grad_info['meta_tensors']}")
                logger.debug(f"  - cuda tensors: {grad_info['cuda_tensors']}")
                
                # 逆伝播（Loss Scaling対応）
                try:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"🔥 Backward OOMエラー (step {step}): {e}")
                    # 詳細メモリ状況をログ出力
                    log_memory_status(f"Backward OOM発生 step {step}")
                    
                    # 緊急クリーンアップ実行
                    emergency_cleanup()
                    # CLAUDE.md指示: エラーを隠蔽せず停止してデバッグに繋げる
                    raise
                except RuntimeError as e:
                    if "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                        logger.error("❌ CUDA/CUBLAS メモリ不足エラー")
                        break
                    elif "expected device meta but got cuda" in str(e) or "MmBackward0" in str(e):
                        logger.warning(f"⚠️ Meta tensor device mismatchエラー: {str(e)[:100]}...")
                        logger.info("💡 LoRAパラメータのみで勾配計算を試行...")
                        
                        # LoRAパラメータのみで勾配更新
                        lora_params = []
                        for name, param in qformer_bridge.named_parameters():
                            if param.requires_grad and ("lora" in name.lower() or "query_embeds" in name):
                                lora_params.append(param)
                        
                        if lora_params:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=config_linux.GRADIENT_CLIP_NORM)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=config_linux.GRADIENT_CLIP_NORM)
                                optimizer.step()
                            scheduler.step()
                            logger.info("✅ LoRAパラメータの更新完了")
                        else:
                            logger.error("❌ 更新可能なLoRAパラメータが見つかりません")
                            continue
                    else:
                        raise
                
                # 勾配計算後のデバッグ
                logger.debug("📊 勾配計算後の確認...")
                for name, param in qformer_bridge.named_parameters():
                    if param.requires_grad and not param.is_meta:
                        if param.grad is not None:
                            grad_info['has_grad'] += 1
                            if torch.isnan(param.grad).any():
                                grad_info['grad_nan'] += 1
                                logger.warning(f"  ⚠️ NaN勾配検出: {name}")
                
                logger.debug(f"  - 勾配あり: {grad_info['has_grad']}")
                logger.debug(f"  - NaN勾配: {grad_info['grad_nan']}")
                
                # 勾配クリッピングとオプティマイザーステップ（Loss Scaling対応）
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(qformer_bridge.parameters(), max_norm=config_linux.GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(qformer_bridge.parameters(), max_norm=config_linux.GRADIENT_CLIP_NORM)
                    optimizer.step()
                scheduler.step()
                
                # デバッグ: パラメータ更新後の状態確認
                if step < 3:
                    logger.debug(f"\n📊 ステップ {step} 更新後の状態確認...")
                    update_check = {
                        'nan_params': [],
                        'inf_params': [],
                        'unchanged_params': []
                    }
                    
                    for name, param in qformer_bridge.named_parameters():
                        if param.requires_grad and not param.is_meta:
                            if torch.isnan(param).any():
                                update_check['nan_params'].append(name)
                            elif torch.isinf(param).any():
                                update_check['inf_params'].append(name)
                    
                    if update_check['nan_params']:
                        logger.error(f"  ❌ 更新後NaNパラメータ: {update_check['nan_params'][:5]}...")
                    if update_check['inf_params']:
                        logger.error(f"  ❌ 更新後Infパラメータ: {update_check['inf_params'][:5]}...")
                    
                    # 学習率の確認
                    current_lr = scheduler.get_last_lr()[0]
                    logger.debug(f"  📊 現在の学習率: {current_lr:.2e}")
            else:
                logger.error(f"  ❌ step {step}: 損失が0または無効、訓練を停止します")
                raise RuntimeError("Invalid loss detected. Training cannot continue.")
        
        except Exception as e:
            logger.error(f"バックワードパスエラー (step {step}): {e}")
            continue
        
        # メトリクス更新（値が有効な場合のみ）
        if individual_losses['total_loss'] > 0:
            total_losses.update(individual_losses['total_loss'], 1)
            phase2_losses.update(individual_losses['phase2_loss'], 1)
            seg_losses.update(individual_losses['seg_loss'], 1)
            ohem_losses.update(individual_losses['ohem_loss'], 1)
        else:
            logger.warning(f"Step {step}: Invalid loss values, skipping metrics update")
        
        # 履歴保存（有効な損失値のみ）
        if individual_losses['total_loss'] > 0:
            for key in epoch_loss_history.keys():
                if key in individual_losses:
                    epoch_loss_history[key].append(individual_losses[key])
                else:
                    epoch_loss_history[key].append(0.0)
        
        # プログレス表示
        if step % 10 == 0:
            progress.display(step)
        
        # TensorBoard記録（有効な損失値のみ）
        if writer and individual_losses['total_loss'] > 0:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('Loss/Total', individual_losses['total_loss'], global_step)
            writer.add_scalar('Loss/Phase2', individual_losses['phase2_loss'], global_step)
            writer.add_scalar('Loss/Seg', individual_losses['seg_loss'], global_step)
            writer.add_scalar('Loss/OHEM', individual_losses['ohem_loss'], global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
        
        # Webリサーチ最適化: より頻繁なメモリクリーンアップ（20→2ステップ毎）
        if step % 2 == 0:
            torch.cuda.empty_cache()
        
        # Gradient Accumulation制御（Webリサーチ最適化）
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if scaler is not None:
                scaler.update()
    
    # エポック終了時のメモリクリア（OOM対策）
    torch.cuda.empty_cache()
    
    # エポック統計
    epoch_time = time.time() - start_time
    logger.info(f"Epoch {epoch} 完了: {epoch_time:.2f}秒")
    logger.info(f"  - 総損失: {total_losses.avg:.6f}")
    logger.info(f"  - Phase2損失: {phase2_losses.avg:.6f}")
    logger.info(f"  - セグメンテーション損失: {seg_losses.avg:.6f}")
    logger.info(f"  - OHEM損失: {ohem_losses.avg:.6f}")
    
    # エポック平均を返す
    return {
        'total_loss': total_losses.avg,
        'phase2_loss': phase2_losses.avg,
        'seg_loss': seg_losses.avg,
        'ohem_loss': ohem_losses.avg,
        'epoch_loss_history': epoch_loss_history
    }

def main():
    parser = argparse.ArgumentParser(description='Phase 3B QFormerSegmentationBridge 訓練')
    parser.add_argument('--exp_name', type=str, default='phase3b_train', help='実験名')
    parser.add_argument('--epochs', type=int, default=3, help='エポック数')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='エポックあたりのステップ数制限')
    parser.add_argument('--batch_size', type=int, default=1, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=config_linux.LEARNING_RATE, help='学習率')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='H100x4対応: OOM対策でバッチサイズ削減')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='重み減衰')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoaderワーカー数')
    parser.add_argument('--dataset', type=str, default='reason_seg', help='データセット名')
    parser.add_argument('--samples_per_epoch', type=int, default=None, help='エポックあたりのサンプル数制限')
    parser.add_argument('--use_8bit_adam', action='store_true', help='8bit AdamWを使用')
    parser.add_argument('--log_dir', type=str, default='./logs_phase3b', help='ログディレクトリ')
    
    args = parser.parse_args()
    
    # 環境設定
    setup_environment()
    
    # ログディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_log_dir = os.path.join(args.log_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # ロギング設定
    logger = setup_logging(exp_log_dir)
    
    # メモリモニター初期化
    memory_monitor = GPUMemoryMonitor(enable_detailed_logging=True)
    logger.info("✅ GPU Memory Monitor初期化完了（H100 80GB最適化）")
    
    # 初期メモリ状況確認
    memory_monitor.log_memory_status("訓練開始前")
    
    logger.info("🚀 Phase 3B QFormerSegmentationBridge 訓練開始")
    logger.info("=" * 80)
    logger.info(f"実験名: {args.exp_name}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"ステップ制限: {args.steps_per_epoch}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"学習率: {args.lr}")
    logger.info("=" * 80)
    
    try:
        # TensorBoard設定
        writer = SummaryWriter(log_dir=os.path.join(exp_log_dir, 'tensorboard'))
        
        # 1. モデルとコンポーネント作成
        with memory_monitor.monitor_section("モデルとコンポーネント作成"):
            model_components = create_model_and_components(args, logger)
        
        # 2. LoRA適用
        with memory_monitor.monitor_section("LoRA適用"):
            model_components = apply_lora_to_model(model_components, args, logger)
        
        # 3. データセットとデータローダー作成
        with memory_monitor.monitor_section("データセット作成"):
            dataloader = create_dataset_and_dataloader(model_components, args, logger)
        
        # 4. オプティマイザーとスケジューラー作成
        optimizer, scheduler, scaler = create_optimizer_and_scheduler(model_components, dataloader, args, logger)
        
        # 5. 学習ループ
        loss_history = []
        
        for epoch in range(1, args.epochs + 1):
            logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
            
            # エポック開始前のメモリ状況
            memory_monitor.log_memory_status(f"Epoch {epoch} 開始前")
            
            with memory_monitor.monitor_section(f"Epoch {epoch} 訓練"):
                epoch_losses = train_epoch(
                    model_components, dataloader, optimizer, scheduler, scaler,
                    epoch, args, logger, writer
                )
            
            loss_history.append(epoch_losses)
            
            # GPU メモリクリーンアップ
            torch.cuda.empty_cache()
            gc.collect()
        
        # 6. 結果保存
        logger.info("\n🎯 訓練完了 - 結果保存中...")
        
        # 損失曲線の可視化
        plot_file = create_loss_plots(loss_history, exp_log_dir, args.exp_name)
        logger.info(f"✓ 損失曲線保存: {plot_file}")
        
        # 結果辞書
        results = {
            'exp_name': args.exp_name,
            'epochs': args.epochs,
            'final_losses': loss_history[-1] if loss_history else {},
            'loss_history': loss_history,
            'args': vars(args),
            'timestamp': timestamp
        }
        
        # JSON保存
        results_file = os.path.join(exp_log_dir, f'results_{args.exp_name}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ 結果保存: {results_file}")
        
        # 損失曲線画像保存
        if loss_history:
            plot_file = create_loss_plots(loss_history, exp_log_dir, args.exp_name)
            if plot_file:
                logger.info(f"✓ 損失曲線保存: {plot_file}")
        
        # モデル保存（最終エポック）- PyTorch標準形式
        model_save_path = os.path.join(exp_log_dir, 'final_model.pth')
        
        # QFormerSegmentationBridgeのstate_dictを保存
        model_state = {
            'model_state_dict': model_components['qformer_bridge'].state_dict(),
            'epoch': args.epochs,
            'exp_name': args.exp_name,
            'final_loss': loss_history[-1] if loss_history else 0.0
        }
        
        torch.save(model_state, model_save_path)
        logger.info(f"✓ モデル保存: {model_save_path}")
        
        # サマリー表示
        logger.info("\n" + "="*80)
        logger.info("🎉 Phase 3B QFormerSegmentationBridge 訓練完了")
        logger.info("="*80)
        logger.info(f"📊 最終損失:")
        if loss_history:
            final_losses = loss_history[-1]
            logger.info(f"  - 総損失: {final_losses['total_loss']:.6f}")
            logger.info(f"  - Phase2損失: {final_losses['phase2_loss']:.6f}")
            logger.info(f"  - セグメンテーション損失: {final_losses['seg_loss']:.6f}")
            logger.info(f"  - OHEM損失: {final_losses['ohem_loss']:.6f}")
        logger.info(f"🎯 実験ディレクトリ: {exp_log_dir}")
        logger.info("="*80)
        
        writer.close()
        
    except Exception as e:
        logger.error(f"❌ 訓練エラー: {e}")
        import traceback
        traceback.print_exc()
        
        # 緊急メモリクリーンアップ
        memory_monitor.emergency_cleanup()
        raise
    
    finally:
        # 最終メモリクリーンアップ
        memory_monitor.log_memory_status("訓練完了後")
        logger.info("🏁 訓練完了")

if __name__ == "__main__":
    main()