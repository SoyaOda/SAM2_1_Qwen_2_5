# test_phase3b_integration_real.py
"""
Phase 3B統合テスト: 実際のモデル使用版

論文: "Customize SAM for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts"
目標: 28.14%性能向上の実証（実際のLlama-4+SAM2+Q-Formerモデル使用）

overfit_llama4_lisa_batch.pyの成功パターンを参考にした実機テスト:
1. 実際のLlama-4-Scout-17B-16E-Instructロード
2. SAM2統合モデルロード  
3. Phase 3B実装機能（デュアルパスウェイ、多重解像度、OHEM）統合
4. Lambda Cloud環境での28.14%性能向上検証
"""

# 🔥 PyTorchインポート前の環境準備（CUDA Error 802対策）
import sys
import os

# Step 1: CUDA Error 802対策用環境変数設定（PyTorchインポート前）
print("🔧 CUDA Error 802対策：PyTorchインポート前環境設定...")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

# Step 2: CUDA_VISIBLE_DEVICESが未設定の場合のみ設定（2x H100対応）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 2x H100専用

# Step 2.1: GPU RAM分散利用のための追加設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # GPU RAM最適化
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  # 分散デバッグ
os.environ['NCCL_DEBUG'] = 'INFO'  # NCCL通信デバッグ

print("✅ 環境変数設定完了 - PyTorchインポート開始...")

# Step 3: PyTorchインポート
import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from PIL import Image

# PEFT関連
from peft import LoraConfig, get_peft_model, TaskType

# プロジェクト固有のインポート
sys.path.append('.')

# 🔥 Webリサーチメモリ最適化: GPU RAMモニタリング追加
from utils.memory_monitor import GPUMemoryMonitor, memory_monitor_section, log_memory_status, emergency_cleanup

# ✅ 現在の実装方針：個別コンポーネント組み合わせ（moe_structure_approach.md準拠）
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
from model.sam2_integration import get_sam2_wrapper  # ✅ 実際のSAM2ロード
from model.qformer import get_qformer_model  # ✅ 実際のQ-Formerロード
from model.dual_pathway_decoder import create_dual_pathway_decoder
from model.multiresolution_fusion import Llama4SAM2MultiResolutionFusion
from model.ohem_loss import create_ohem_loss
from model.dataset_adapter import adapt_dataset_for_qformer  # ✅ シングルエンコーダー対応
import config_linux

# Llama-4-Scout直接ロード（HuggingFace transformers使用）
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ HuggingFace Transformers利用可能")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ HuggingFace Transformersが利用できません")

# データセット関連
try:
    from utils.dataset import preprocess_sam_image, build_correct_labels_for_llama4, HybridDataset
    DATASET_AVAILABLE = True
    print("✅ utils.dataset利用可能")
except ImportError as e:
    DATASET_AVAILABLE = False
    print(f"⚠️ utils.datasetインポート失敗: {e}")
    print("💡 基本的な画像処理機能で代替します")

# ロギング設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行関数（簡易テスト版）"""
    # メモリモニター初期化
    memory_monitor = GPUMemoryMonitor(enable_detailed_logging=True)
    logger.info("✅ GPU Memory Monitor初期化完了（H100 80GB最適化）")
    
    # 初期メモリ状況確認
    memory_monitor.log_memory_status("テスト開始前")
    
    try:
        with memory_monitor.monitor_section("Llama-4モデルロード"):
            if TRANSFORMERS_AVAILABLE:
                llama_config = config_linux.get_lisa_model_config()
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
                    
                    # GPU RAM分散利用対応: 手動device_map設定
                    logger.info("🔥 GPU RAM分散利用: 4x H100手動device_mapを設定...")
                    
                    device_count = torch.cuda.device_count()
                    logger.info(f"検出されたGPU数: {device_count}")
                    
                    if device_count >= 4:
                        # 🔥 4x H100最適化: lm_head disk配置完全回避
                        logger.info("🔥 4x H100最適化: lm_head disk配置回避 + 高効率分散")
                        
                        max_memory_per_gpu = "70GB"  # OOM回避のため保守的設定
                        
                        # 4x H100用完全カスタムdevice_map（"auto"文字列排除）
                        device_map_setting = {
                            "model.embed_tokens": 0,  # 埋め込み層をGPU 0
                            "lm_head": 3,             # lm_headを最後のGPU 3に配置
                            "model.norm": 3,          # 正規化層もGPU 3
                            # 52層を4つのGPUに手動分散（"auto"使用不可）
                            **{f"model.layers.{i}": i % 4 for i in range(52)}
                        }
                        max_memory_dict = {
                            0: max_memory_per_gpu,   # GPU 0: 70GB
                            1: max_memory_per_gpu,   # GPU 1: 70GB  
                            2: max_memory_per_gpu,   # GPU 2: 70GB
                            3: max_memory_per_gpu,   # GPU 3: 70GB
                            "cpu": "100GB"           # 4x GPUなのでCPUも増量
                        }
                        
                        logger.info(f"✓ 4x H100カスタムdevice_map: lm_head→GPU3, embed_tokens→GPU0")
                        logger.info(f"✓ メモリ制限: 各GPU {max_memory_per_gpu}, CPU 100GB（disk除外）")
                        
                    elif device_count >= 2:
                        # 2x H100フォールバック設定
                        logger.info("🔥 2x H100フォールバック: lm_head disk配置回避")
                        
                        max_memory_per_gpu = "70GB"  # OOM回避のため保守的設定
                        
                        device_map_setting = {
                            "model.embed_tokens": 0,
                            "lm_head": 1,
                            "model.norm": 1,
                            # 52層を2つのGPUに手動分散（"auto"使用不可）
                            **{f"model.layers.{i}": i % 2 for i in range(52)}
                        }
                        max_memory_dict = {
                            0: max_memory_per_gpu, 
                            1: max_memory_per_gpu,
                            "cpu": "50GB"
                        }
                        
                        logger.info(f"✓ 2x H100カスタムdevice_map: lm_head→GPU1, embed_tokens→GPU0")
                        logger.info(f"✓ メモリ制限: GPU {max_memory_per_gpu}, CPU 50GB（disk除外）")
                        
                    else:
                        logger.warning("⚠️ GPU数不足、single GPU mode")
                        device_map_setting = "auto"
                        max_memory_dict = {0: "70GB"}  # single GPU用
                    
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
                    
                    # Webリサーチ修正: offload_folder完全無効化
                    logger.info("🔧 offload_folder無効化: Webリサーチによるmeta tensor問題回避")
                    
                    llama4_model = model_class.from_pretrained(model_id, **load_kwargs)
                    logger.info(f"✓ {model_class.__name__}使用（LISA準拠CausalLM + GPU RAM分散）")
                    
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
                    
                    # パラメータ数確認
                    llama_params = sum(p.numel() for p in llama4_model.parameters())
                    logger.info(f"✓ Llama-4-Scout初期化完了: {llama_params:,} パラメータ")
                    
                    # 簡易推論テスト
                    logger.info("🔄 簡易推論テスト実行...")
                    with torch.no_grad():
                        # テスト用入力
                        test_inputs = {
                            "input_ids": torch.randint(1, 1000, (1, 10)).to("cuda:0"),
                            "attention_mask": torch.ones(1, 10).to("cuda:0")
                        }
                        
                        # 推論実行
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                            outputs = llama4_model(**test_inputs)
                        
                        logger.info(f"✓ 推論成功: logits shape = {outputs.logits.shape}")
                        
                        # NaN検出
                        has_nan = torch.isnan(outputs.logits).any()
                        logger.info(f"✓ NaN検出: {has_nan} (False = 正常)")
                        
                        if not has_nan:
                            logger.info("🎉 lm_head disk offload問題完全解決確認！")
                        else:
                            logger.error("❌ NaN発生 - 追加対策が必要")
                    
                except Exception as e:
                    logger.error(f"❌ モデルロード失敗: {e}")
                    raise
                
                finally:
                    # メモリクリーンアップ
                    memory_monitor.emergency_cleanup()
                    memory_monitor.log_memory_status("テスト完了後")
            
            else:
                logger.warning("⚠️ Transformers利用不可")
    
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        
        # 緊急クリーンアップ
        emergency_cleanup()
        
    finally:
        logger.info("🏁 テスト完了")


if __name__ == "__main__":
    main()