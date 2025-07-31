# test_phase3b_enhanced_fixed.py
"""
Enhanced Model改修版テスト（Phase 3B）- 修正版

test_phase3b_simple.pyで成功した修正を反映し、
マルチスケールとLoRAを段階的に検証できるようにしたバージョン
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
import gc
import sys
import os
from dataclasses import dataclass
import json
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_repo'))

# Enhanced Model（改修版）
try:
    from model.enhanced_llama4_qformer_sam2 import (
        EnhancedQFormerSegmentationBridge, 
        EnhancedLlamaQFormerSAM2Config
    )
    ENHANCED_MODEL_AVAILABLE = True
except ImportError as e:
    ENHANCED_MODEL_AVAILABLE = False
    print(f"❌ Enhanced Modelインポート失敗: {e}")

# Baseline Model（既存版）
try:
    from model.llama4_qformer_sam2 import (
        QFormerSegmentationBridge,
        LlamaQFormerSAM2Config
    )
    BASELINE_MODEL_AVAILABLE = True
except ImportError as e:
    BASELINE_MODEL_AVAILABLE = False
    print(f"❌ Baseline Modelインポート失敗: {e}")

# ユーティリティ
try:
    from utils.dataset import preprocess_sam_image, build_correct_labels_for_llama4
    DATASET_UTILS_AVAILABLE = True
except ImportError:
    DATASET_UTILS_AVAILABLE = False
    print("⚠️ utils.dataset一部機能が利用できません（ダミーデータを使用）")

import config_linux

# Transformers
try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ HuggingFace Transformers利用可能")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ HuggingFace Transformersが利用できません")

# ロギング設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """IoU計算"""
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()


def calculate_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Dice係数計算"""
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum()
    return (2.0 * intersection / (pred.float().sum() + target.float().sum() + 1e-8)).item()


def evaluate_masks(predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, float]:
    """マスク評価メトリクス計算"""
    ious = []
    dices = []
    
    for pred, gt in zip(predictions, targets):
        # 最良マスクを選択（複数マスクの場合）
        if pred.dim() == 3 and pred.size(0) > 1:
            # IoUベースで最良マスクを選択
            best_iou = 0
            best_idx = 0
            for i in range(pred.size(0)):
                iou = calculate_iou(pred[i], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            pred = pred[best_idx]
        elif pred.dim() == 3:
            pred = pred[0]
        
        if gt.dim() == 3:
            gt = gt[0]
        
        # IoUとDice計算
        iou = calculate_iou(pred, gt)
        dice = calculate_dice(pred, gt)
        
        ious.append(iou)
        dices.append(dice)
    
    return {
        'mIoU': np.mean(ious),
        'mDice': np.mean(dices),
        'IoUs': ious,
        'Dices': dices
    }


@dataclass
class EnhancedTestConfig:
    """改修版テスト設定"""
    # 基本設定
    output_dir: str = "./phase3b_enhanced_test_results"
    batch_size: int = 1
    
    # 改修機能テスト設定（段階的検証用）
    test_baseline: bool = True                # ベースライン（全機能無効）
    test_multiscale_only: bool = True         # マルチスケールのみ
    test_lora_only: bool = True               # LoRAのみ
    test_multiscale_and_lora: bool = True     # 両方有効
    test_text_input: bool = True              # テキスト入力対応
    test_modality_separation: bool = True      # 視覚・言語分離
    
    # 評価設定
    num_test_samples: int = 5
    test_modes: List[str] = None
    
    def __post_init__(self):
        """デフォルト値設定"""
        if self.test_modes is None:
            self.test_modes = ['itg']  # 基本はITGモードのみでテスト
        
        # config_linux設定の取得
        self.llama_model_id = config_linux.LLAMA_MODEL_ID
        self.sam2_model_id = config_linux.SAM2_HF_MODEL_ID


class GPUMemoryMonitor:
    """GPU メモリ監視ユーティリティ"""
    
    def __init__(self, enable_detailed_logging: bool = False):
        self.enable_detailed_logging = enable_detailed_logging
        self.initial_memory = {}
        self.peak_memory = {}
        
        # 初期メモリ状況を記録
        for i in range(torch.cuda.device_count()):
            self.initial_memory[i] = torch.cuda.memory_allocated(i)
            self.peak_memory[i] = self.initial_memory[i]
    
    def log_memory_status(self, phase: str):
        """現在のメモリ状況をログ"""
        if not self.enable_detailed_logging:
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 GPU Memory Status - {phase}")
        logger.info(f"{'='*60}")
        
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_memory = torch.cuda.max_memory_allocated(i) / 1024**3
            
            logger.info(f"GPU {i}: Allocated: {allocated:.2f}GB, "
                       f"Reserved: {reserved:.2f}GB, Peak: {max_memory:.2f}GB")
            
            # ピークメモリ更新
            current = torch.cuda.memory_allocated(i)
            if current > self.peak_memory[i]:
                self.peak_memory[i] = current


class EnhancedModelTester:
    """改修版モデルテスター"""
    
    def __init__(self, config: EnhancedTestConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 結果保存用
        self.results = {
            "test_logs": [],
            "metrics": {},
            "comparison_results": {},
            "config": config.__dict__
        }
        
        # 出力ディレクトリ作成
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # GPUメモリモニター
        self.memory_monitor = GPUMemoryMonitor(enable_detailed_logging=True)
        logger.info("✅ GPU Memory Monitor初期化完了")
        
        # モデル初期化
        self.enhanced_model = None
        self.baseline_model = None
        self.llama4_model = None
        self.llama4_processor = None
    
    def setup_models(self):
        """モデル初期化"""
        logger.info("=== モデル初期化開始 ===")
        
        try:
            # メモリ状況確認
            self.memory_monitor.log_memory_status("初期化開始前")
            
            # 1. Llama-4初期化（共有インスタンス）
            logger.info("🧠 Llama-4-Scout初期化...")
            if TRANSFORMERS_AVAILABLE:
                self.llama4_model = AutoModelForCausalLM.from_pretrained(
                    self.config.llama_model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
                
                self.llama4_processor = AutoProcessor.from_pretrained(
                    self.config.llama_model_id,
                    trust_remote_code=True
                )
                
                logger.info("✅ Llama-4初期化完了")
            else:
                raise RuntimeError("Transformersが利用できません")
            
            # メモリ状況確認
            self.memory_monitor.log_memory_status("モデル初期化完了")
            
        except Exception as e:
            logger.error(f"❌ モデル初期化失敗: {e}")
            raise
    
    def create_enhanced_model(self, enable_multiscale: bool = False, enable_lora: bool = False):
        """Enhanced Modelを指定された設定で作成"""
        logger.info(f"\n🚀 Enhanced Model初期化 (multiscale={enable_multiscale}, lora={enable_lora})...")
        
        if ENHANCED_MODEL_AVAILABLE:
            enhanced_config = EnhancedLlamaQFormerSAM2Config()
            
            model = EnhancedQFormerSegmentationBridge(
                config=enhanced_config,
                shared_llama_model=self.llama4_model,
                shared_llama_processor=self.llama4_processor,
                enable_lora=enable_lora,
                enable_multiscale=enable_multiscale,
                training_stage=1
            )
            
            logger.info(f"✅ Enhanced Model初期化完了")
            logger.info(f"  - マルチスケール: {'有効' if enable_multiscale else '無効'}")
            logger.info(f"  - LoRA: {'有効' if enable_lora else '無効'}")
            
            return model
        else:
            raise RuntimeError("Enhanced Modelが利用できません")
    
    def generate_test_samples(self, num_samples: int = 5) -> List[Dict[str, torch.Tensor]]:
        """テスト用サンプル生成"""
        samples = []
        
        for i in range(num_samples):
            # ダミー画像（448x448）
            image = torch.randn(3, 448, 448).to(self.device)
            
            # ダミーラベル（セグメンテーションマスク）
            labels = torch.randint(0, 2, (448, 448), dtype=torch.long).to(self.device)
            
            # テキスト入力
            text_input = f"Segment the object number {i+1} in this image"
            
            samples.append({
                'image': image.unsqueeze(0),  # バッチ次元追加
                'labels': labels.unsqueeze(0),
                'text_input': text_input
            })
        
        logger.info(f"✅ {num_samples}個のテストサンプル生成完了")
        return samples
    
    def test_configuration(self, config_name: str, enable_multiscale: bool, enable_lora: bool):
        """特定の設定でモデルをテスト"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 設定テスト: {config_name}")
        logger.info(f"{'='*80}")
        
        try:
            # モデル作成
            model = self.create_enhanced_model(
                enable_multiscale=enable_multiscale,
                enable_lora=enable_lora
            )
            
            # テストサンプル生成
            samples = self.generate_test_samples(self.config.num_test_samples)
            
            # メトリクス記録
            all_metrics = []
            
            for idx, sample in enumerate(samples):
                logger.info(f"\n📊 サンプル {idx+1}/{len(samples)} 処理中...")
                
                try:
                    # Forward処理
                    with torch.no_grad():
                        outputs = model(
                            images=sample['image'],
                            text_input=[sample['text_input']],
                            labels=sample['labels'],
                            mode='itg'
                        )
                    
                    # 結果確認
                    logger.info(f"✅ Forward成功")
                    logger.info(f"  - マスク形状: {outputs['masks'].shape}")
                    logger.info(f"  - 損失値: {outputs.get('loss', 'N/A')}")
                    
                    # メトリクス計算
                    if 'masks' in outputs:
                        masks = outputs['masks']
                        # 最良マスクを選択
                        if masks.dim() == 4 and masks.size(1) > 1:
                            # 複数マスクの場合、IoUスコアが最高のものを選択
                            if 'iou_scores' in outputs:
                                best_idx = outputs['iou_scores'].argmax(dim=1)
                                masks = masks[torch.arange(masks.size(0)), best_idx]
                            else:
                                masks = masks[:, 0]  # 最初のマスク
                        elif masks.dim() == 4:
                            masks = masks[:, 0]
                        
                        # 閾値処理
                        pred_masks = (masks > 0.5).float()
                        
                        # メトリクス計算
                        metrics = evaluate_masks([pred_masks[0]], [sample['labels'][0]])
                        all_metrics.append(metrics)
                        logger.info(f"  - IoU: {metrics['mIoU']:.4f}")
                        logger.info(f"  - Dice: {metrics['mDice']:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ サンプル処理エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # 平均メトリクス計算
            if all_metrics:
                avg_iou = np.mean([m['mIoU'] for m in all_metrics])
                avg_dice = np.mean([m['mDice'] for m in all_metrics])
                
                result = {
                    'config_name': config_name,
                    'enable_multiscale': enable_multiscale,
                    'enable_lora': enable_lora,
                    'avg_iou': avg_iou,
                    'avg_dice': avg_dice,
                    'num_samples': len(all_metrics),
                    'status': 'success'
                }
                
                logger.info(f"\n📈 平均メトリクス:")
                logger.info(f"  - 平均IoU: {avg_iou:.4f}")
                logger.info(f"  - 平均Dice: {avg_dice:.4f}")
            else:
                result = {
                    'config_name': config_name,
                    'enable_multiscale': enable_multiscale,
                    'enable_lora': enable_lora,
                    'status': 'no_metrics'
                }
            
            self.results['test_logs'].append(result)
            
            # メモリクリーンアップ
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 設定テスト失敗: {str(e)}")
            result = {
                'config_name': config_name,
                'enable_multiscale': enable_multiscale,
                'enable_lora': enable_lora,
                'status': 'failed',
                'error': str(e)
            }
            self.results['test_logs'].append(result)
            return result
    
    def run_tests(self):
        """全テスト実行"""
        logger.info("\n" + "="*80)
        logger.info("🚀 Enhanced Model段階的検証開始")
        logger.info("="*80)
        
        # モデル初期化
        self.setup_models()
        
        # 段階的テスト実行
        test_configs = []
        
        if self.config.test_baseline:
            test_configs.append(("ベースライン（全機能無効）", False, False))
        
        if self.config.test_multiscale_only:
            test_configs.append(("マルチスケールのみ有効", True, False))
        
        if self.config.test_lora_only:
            test_configs.append(("LoRAのみ有効", False, True))
        
        if self.config.test_multiscale_and_lora:
            test_configs.append(("マルチスケール＋LoRA有効", True, True))
        
        # 各設定でテスト
        for config_name, enable_multiscale, enable_lora in test_configs:
            self.test_configuration(config_name, enable_multiscale, enable_lora)
        
        # 結果サマリー
        self.generate_summary()
        
        # 結果保存
        self.save_results()
    
    def generate_summary(self):
        """テスト結果のサマリー生成"""
        logger.info("\n" + "="*80)
        logger.info("📊 テスト結果サマリー")
        logger.info("="*80)
        
        for result in self.results['test_logs']:
            logger.info(f"\n設定: {result['config_name']}")
            logger.info(f"  - マルチスケール: {'有効' if result.get('enable_multiscale', False) else '無効'}")
            logger.info(f"  - LoRA: {'有効' if result.get('enable_lora', False) else '無効'}")
            logger.info(f"  - ステータス: {result['status']}")
            
            if result['status'] == 'success':
                logger.info(f"  - 平均IoU: {result.get('avg_iou', 'N/A'):.4f}")
                logger.info(f"  - 平均Dice: {result.get('avg_dice', 'N/A'):.4f}")
            elif result['status'] == 'failed':
                logger.info(f"  - エラー: {result.get('error', 'Unknown')}")
    
    def save_results(self):
        """結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.output_dir) / f"test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✅ テスト結果を保存: {output_file}")


def main():
    """メイン実行関数"""
    # テスト設定
    config = EnhancedTestConfig(
        output_dir="./phase3b_enhanced_test_results",
        batch_size=1,
        num_test_samples=3,  # 各設定で3サンプルテスト
        test_baseline=True,
        test_multiscale_only=True,
        test_lora_only=True,
        test_multiscale_and_lora=True
    )
    
    # テスター作成
    tester = EnhancedModelTester(config)
    
    # テスト実行
    try:
        tester.run_tests()
        logger.info("\n🎉 全テスト完了！")
    except Exception as e:
        logger.error(f"\n❌ テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()