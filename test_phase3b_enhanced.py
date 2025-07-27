# test_phase3b_enhanced.py
"""
Phase 3B Enhanced統合テスト: 改修版モデル使用

o3-modification20250727.mdに基づく改修版モデルのテスト:
1. Enhanced Q-Former (BLIP-2準拠テキスト入力対応)
2. 視覚・言語特徴分離機構
3. LoRAエキスパート統合
4. マルチスケールセグメンテーションヘッド

実行コマンド:
python test_phase3b_enhanced.py
"""

# PyTorchインポート前の環境準備
import sys
import os

# CUDA環境設定
print("🔧 CUDA環境設定...")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

# メモリ最適化設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# CUDA_VISIBLE_DEVICES設定
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 2GPU使用

print("✅ 環境変数設定完了")

# 基本インポート
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from PIL import Image
import numpy as np

# プロジェクト固有のインポート
sys.path.append('.')

# メモリモニタリング
from utils.memory_monitor import GPUMemoryMonitor, memory_monitor_section, log_memory_status, emergency_cleanup

# 改修版モデル
try:
    from model.enhanced_llama4_qformer_sam2 import EnhancedQFormerSegmentationBridge, EnhancedLlamaQFormerSAM2Config
    ENHANCED_MODEL_AVAILABLE = True
    print("✅ Enhanced Model利用可能")
except ImportError as e:
    ENHANCED_MODEL_AVAILABLE = False
    print(f"❌ Enhanced Modelインポート失敗: {e}")
    print("💡 標準モデルでテストを継続します")

# 既存コンポーネント（比較用）
try:
    from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
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


def calculate_iou(pred_mask, gt_mask, smooth=1e-6):
    """IoU計算"""
    pred = pred_mask.float()
    gt = gt_mask.float()
    
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_dice(pred_mask, gt_mask, smooth=1e-6):
    """Dice係数計算"""
    pred = pred_mask.float()
    gt = gt_mask.float()
    
    intersection = (pred * gt).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    
    return dice.item()


def evaluate_segmentation_metrics(pred_masks, gt_masks, num_classes=2):
    """セグメンテーションメトリクス評価"""
    batch_size = pred_masks.shape[0]
    
    ious = []
    dices = []
    
    for b in range(batch_size):
        pred = pred_masks[b].squeeze()
        gt = gt_masks[b].squeeze()
        
        # バイナリマスクに変換
        if pred.dim() > 2:
            pred = pred[0]  # 最初のチャネル
        if gt.dim() > 2:
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
    
    # 改修機能テスト設定
    test_text_input: bool = True          # テキスト入力対応
    test_modality_separation: bool = True  # 視覚・言語分離
    test_lora_experts: bool = True         # LoRAエキスパート
    test_multiscale: bool = True          # マルチスケール
    
    # 評価設定
    num_test_samples: int = 5
    test_modes: List[str] = None
    
    def __post_init__(self):
        """デフォルト値設定"""
        if self.test_modes is None:
            self.test_modes = ['itc', 'itm', 'itg']  # BLIP-2の3モード
        
        # config_linux設定の取得
        lisa_config = config_linux.get_lisa_model_config()
        self.llama_model_id = lisa_config["llama_model_id"]
        self.sam_checkpoint_path = lisa_config["sam_checkpoint_path"]
        self.torch_dtype = lisa_config["torch_dtype"]
        
        logger.info("📋 Enhanced Phase 3Bテスト設定:")
        logger.info(f"  - テキスト入力: {self.test_text_input}")
        logger.info(f"  - モダリティ分離: {self.test_modality_separation}")
        logger.info(f"  - LoRA: {self.test_lora_experts}")
        logger.info(f"  - マルチスケール: {self.test_multiscale}")


class Phase3BEnhancedTest:
    """改修版モデルテストクラス"""
    
    def __init__(self, config: EnhancedTestConfig):
        self.config = config
        self.results = {
            "test_logs": [],
            "performance_metrics": {},
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
            
            # 2. Enhanced Model初期化
            if ENHANCED_MODEL_AVAILABLE:
                logger.info("🚀 Enhanced Model初期化...")
                enhanced_config = EnhancedLlamaQFormerSAM2Config()
                
                self.enhanced_model = EnhancedQFormerSegmentationBridge(
                    config=enhanced_config,
                    shared_llama_model=self.llama4_model,
                    shared_llama_processor=self.llama4_processor,
                    enable_lora=self.config.test_lora_experts,
                    enable_multiscale=self.config.test_multiscale,
                    training_stage=1
                )
                
                logger.info("✅ Enhanced Model初期化完了")
            else:
                logger.warning("⚠️ Enhanced Modelが利用できません - ベースラインモデルを使用")
                if BASELINE_MODEL_AVAILABLE:
                    enhanced_config = LlamaQFormerSAM2Config()
                    self.enhanced_model = QFormerSegmentationBridge(
                        config=enhanced_config,
                        shared_llama_model=self.llama4_model,
                        shared_llama_processor=self.llama4_processor,
                        training_stage=1,
                        enable_moe=False
                    )
                    logger.info("✅ Baseline Modelで代替初期化完了")
                else:
                    raise RuntimeError("利用可能なモデルがありません")
            
            # 3. Baseline Model初期化（比較用）
            if False:  # 比較テストが必要な場合はTrueに
                logger.info("📊 Baseline Model初期化...")
                baseline_config = LlamaQFormerSAM2Config()
                
                self.baseline_model = QFormerSegmentationBridge(
                    config=baseline_config,
                    shared_llama_model=self.llama4_model,
                    shared_llama_processor=self.llama4_processor,
                    training_stage=1,
                    enable_moe=False
                )
                
                logger.info("✅ Baseline Model初期化完了")
            
            # メモリ状況確認
            self.memory_monitor.log_memory_status("モデル初期化完了後")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ モデル初期化エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_test_samples(self, num_samples: int = 5):
        """テストサンプル作成"""
        logger.info(f"📦 {num_samples}個のテストサンプル作成...")
        
        samples = []
        for i in range(num_samples):
            # テスト用ランダム画像（実際の画像データセットが必要）
            # 注意: 本格テストでは実画像データが必要
            image = torch.randn(1, 3, 448, 448).cuda()
            
            # テキスト入力（多様なプロンプト）
            text_prompts = [
                "Segment the cat in this image.",
                "Find and highlight all people in the scene.",
                "Mark the boundaries of the building.",
                "Identify the main object in the foreground.",
                "Segment all vehicles visible in the image."
            ]
            text_input = text_prompts[i % len(text_prompts)]
            
            # テスト用ランダムラベル（実際のアノテーションが必要）
            # 注意: 本格テストでは実アノテーションデータが必要
            labels = torch.randint(0, 2, (448, 448)).cuda()
            
            samples.append({
                'image': image,
                'text_input': text_input,
                'labels': labels,
                'sample_id': i
            })
        
        logger.info(f"✅ {len(samples)}個のテストサンプル作成完了")
        return samples
    
    def test_text_input_modes(self):
        """テキスト入力と学習モードのテスト"""
        logger.info("\n=== テキスト入力モードテスト ===")
        
        if not self.config.test_text_input:
            logger.info("⏭️ テキスト入力テストはスキップされました")
            return
        
        # テストサンプル作成
        samples = self.create_test_samples(3)
        
        # 各モードでテスト
        for mode in self.config.test_modes:
            logger.info(f"\n📚 モード: {mode}")
            
            try:
                # Forward処理
                with torch.no_grad():
                    outputs = self.enhanced_model(
                        images=samples[0]['image'],
                        text_input=[samples[0]['text_input']],
                        labels=samples[0]['labels'],
                        mode=mode
                    )
                
                # 結果確認
                logger.info(f"✅ {mode}モード成功")
                logger.info(f"  - マスク形状: {outputs['masks'].shape}")
                logger.info(f"  - テキストロジット形状: {outputs['text_logits'].shape}")
                logger.info(f"  - 視覚特徴形状: {outputs['visual_features'].shape}")
                
                # Q-Former出力確認
                if 'qformer_outputs' in outputs:
                    qf_out = outputs['qformer_outputs']
                    logger.info(f"  - Q-Formerクエリ: {qf_out['query_embeds'].shape}")
                    logger.info(f"  - LLM埋め込み: {qf_out['llm_embeds'].shape}")
                
                self.results['test_logs'].append({
                    'test': 'text_input_mode',
                    'mode': mode,
                    'status': 'success'
                })
                
            except Exception as e:
                import traceback
                logger.error(f"❌ {mode}モードエラー: {e}")
                logger.error(f"詳細traceback:\n{traceback.format_exc()}")
                self.results['test_logs'].append({
                    'test': 'text_input_mode',
                    'mode': mode,
                    'status': 'error',
                    'error': str(e)
                })
    
    def test_modality_separation(self):
        """視覚・言語分離機構のテスト"""
        logger.info("\n=== 視覚・言語分離機構テスト ===")
        
        if not self.config.test_modality_separation:
            logger.info("⏭️ モダリティ分離テストはスキップされました")
            return
        
        # テストサンプル
        samples = self.create_test_samples(1)
        
        try:
            with torch.no_grad():
                outputs = self.enhanced_model(
                    images=samples[0]['image'],
                    text_input=[samples[0]['text_input']],
                    mode='itg'
                )
            
            # 分離結果確認
            logger.info("✅ モダリティ分離成功")
            
            # 視覚出力
            visual_features = outputs['visual_features']
            logger.info(f"👁️ 視覚特徴:")
            logger.info(f"  - 形状: {visual_features.shape}")
            logger.info(f"  - 次元: {visual_features.shape[-1]} (SAM2互換)")
            
            # 言語出力
            text_logits = outputs['text_logits']
            logger.info(f"💬 言語出力:")
            logger.info(f"  - 形状: {text_logits.shape}")
            logger.info(f"  - 語彙サイズ: {text_logits.shape[-1]}")
            
            # プールされた視覚特徴
            if 'visual_pooled' in outputs:
                logger.info(f"🔄 プール視覚特徴: {outputs['visual_pooled'].shape}")
            
            self.results['test_logs'].append({
                'test': 'modality_separation',
                'status': 'success',
                'visual_dim': visual_features.shape[-1],
                'vocab_size': text_logits.shape[-1]
            })
            
        except Exception as e:
            import traceback
            logger.error(f"❌ モダリティ分離エラー: {e}")
            logger.error(f"詳細traceback:\n{traceback.format_exc()}")
            self.results['test_logs'].append({
                'test': 'modality_separation',
                'status': 'error',
                'error': str(e)
            })
    
    def test_lora_adaptation(self):
        """LoRAエキスパート適応のテスト"""
        logger.info("\n=== LoRAエキスパート適応テスト ===")
        
        if not self.config.test_lora_experts:
            logger.info("⏭️ LoRAテストはスキップされました")
            return
        
        # LoRAモジュールの確認
        lora_count = 0
        lora_params = 0
        
        for name, module in self.enhanced_model.named_modules():
            if 'lora' in name.lower():
                lora_count += 1
                for param in module.parameters():
                    lora_params += param.numel()
        
        logger.info(f"🔧 LoRAモジュール統計:")
        logger.info(f"  - LoRAモジュール数: {lora_count}")
        logger.info(f"  - LoRAパラメータ数: {lora_params:,}")
        
        # テスト推論（LoRA有効）
        samples = self.create_test_samples(1)
        
        try:
            with torch.no_grad():
                outputs = self.enhanced_model(
                    images=samples[0]['image'],
                    text_input=[samples[0]['text_input']],
                    mode='itg'
                )
            
            logger.info("✅ LoRA適応推論成功")
            
            self.results['test_logs'].append({
                'test': 'lora_adaptation',
                'status': 'success',
                'lora_modules': lora_count,
                'lora_params': lora_params
            })
            
        except Exception as e:
            logger.error(f"❌ LoRA適応エラー: {e}")
            self.results['test_logs'].append({
                'test': 'lora_adaptation',
                'status': 'error',
                'error': str(e)
            })
    
    def test_multiscale_segmentation(self):
        """マルチスケールセグメンテーションのテスト"""
        logger.info("\n=== マルチスケールセグメンテーションテスト ===")
        
        if not self.config.test_multiscale:
            logger.info("⏭️ マルチスケールテストはスキップされました")
            return
        
        # テストサンプル
        samples = self.create_test_samples(1)
        
        try:
            with torch.no_grad():
                outputs = self.enhanced_model(
                    images=samples[0]['image'],
                    text_input=[samples[0]['text_input']],
                    labels=samples[0]['labels'],
                    mode='itg'
                )
            
            # マスク品質確認
            masks = outputs['masks']
            iou_scores = outputs['iou_scores']
            
            logger.info("✅ マルチスケールセグメンテーション成功")
            logger.info(f"🎯 マスク統計:")
            logger.info(f"  - 形状: {masks.shape}")
            logger.info(f"  - マスク数: {masks.shape[1]}")
            logger.info(f"  - IoUスコア: {iou_scores.mean().item():.4f}")
            
            # 損失計算（ある場合）
            if 'loss_dict' in outputs:
                loss_dict = outputs['loss_dict']
                logger.info(f"📊 損失内訳:")
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  - {key}: {value.item():.4f}")
            
            self.results['test_logs'].append({
                'test': 'multiscale_segmentation',
                'status': 'success',
                'num_masks': masks.shape[1],
                'avg_iou': iou_scores.mean().item()
            })
            
        except Exception as e:
            logger.error(f"❌ マルチスケールエラー: {e}")
            self.results['test_logs'].append({
                'test': 'multiscale_segmentation',
                'status': 'error',
                'error': str(e)
            })
    
    def test_text_generation(self):
        """テキスト生成機能のテスト"""
        logger.info("\n=== テキスト生成テスト ===")
        
        if not ENHANCED_MODEL_AVAILABLE:
            logger.info("⏭️ テキスト生成テストはEnhanced Modelでのみ利用可能です")
            return
        
        # テストサンプル
        samples = self.create_test_samples(1)
        
        try:
            # テキスト生成（Enhanced Modelの機能）
            if hasattr(self.enhanced_model, 'generate_text'):
                generated_text = self.enhanced_model.generate_text(
                    images=samples[0]['image'],
                    text_prompt="What objects are visible in this image?",
                    max_length=50,
                    temperature=0.7
                )
                
                logger.info("✅ テキスト生成成功")
                logger.info(f"📝 生成テキスト: {generated_text}")
                
                self.results['test_logs'].append({
                    'test': 'text_generation',
                    'status': 'success',
                    'generated_text': generated_text
                })
            else:
                logger.info("⏭️ 現在のモデルはテキスト生成機能をサポートしていません")
                self.results['test_logs'].append({
                    'test': 'text_generation',
                    'status': 'skipped',
                    'reason': 'function_not_available'
                })
            
        except Exception as e:
            logger.error(f"❌ テキスト生成エラー: {e}")
            self.results['test_logs'].append({
                'test': 'text_generation',
                'status': 'error',
                'error': str(e)
            })
    
    def run_performance_comparison(self):
        """性能比較テスト（ベースラインがある場合）"""
        logger.info("\n=== 性能比較テスト ===")
        
        if self.baseline_model is None:
            logger.info("⏭️ ベースラインモデルがないため比較テストはスキップ")
            return
        
        # テストサンプル
        samples = self.create_test_samples(self.config.num_test_samples)
        
        enhanced_metrics = {'iou': [], 'dice': [], 'time': []}
        baseline_metrics = {'iou': [], 'dice': [], 'time': []}
        
        for sample in samples:
            # Enhanced Model
            start_time = time.time()
            with torch.no_grad():
                enhanced_outputs = self.enhanced_model(
                    images=sample['image'],
                    text_input=[sample['text_input']],
                    labels=sample['labels'],
                    mode='itg'
                )
            enhanced_time = time.time() - start_time
            
            # メトリクス計算
            if 'masks' in enhanced_outputs:
                masks = enhanced_outputs['masks'][:, 0:1] > 0.5
                metrics = evaluate_segmentation_metrics(
                    masks,
                    sample['labels'].unsqueeze(0).unsqueeze(0),
                    num_classes=2
                )
                enhanced_metrics['iou'].append(metrics['mIoU'])
                enhanced_metrics['dice'].append(metrics['mDice'])
                enhanced_metrics['time'].append(enhanced_time)
        
        # 平均計算
        avg_enhanced = {
            'mIoU': np.mean(enhanced_metrics['iou']),
            'mDice': np.mean(enhanced_metrics['dice']),
            'avg_time': np.mean(enhanced_metrics['time'])
        }
        
        logger.info("📊 Enhanced Model性能:")
        logger.info(f"  - mIoU: {avg_enhanced['mIoU']:.4f}")
        logger.info(f"  - mDice: {avg_enhanced['mDice']:.4f}")
        logger.info(f"  - 平均推論時間: {avg_enhanced['avg_time']:.3f}秒")
        
        self.results['performance_metrics'] = avg_enhanced
    
    def save_results(self):
        """結果保存"""
        results_path = Path(self.config.output_dir) / "test_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"💾 結果保存完了: {results_path}")
        
        # サマリー表示
        logger.info("\n=== テストサマリー ===")
        
        success_count = sum(1 for log in self.results['test_logs'] 
                          if log['status'] == 'success')
        total_count = len(self.results['test_logs'])
        
        logger.info(f"✅ 成功: {success_count}/{total_count}")
        
        for log in self.results['test_logs']:
            status_emoji = '✅' if log['status'] == 'success' else '❌'
            logger.info(f"{status_emoji} {log['test']}: {log['status']}")
    
    def run_all_tests(self):
        """全テスト実行"""
        logger.info("\n" + "="*60)
        logger.info("🚀 Enhanced Phase 3Bテスト開始")
        logger.info("="*60)
        
        # モデル初期化
        if not self.setup_models():
            logger.error("❌ モデル初期化に失敗しました")
            return
        
        # 各種テスト実行
        self.test_text_input_modes()
        self.test_modality_separation()
        self.test_lora_adaptation()
        self.test_multiscale_segmentation()
        self.test_text_generation()
        self.run_performance_comparison()
        
        # 結果保存
        self.save_results()
        
        # メモリ状況最終確認
        summary = self.memory_monitor.get_memory_summary()
        logger.info("\n📊 メモリ使用状況サマリー:")
        logger.info(f"  - 最大GPU使用量: {summary.get('max_gpu_memory_gb', 0):.1f}GB")
        logger.info(f"  - 平均GPU使用量: {summary.get('avg_gpu_memory_gb', 0):.1f}GB")
        
        logger.info("\n✅ Enhanced Phase 3Bテスト完了!")


def main():
    """メイン実行関数"""
    # テスト設定
    config = EnhancedTestConfig(
        test_text_input=True,
        test_modality_separation=True,
        test_lora_experts=True,
        test_multiscale=True,
        num_test_samples=5
    )
    
    # テスト実行
    tester = Phase3BEnhancedTest(config)
    tester.run_all_tests()


if __name__ == "__main__":
    main()