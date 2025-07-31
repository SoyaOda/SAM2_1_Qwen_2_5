#!/usr/bin/env python3
"""
LISA-Llama4-Scout + Q-Former + SAM2 統合モデル 学習能力検証スクリプト

model/llama4_qformer_sam2.py と model/losses_qformer_sam2.py の過学習テスト

実行方法:
Lambda Cloud:
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip_address>:/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip_address> "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u test/overfit_llama4_qformer_sam2_batch.py 2>&1"
"""

import sys
import os
import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話モードに設定
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# PEFT関連
from peft import LoraConfig, get_peft_model, TaskType

# プロジェクト固有のインポート
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
from model.losses_qformer_sam2 import get_composite_loss_qformer_sam2
import config_linux

# ロギング設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QFormerSAM2OverfitConfig:
    """LISA-QFormer-SAM2統合モデル過学習テスト用設定"""
    # 学習固有設定
    max_length: int = 256
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # バッチ・メモリ設定
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 出力設定
    output_dir: str = "test/overfit_results_qformer_sam2"
    save_steps: int = 5
    logging_steps: int = 1
    
    # 学習段階設定
    training_stage: int = 2  # Stage 2: フルスタック適応

class QFormerSAM2OverfitTester:
    """LISA-QFormer-SAM2統合モデルの過学習テスト実行クラス"""
    
    def __init__(self, config: QFormerSAM2OverfitConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.loss_history = []
        self.device = None
        
        # 出力ディレクトリ作成
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def setup_model(self) -> bool:
        """モデルとオプティマイザーの設定"""
        logger.info("=== モデル設定開始 ===")
        
        try:
            # デバイス設定
            if torch.cuda.is_available():
                self.device = "cuda"
                device_count = torch.cuda.device_count()
                logger.info(f"🔧 CUDA利用可能: {device_count}個のGPU")
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            else:
                self.device = "cpu"
                logger.warning("⚠️ CUDA利用不可、CPUモードで実行")
            
            # モデル初期化
            logger.info("🏗️ QFormer-SAM2モデル初期化中...")
            model_config = LlamaQFormerSAM2Config()
            
            self.model = QFormerSegmentationBridge(
                config=model_config, 
                training_stage=self.config.training_stage
            )
            
            # LoRA設定適用
            logger.info("🔧 LoRA設定適用中...")
            lora_config = LoraConfig(
                r=config_linux.LORA_R,
                lora_alpha=config_linux.LORA_ALPHA,
                target_modules=config_linux.LORA_TARGET_MODULES,
                lora_dropout=config_linux.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # LoRA適用（Llama-4部分のみ）
            if hasattr(self.model, 'llama_model'):
                self.model.llama_model = get_peft_model(self.model.llama_model, lora_config)
                logger.info("✅ LoRA適用完了 (Llama-4部分)")
            
            # パラメータ統計
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"📊 モデル統計:")
            logger.info(f"  - 総パラメータ数: {total_params:,}")
            logger.info(f"  - 学習可能パラメータ数: {trainable_params:,}")
            logger.info(f"  - 学習可能比率: {trainable_params/total_params*100:.2f}%")
            
            # オプティマイザー設定
            logger.info("⚙️ オプティマイザー設定中...")
            self.optimizer = optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            logger.info("✅ モデル設定完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ モデル設定失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """合成データセット作成（過学習用）"""
        logger.info("📊 合成データセット作成中...")
        
        # config統一設定を使用
        test_config = config_linux.get_test_config()
        batch_size = self.config.batch_size
        image_size = test_config['image_size']      # config統一: 448
        mask_size = test_config['mask_size']        # config統一: 448
        seq_len = 64
        
        # パターン1: 格子パターン画像（BFloat16統一）
        images = torch.zeros(batch_size, 3, image_size, image_size, dtype=torch.bfloat16)
        
        # チェッカーボードパターン
        checker_size = 32
        for i in range(0, image_size, checker_size):
            for j in range(0, image_size, checker_size):
                if (i // checker_size + j // checker_size) % 2 == 0:
                    images[:, :, i:i+checker_size, j:j+checker_size] = 255.0
        
        # ランダムノイズ追加（BFloat16対応）
        noise = torch.randn_like(images) * 10
        images = torch.clamp(images + noise, 0, 255)
        
        # セグメンテーション指示テキスト
        seg_text = "この画像の白い四角形を[SEG]してください"
        
        # ダミートークナイゼーション（SEGトークン含む）
        input_ids = torch.randint(1, 32000, (batch_size, seq_len))
        # SEGトークンの位置をマーク（ID=50000と仮定）
        input_ids[:, 32] = 50000  # 中央にSEGトークン配置
        
        attention_mask = torch.ones(batch_size, seq_len)
        
        # 正解マスク: 白い四角形領域（BFloat16統一）
        target_masks = torch.zeros(batch_size, mask_size, mask_size, dtype=torch.bfloat16)
        
        # 格子パターンに対応するマスク作成
        mask_checker_size = mask_size * checker_size // image_size
        for i in range(0, mask_size, mask_checker_size):
            for j in range(0, mask_size, mask_checker_size):
                if (i // mask_checker_size + j // mask_checker_size) % 2 == 0:
                    end_i = min(i + mask_checker_size, mask_size)
                    end_j = min(j + mask_checker_size, mask_size)
                    target_masks[:, i:end_i, j:end_j] = 1.0
        
        # ガウシアンブラーで滑らかにする（BFloat16対応）
        from scipy.ndimage import gaussian_filter
        for b in range(batch_size):
            # Float32で処理してからBFloat16に変換
            mask_float32 = target_masks[b].float()
            blurred = gaussian_filter(mask_float32.numpy(), sigma=2.0)
            target_masks[b] = torch.from_numpy(blurred).to(torch.bfloat16)
            target_masks[b] = (target_masks[b] > 0.5).to(torch.bfloat16)
        
        logger.info(f"  画像: {images.shape} (範囲: {images.min():.1f}-{images.max():.1f})")
        logger.info(f"  Input IDs: {input_ids.shape}")
        logger.info(f"  正解マスク: {target_masks.shape} (正例率: {target_masks.mean().item():.3f})")
        
        return images, input_ids, attention_mask, target_masks
    
    def train_single_epoch(
        self, 
        images: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        target_masks: torch.Tensor,
        epoch: int
    ) -> Dict[str, float]:
        """単一エポックの学習"""
        
        self.model.train()
        epoch_losses = {}
        
        # 2025年ベストプラクティス: Model Parallelismに対応したデバイス移動
        # メインデバイス取得（train_llama4_lisa_single_process.pyパターン）
        main_device = None
        if hasattr(self.model, 'llama_model') and hasattr(self.model.llama_model, 'parameters'):
            main_device = next(self.model.llama_model.parameters()).device
        else:
            main_device = next(self.model.parameters()).device
            
        # 統一デバイス移動
        images = images.to(main_device)
        input_ids = input_ids.to(main_device)
        attention_mask = attention_mask.to(main_device)
        target_masks = target_masks.to(main_device)
        
        # デバイス確認ログ（初回のみ）
        if epoch == 0:
            logger.info(f"  📍 デバイス統一: {main_device}")
            logger.info(f"    - images: {images.device}")
            logger.info(f"    - target_masks: {target_masks.device}")
        
        # 勾配リセット
        self.optimizer.zero_grad()
        
        try:
            # フォワードパス
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # 損失計算
            if hasattr(self.model, 'compute_loss'):
                losses = self.model.compute_loss(
                    predicted_masks=outputs['predicted_masks'],
                    target_masks=target_masks,
                    query_embeds=outputs.get('query_embeddings'),
                    text_embeds=outputs.get('text_embeds'),
                    sam_prompts=outputs.get('sam_prompts')
                )
                total_loss = losses['total_loss']
                
                # 各損失をepoch_lossesに記録
                for loss_name, loss_value in losses.items():
                    epoch_losses[loss_name] = loss_value.item()
                    
            else:
                # フォールバック損失
                predicted_masks = outputs['predicted_masks']
                if predicted_masks.dim() == 4:
                    predicted_masks = predicted_masks[:, 0]  # 最初のクエリ
                
                total_loss = nn.functional.binary_cross_entropy_with_logits(
                    predicted_masks, target_masks
                )
                epoch_losses = {'total_loss': total_loss.item(), 'bce_loss': total_loss.item()}
            
            # バックワードパス
            total_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            # オプティマイザーステップ
            self.optimizer.step()
            
            # IoU計算（評価用）
            with torch.no_grad():
                predicted_masks = outputs['predicted_masks']
                if predicted_masks.dim() == 4:
                    # 複数クエリの場合、平均IoUを計算
                    ious = []
                    for i in range(predicted_masks.size(1)):
                        pred_i = torch.sigmoid(predicted_masks[:, i])
                        pred_binary = (pred_i > 0.5).float()
                        
                        intersection = (pred_binary * target_masks).sum()
                        union = (pred_binary + target_masks).clamp(0, 1).sum()
                        iou = intersection / (union + 1e-8)
                        ious.append(iou.item())
                    
                    epoch_losses['mean_iou'] = np.mean(ious)
                    epoch_losses['max_iou'] = np.max(ious)
                else:
                    pred_binary = (torch.sigmoid(predicted_masks) > 0.5).float()
                    intersection = (pred_binary * target_masks).sum()
                    union = (pred_binary + target_masks).clamp(0, 1).sum()
                    epoch_losses['iou'] = (intersection / (union + 1e-8)).item()
            
        except Exception as e:
            logger.error(f"❌ エポック {epoch} でエラー: {e}")
            epoch_losses = {'total_loss': float('inf'), 'error': str(e)}
        
        return epoch_losses
    
    def run_overfit_test(self) -> bool:
        """過学習テストの実行"""
        logger.info("=== LISA-QFormer-SAM2 過学習テスト開始 ===")
        
        try:
            # モデル設定
            if not self.setup_model():
                return False
            
            # 合成データ作成
            images, input_ids, attention_mask, target_masks = self.create_synthetic_data()
            
            # 学習ループ
            logger.info(f"🚀 {self.config.num_epochs}エポック学習開始...")
            
            for epoch in range(self.config.num_epochs):
                # 学習実行
                epoch_losses = self.train_single_epoch(
                    images, input_ids, attention_mask, target_masks, epoch
                )
                
                # 損失履歴記録
                epoch_losses['epoch'] = epoch
                self.loss_history.append(epoch_losses)
                
                # ログ出力
                if epoch % self.config.logging_steps == 0:
                    loss_str = ", ".join([
                        f"{k}: {v:.6f}" for k, v in epoch_losses.items() 
                        if k != 'epoch' and isinstance(v, (int, float))
                    ])
                    logger.info(f"エポック {epoch:2d}: {loss_str}")
                
                # 中間保存
                if epoch % self.config.save_steps == 0:
                    self.save_intermediate_results(epoch)
                
                # メモリクリーンアップ
                if epoch % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 最終結果保存
            self.save_final_results()
            
            # 学習成功判定
            success = self.evaluate_learning_success()
            
            if success:
                logger.info("✅ 過学習テスト成功: モデルは学習能力を持っています")
            else:
                logger.warning("⚠️ 過学習テスト部分成功: 学習はしているが改善の余地があります")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 過学習テスト失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_intermediate_results(self, epoch: int):
        """中間結果保存"""
        try:
            # 損失グラフ
            self.plot_loss_curves(epoch)
            
            # 数値データ
            results_path = Path(self.config.output_dir) / f"intermediate_epoch_{epoch}.json"
            with open(results_path, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ 中間結果保存失敗 (エポック {epoch}): {e}")
    
    def save_final_results(self):
        """最終結果保存"""
        try:
            # 最終損失グラフ
            self.plot_loss_curves(final=True)
            
            # 全損失履歴
            final_path = Path(self.config.output_dir) / "final_results.json"
            with open(final_path, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
            
            # 学習サマリー
            summary = self.generate_learning_summary()
            summary_path = Path(self.config.output_dir) / "learning_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"📁 最終結果保存完了: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ 最終結果保存失敗: {e}")
    
    def plot_loss_curves(self, epoch: int = None, final: bool = False):
        """損失曲線プロット"""
        try:
            if not self.loss_history:
                return
            
            epochs = [h['epoch'] for h in self.loss_history]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"LISA-QFormer-SAM2 Learning Progress {'(Final)' if final else f'(Epoch {epoch})'}")
            
            # 総損失
            total_losses = [h.get('total_loss', 0) for h in self.loss_history]
            axes[0, 0].plot(epochs, total_losses, 'b-', linewidth=2)
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # IoU
            ious = [h.get('mean_iou', h.get('iou', 0)) for h in self.loss_history]
            axes[0, 1].plot(epochs, ious, 'g-', linewidth=2)
            axes[0, 1].set_title('IoU Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('IoU')
            axes[0, 1].grid(True)
            
            # セグメンテーション損失
            seg_losses = [h.get('focal_tversky_loss', h.get('bce_loss', 0)) for h in self.loss_history]
            axes[1, 0].plot(epochs, seg_losses, 'r-', linewidth=2)
            axes[1, 0].set_title('Segmentation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            
            # Q-Former損失
            qformer_losses = [h.get('qformer_itc_loss', 0) for h in self.loss_history]
            if any(l > 0 for l in qformer_losses):
                axes[1, 1].plot(epochs, qformer_losses, 'm-', linewidth=2)
                axes[1, 1].set_title('Q-Former ITC Loss')
            else:
                axes[1, 1].text(0.5, 0.5, 'Q-Former Loss\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            filename = "final_learning_curves.png" if final else f"learning_curves_epoch_{epoch}.png"
            save_path = Path(self.config.output_dir) / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"⚠️ プロット作成失敗: {e}")
    
    def generate_learning_summary(self) -> Dict[str, Any]:
        """学習サマリー生成"""
        if not self.loss_history:
            return {}
        
        # 最初と最後の損失
        first_loss = self.loss_history[0].get('total_loss', float('inf'))
        last_loss = self.loss_history[-1].get('total_loss', float('inf'))
        
        # 最良IoU
        ious = [h.get('mean_iou', h.get('iou', 0)) for h in self.loss_history]
        best_iou = max(ious) if ious else 0
        
        # 損失減少率
        loss_reduction = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0
        
        summary = {
            'training_stage': self.config.training_stage,
            'num_epochs': self.config.num_epochs,
            'learning_rate': self.config.learning_rate,
            'first_loss': first_loss,
            'last_loss': last_loss,
            'loss_reduction_percent': loss_reduction,
            'best_iou': best_iou,
            'converged': loss_reduction > 50,  # 50%以上減少で収束と判定
            'learning_stable': len([h for h in self.loss_history[-5:] 
                                  if h.get('total_loss', float('inf')) < first_loss * 0.8]) >= 3
        }
        
        return summary
    
    def evaluate_learning_success(self) -> bool:
        """学習成功の評価"""
        if not self.loss_history:
            return False
        
        summary = self.generate_learning_summary()
        
        # 成功基準
        criteria = [
            summary.get('loss_reduction_percent', 0) > 30,  # 30%以上の損失減少
            summary.get('best_iou', 0) > 0.1,               # IoU 0.1以上達成
            summary.get('learning_stable', False),          # 安定した学習
        ]
        
        success_rate = sum(criteria) / len(criteria)
        
        logger.info(f"📊 学習成功評価:")
        logger.info(f"  - 損失減少: {summary.get('loss_reduction_percent', 0):.1f}% {'✅' if criteria[0] else '❌'}")
        logger.info(f"  - 最良IoU: {summary.get('best_iou', 0):.3f} {'✅' if criteria[1] else '❌'}")
        logger.info(f"  - 学習安定性: {'✅' if criteria[2] else '❌'}")
        logger.info(f"  - 総合成功率: {success_rate:.1%}")
        
        return success_rate >= 0.67  # 3分の2以上の基準クリアで成功

def main():
    """メイン実行関数"""
    print("=== LISA-Llama4-QFormer-SAM2 過学習テスト ===")
    
    config = QFormerSAM2OverfitConfig()
    
    logger.info(f"📋 テスト設定:")
    logger.info(f"  - エポック数: {config.num_epochs}")
    logger.info(f"  - 学習率: {config.learning_rate}")
    logger.info(f"  - バッチサイズ: {config.batch_size}")
    logger.info(f"  - 学習段階: Stage {config.training_stage}")
    logger.info(f"  - 出力ディレクトリ: {config.output_dir}")
    
    tester = QFormerSAM2OverfitTester(config)
    success = tester.run_overfit_test()
    
    if success:
        logger.info("✅ 過学習テスト完了 - 成功")
        return 0
    else:
        logger.error("❌ 過学習テスト完了 - 失敗")
        return 1

if __name__ == "__main__":
    exit(main())