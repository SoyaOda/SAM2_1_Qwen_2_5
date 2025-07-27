#!/usr/bin/env python3
"""
LISA-Llama4統合モデル 単独テストスクリプト（シングルエンコーダー構成対応版）
test_llama4_standalone.pyを参考に、LISA統合モデル特有の機能もテスト

実行方法:
Lambda Cloud:
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip_address>:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip_address> "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 python -u test_llama4_lisa_standalone.py 2>&1"
"""

import os
import sys
import logging
import torch
import traceback
from PIL import Image
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LISA-Llama4モデルとLoRA設定をインポート
from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
from peft import LoraConfig, TaskType
import config_linux

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_llama4_lisa_standalone.log')
    ]
)
logger = logging.getLogger(__name__)

class LisaLlama4StandaloneTest:
    """LISA-Llama4統合モデルの単独テスト"""
    
    def __init__(self):
        self.model = None
        self.test_results = {}
        
    def run_full_test(self):
        """完全テストの実行"""
        logger.info("=== LISA-Llama4統合モデル 単独テスト開始 ===")
        start_time = datetime.now()
        
        try:
            # テスト1: モデル初期化
            if not self.test_model_initialization():
                return False
            
            # テスト2: パラメータ分析
            if not self.test_parameter_analysis():
                return False
            
            # テスト3: LoRA適用
            if not self.test_lora_application():
                return False
            
            # テスト4: GPU分散確認
            if not self.test_gpu_distribution():
                return False
            
            # テスト5: マルチモーダル入力準備
            if not self.test_multimodal_input():
                return False
            
            # テスト6: 順伝播テスト
            if not self.test_forward_pass():
                return False
            
            # テスト7: SAM機能テスト（SAMが利用可能な場合）
            if not self.test_sam_functionality():
                return False
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("✅ 全テスト完了")
            logger.info(f"総実行時間: {duration:.2f}秒")
            self.print_test_summary()
            return True
            
        except Exception as e:
            logger.error(f"❌ テスト実行エラー: {e}")
            traceback.print_exc()
            return False
    
    def test_model_initialization(self):
        """テスト1: モデル初期化"""
        logger.info("=== テスト1: モデル初期化 ===")
        
        try:
            # LISA-Llama4設定作成（config_linux統一設定を使用）
            config = LisaLlama4Config(**config_linux.get_lisa_model_config())
            
            logger.info("LISA-Llama4モデル初期化開始...")
            self.model = LisaLlama4ForCausalLM(config)
            
            # 初期化確認
            assert self.model is not None, "モデルが初期化されていません"
            assert hasattr(self.model, 'llama_model'), "Llama4モデルが存在しません"
            assert hasattr(self.model, 'multi_modal_projector'), "プロジェクタが存在しません"
            
            logger.info("✅ モデル初期化成功")
            self.test_results["model_initialization"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ モデル初期化失敗: {e}")
            self.test_results["model_initialization"] = False
            return False
    
    def test_parameter_analysis(self):
        """テスト2: パラメータ分析"""
        logger.info("=== テスト2: パラメータ分析 ===")
        
        try:
            # パラメータ統計
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            logger.info(f"総パラメータ数: {total_params:,}")
            logger.info(f"学習可能パラメータ: {trainable_params:,}")
            logger.info(f"凍結パラメータ: {frozen_params:,}")
            logger.info(f"学習可能割合: {100 * trainable_params / total_params:.6f}%")
            
            # パラメータカテゴリ分析
            categories = {
                'llama_model': 0,
                'multi_modal_projector': 0,
                'sam_model': 0,
                'others': 0
            }
            
            for name, param in self.model.named_parameters():
                param_count = param.numel()
                if 'llama_model' in name:
                    categories['llama_model'] += param_count
                elif 'multi_modal_projector' in name:
                    categories['multi_modal_projector'] += param_count
                elif 'sam_model' in name:
                    categories['sam_model'] += param_count
                else:
                    categories['others'] += param_count
            
            logger.info("パラメータカテゴリ分析:")
            for category, count in categories.items():
                percentage = 100 * count / total_params if total_params > 0 else 0
                logger.info(f"  {category}: {count:,} ({percentage:.2f}%)")
            
            # プロジェクタパラメータの学習可能性確認
            projector_trainable = sum(p.numel() for p in self.model.multi_modal_projector.parameters() if p.requires_grad)
            logger.info(f"プロジェクタ学習可能パラメータ: {projector_trainable:,}")
            
            self.test_results["parameter_analysis"] = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "categories": categories,
                "projector_trainable": projector_trainable
            }
            
            logger.info("✅ パラメータ分析完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ パラメータ分析失敗: {e}")
            self.test_results["parameter_analysis"] = False
            return False
    
    def test_lora_application(self):
        """テスト3: LoRA適用"""
        logger.info("=== テスト3: LoRA適用 ===")
        
        try:
            # LoRA設定（config_linux統一設定を使用）
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **config_linux.get_lora_config()
            )
            
            logger.info("LoRA設定適用中...")
            self.model = self.model.apply_lora_configuration(lora_config)
            
            # LoRA適用後のパラメータ統計
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            percentage = 100 * trainable_params / total_params if total_params > 0 else 0
            
            logger.info(f"LoRA適用後 - 学習可能パラメータ: {trainable_params:,} ({percentage:.3f}%)")
            
            # 期待値チェック: 学習可能パラメータが適切な範囲内か
            if 0.01 <= percentage <= 5.0:  # 0.01% - 5.0%の範囲
                logger.info("✅ 学習可能パラメータ割合が適切です")
            else:
                logger.warning(f"⚠️ 学習可能パラメータ割合が予想外: {percentage:.3f}%")
            
            self.test_results["lora_application"] = {
                "success": True,
                "trainable_params": trainable_params,
                "percentage": percentage
            }
            
            logger.info("✅ LoRA適用完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ LoRA適用失敗: {e}")
            self.test_results["lora_application"] = False
            return False
    
    def test_gpu_distribution(self):
        """テスト4: GPU分散確認"""
        logger.info("=== テスト4: GPU分散確認 ===")
        
        try:
            # GPU利用可能性確認
            if not torch.cuda.is_available():
                logger.warning("CUDA利用不可 - CPUで継続")
                return True
            
            device_count = torch.cuda.device_count()
            logger.info(f"利用可能GPU数: {device_count}")
            
            # GPU分散状況確認
            if hasattr(self.model.llama_model, 'hf_device_map'):
                device_map = self.model.llama_model.hf_device_map
                logger.info("GPU分散状況:")
                for module, device in device_map.items():
                    logger.info(f"  {module}: {device}")
                
                # GPU使用状況確認
                gpu_usage = {}
                for device in device_map.values():
                    if isinstance(device, int) and device >= 0:
                        if device not in gpu_usage:
                            gpu_usage[device] = []
                        gpu_usage[device].append(module)
                
                logger.info("GPU別モジュール配置:")
                for gpu_id, modules in gpu_usage.items():
                    logger.info(f"  GPU {gpu_id}: {len(modules)}個のモジュール")
            
            # GPU メモリ使用量確認
            for i in range(min(device_count, 8)):  # 最大8GPU確認
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    logger.info(f"GPU {i}: 割当 {allocated:.2f}GB, 予約 {reserved:.2f}GB")
                except:
                    pass
            
            self.test_results["gpu_distribution"] = True
            logger.info("✅ GPU分散確認完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU分散確認失敗: {e}")
            self.test_results["gpu_distribution"] = False
            return False
    
    def test_multimodal_input(self):
        """テスト5: マルチモーダル入力準備"""
        logger.info("=== テスト5: マルチモーダル入力準備 ===")
        
        try:
            # テスト用画像作成
            test_image = Image.new('RGB', (224, 224), color='red')
            test_prompt = "この画像を説明してください。"
            
            logger.info("マルチモーダル入力準備中...")
            inputs = self.model.prepare_multimodal_input(
                image=test_image,
                text_prompt=test_prompt,
                for_training=False
            )
            
            # 入力確認
            assert isinstance(inputs, dict), "入力が辞書形式ではありません"
            logger.info(f"入力キー: {list(inputs.keys())}")
            
            if 'input_ids' in inputs:
                logger.info(f"input_ids shape: {inputs['input_ids'].shape}")
            if 'sam_pixel_values' in inputs:
                logger.info(f"sam_pixel_values shape: {inputs['sam_pixel_values'].shape}")
            if 'attention_mask' in inputs:
                logger.info(f"attention_mask shape: {inputs['attention_mask'].shape}")
            
            # シングルエンコーダー構成の確認
            if 'pixel_values' in inputs:
                logger.warning("⚠️ pixel_valuesが存在します（シングルエンコーダーでは不要）")
            if 'sam_pixel_values' not in inputs:
                logger.warning("⚠️ sam_pixel_valuesが存在しません（シングルエンコーダーでは必須）")
            
            self.test_results["multimodal_input"] = {
                "success": True,
                "input_keys": list(inputs.keys()),
                "input_shapes": {k: v.shape if hasattr(v, 'shape') else str(type(v)) 
                               for k, v in inputs.items()}
            }
            
            logger.info("✅ マルチモーダル入力準備完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ マルチモーダル入力準備失敗: {e}")
            self.test_results["multimodal_input"] = False
            return False
    
    def test_forward_pass(self):
        """テスト6: 順伝播テスト"""
        logger.info("=== テスト6: 順伝播テスト ===")
        
        try:
            # テスト用データ準備
            test_image = Image.new('RGB', (224, 224), color='blue')
            test_prompt = "この画像について教えてください。"
            
            logger.info("順伝播実行中...")
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.forward(
                    image=test_image,
                    text_prompt=test_prompt,
                    generate_mask=False  # SAMなしの場合はマスク生成無効
                )
            
            # 出力確認
            assert isinstance(outputs, dict), "出力が辞書形式ではありません"
            logger.info(f"出力キー: {list(outputs.keys())}")
            
            # 基本出力の確認
            if 'logits' in outputs:
                logits = outputs['logits']
                logger.info(f"logits shape: {logits.shape}")
                logger.info(f"logits type: {type(logits)}")
            
            if 'loss' in outputs:
                loss = outputs['loss']
                logger.info(f"loss: {loss}")
                logger.info(f"loss type: {type(loss)}")
            
            # シングルエンコーダー構成の確認
            if 'sam_image_embeddings' in outputs:
                logger.info(f"✅ SAM画像埋め込み生成: {outputs['sam_image_embeddings'].shape}")
            
            self.test_results["forward_pass"] = {
                "success": True,
                "output_keys": list(outputs.keys()),
                "output_info": {k: {"shape": v.shape if hasattr(v, 'shape') else None,
                                   "type": type(v).__name__} 
                               for k, v in outputs.items()}
            }
            
            logger.info("✅ 順伝播テスト完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ 順伝播テスト失敗: {e}")
            traceback.print_exc()
            self.test_results["forward_pass"] = False
            return False
    
    def test_sam_functionality(self):
        """テスト7: SAM機能テスト（利用可能な場合のみ）"""
        logger.info("=== テスト7: SAM機能テスト ===")
        
        try:
            # SAM機能の利用可能性確認
            has_sam = self.model.has_sam_capability()
            logger.info(f"SAM機能利用可能: {has_sam}")
            
            if not has_sam:
                logger.info("SAM機能が無効のため、テストをスキップします")
                self.test_results["sam_functionality"] = {"skipped": True, "reason": "SAM not available"}
                return True
            
            # SAM付きテスト（実際にSAMが利用可能な場合）
            test_image = Image.new('RGB', (1024, 768), color='green')  # 異なるサイズでテスト
            test_prompt = "この画像の中心部分を[SEG]してください。"
            
            logger.info("SAM機能付き順伝播実行中...")
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.forward(
                    image=test_image,
                    text_prompt=test_prompt,
                    generate_mask=True
                )
            
            # SAM出力確認
            if 'pred_masks' in outputs:
                masks = outputs['pred_masks']
                logger.info(f"予測マスク shape: {masks.shape if masks is not None else 'None'}")
                
                # シングルエンコーダー構成での動作確認
                if masks is not None:
                    # マスクが元の画像サイズにリサイズされているか確認
                    expected_h, expected_w = test_image.height, test_image.width
                    if len(masks.shape) >= 2:
                        mask_h, mask_w = masks.shape[-2:]  # 最後の2次元を取得
                        if mask_h == expected_h and mask_w == expected_w:
                            logger.info(f"✅ マスクが元の画像サイズにリサイズされています: {mask_h}x{mask_w}")
                        else:
                            logger.warning(f"⚠️ マスクサイズが元画像と異なります: {mask_h}x{mask_w} (期待値: {expected_h}x{expected_w})")
            
            self.test_results["sam_functionality"] = {
                "success": True,
                "has_sam": has_sam,
                "mask_generated": 'pred_masks' in outputs and outputs['pred_masks'] is not None,
                "single_encoder_mode": True
            }
            
            logger.info("✅ SAM機能テスト完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ SAM機能テスト失敗: {e}")
            self.test_results["sam_functionality"] = False
            return False
    
    def print_test_summary(self):
        """テスト結果サマリ表示"""
        logger.info("\n=== テスト結果サマリ ===")
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            total_tests += 1
            status = "✅ PASS" if result and result != False else "❌ FAIL"
            if isinstance(result, dict) and result.get("skipped"):
                status = "⏭️ SKIP"
            else:
                passed_tests += 1 if result and result != False else passed_tests
            
            logger.info(f"{test_name}: {status}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"\n成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 85:
            logger.info("🎉 テスト成功！LISA-Llama4統合モデルは正常に動作しています")
        else:
            logger.warning("⚠️ 一部テストに失敗しています。詳細を確認してください")


def main():
    """メイン実行関数"""
    logger.info("LISA-Llama4統合モデル 単独テスト開始")
    
    # CUDA環境確認
    logger.info(f"CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU数: {torch.cuda.device_count()}")
        for i in range(min(torch.cuda.device_count(), 8)):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # テスト実行
    test_runner = LisaLlama4StandaloneTest()
    success = test_runner.run_full_test()
    
    if success:
        logger.info("🎉 全テスト完了 - 成功")
        sys.exit(0)
    else:
        logger.error("❌ テスト失敗")
        sys.exit(1)


if __name__ == "__main__":
    main() 