#!/usr/bin/env python3
"""
LISA-Llama4-Scout + Q-Former + SAM2 統合モデル 単独テストスクリプト

model/llama4_qformer_sam2.py と model/losses_qformer_sam2.py のテスト

実行方法:
Lambda Cloud:
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip_address>:/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip_address> "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u test/test_llama4_qformer_sam2_standalone.py 2>&1"
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LISA-Llama4-QFormer-SAM2モデルと損失関数をインポート
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config, create_lisa_model
from model.losses_qformer_sam2 import get_composite_loss_qformer_sam2, test_composite_loss
import config_linux

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test/test_llama4_qformer_sam2_standalone.log')
    ]
)
logger = logging.getLogger(__name__)

class LlamaQFormerSAM2StandaloneTest:
    """LISA-Llama4-QFormer-SAM2統合モデルの単独テスト"""
    
    def __init__(self):
        self.model = None
        self.loss_function = None
        self.test_results = {}
        
    def run_full_test(self):
        """完全テストの実行"""
        logger.info("=== LISA-Llama4-QFormer-SAM2統合モデル 単独テスト開始 ===")
        
        try:
            # 1. GPU環境確認
            self._test_gpu_environment()
            
            # 2. 損失関数テスト（モデル初期化前）
            self._test_loss_functions()
            
            # 3. モデル初期化テスト
            self._test_model_initialization()
            
            # 4. モデル情報表示テスト
            self._test_model_info()
            
            # 5. ダミーデータテスト
            self._test_dummy_forward()
            
            # 6. テストデータ生成とフォワードパス
            self._test_realistic_forward()
            
            # 7. 損失計算テスト
            self._test_loss_computation()
            
            # 8. メモリ使用量テスト
            self._test_memory_usage()
            
            # 9. デバイス配置テスト
            self._test_device_placement()
            
            self._print_test_summary()
            
        except Exception as e:
            logger.error(f"❌ テスト中にエラーが発生: {e}")
            traceback.print_exc()
            return False
            
        return True
    
    def _test_gpu_environment(self):
        """GPU環境テスト"""
        logger.info("\n=== 1. GPU環境確認 ===")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA が利用できません")
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ CUDA利用可能: {gpu_count}個のGPU検出")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # VRAM使用量表示
        for i in range(gpu_count):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            logger.info(f"  GPU {i} メモリ: {allocated:.2f}GB使用中, {cached:.2f}GB予約済み")
        
        self.test_results['gpu_environment'] = True
    
    def _test_loss_functions(self):
        """損失関数テスト"""
        logger.info("\n=== 2. 損失関数テスト ===")
        
        try:
            # 複合損失関数のテスト実行
            logger.info("📊 複合損失関数テスト実行中...")
            test_composite_loss()
            
            # 各段階の損失関数初期化テスト
            for stage in [1, 2, 3]:
                logger.info(f"  Stage {stage} 損失関数初期化...")
                loss_fn = get_composite_loss_qformer_sam2(stage=stage, device='cpu')
                logger.info(f"  ✅ Stage {stage} 初期化成功")
            
            self.test_results['loss_functions'] = True
            logger.info("✅ 損失関数テスト完了")
            
        except Exception as e:
            logger.error(f"❌ 損失関数テスト失敗: {e}")
            self.test_results['loss_functions'] = False
            raise
    
    def _test_model_initialization(self):
        """モデル初期化テスト"""
        logger.info("\n=== 3. モデル初期化テスト ===")
        
        try:
            # 設定作成
            config = LlamaQFormerSAM2Config()
            logger.info(f"📋 設定作成完了: {config.llama_model_id}")
            
            # 方法3（メイン実装）でモデル初期化
            logger.info("🏗️ 方法3 Q-Formerモデル初期化中...")
            self.model = QFormerSegmentationBridge(config=config, training_stage=1)
            
            logger.info("✅ モデル初期化成功")
            self.test_results['model_initialization'] = True
            
        except Exception as e:
            logger.error(f"❌ モデル初期化失敗: {e}")
            self.test_results['model_initialization'] = False
            raise
    
    def _test_model_info(self):
        """モデル情報表示テスト"""
        logger.info("\n=== 4. モデル情報表示テスト ===")
        
        try:
            if hasattr(self.model, 'get_model_info'):
                info = self.model.get_model_info()
                logger.info("📊 モデル情報:")
                for key, value in info.items():
                    if isinstance(value, int) and value > 1000:
                        logger.info(f"  - {key}: {value:,}")
                    else:
                        logger.info(f"  - {key}: {value}")
            
            # パラメータ数計算
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"📈 パラメータ統計:")
            logger.info(f"  - 総パラメータ数: {total_params:,}")
            logger.info(f"  - 学習可能パラメータ数: {trainable_params:,}")
            logger.info(f"  - 学習可能比率: {trainable_params/total_params*100:.2f}%")
            
            self.test_results['model_info'] = True
            logger.info("✅ モデル情報表示テスト完了")
            
        except Exception as e:
            logger.error(f"❌ モデル情報表示テスト失敗: {e}")
            self.test_results['model_info'] = False
    
    def _test_dummy_forward(self):
        """ダミーデータでのフォワードパステスト"""
        logger.info("\n=== 5. ダミーデータテスト ===")
        
        try:
            batch_size = 1
            image_size = 448  # Llama-4-Scout用
            seq_len = 32
            
            # ダミーデータ作成
            images = torch.randint(0, 255, (batch_size, 3, image_size, image_size), dtype=torch.uint8).float()
            input_ids = torch.randint(1, 32000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            logger.info(f"📊 ダミーデータ形状:")
            logger.info(f"  - 画像: {images.shape}")
            logger.info(f"  - Input IDs: {input_ids.shape}")
            logger.info(f"  - Attention Mask: {attention_mask.shape}")
            
            # デバイス移動
            device = next(self.model.parameters()).device
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # フォワードパス実行（推論モード）
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            logger.info(f"📊 出力形状:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  - {key}: {value.shape}")
                else:
                    logger.info(f"  - {key}: {type(value)} {value}")
            
            self.test_results['dummy_forward'] = True
            logger.info("✅ ダミーデータテスト完了")
            
        except Exception as e:
            logger.error(f"❌ ダミーデータテスト失敗: {e}")
            self.test_results['dummy_forward'] = False
            traceback.print_exc()
    
    def _test_realistic_forward(self):
        """リアルなテストデータでのフォワードパステスト"""
        logger.info("\n=== 6. リアルテストデータテスト ===")
        
        try:
            # テスト画像生成（より現実的な画像）
            image_array = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            # テキスト例
            test_text = "画像内の物体を[SEG]してください"
            
            logger.info(f"📊 テストデータ:")
            logger.info(f"  - 画像サイズ: {image.size}")
            logger.info(f"  - テキスト: '{test_text}'")
            
            # 前処理（簡易版）
            images_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
            
            # ダミートークナイゼーション
            input_ids = torch.randint(1, 32000, (1, 64))
            attention_mask = torch.ones(1, 64)
            
            # デバイス移動
            device = next(self.model.parameters()).device
            images_tensor = images_tensor.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # フォワードパス実行
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(
                    images=images_tensor,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # 結果の詳細表示
            logger.info(f"📊 リアルデータ出力:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  - {key}: {value.shape} (device: {value.device})")
                    if value.numel() < 10:  # 小さなテンソルは値も表示
                        logger.info(f"    値: {value}")
                else:
                    logger.info(f"  - {key}: {value}")
            
            self.test_results['realistic_forward'] = True
            logger.info("✅ リアルテストデータテスト完了")
            
        except Exception as e:
            logger.error(f"❌ リアルテストデータテスト失敗: {e}")
            self.test_results['realistic_forward'] = False
            traceback.print_exc()
    
    def _test_loss_computation(self):
        """損失計算テスト"""
        logger.info("\n=== 7. 損失計算テスト ===")
        
        try:
            # テストデータ準備 - config統一設定を使用
            test_config = config_linux.get_test_config()
            batch_size = 1
            num_queries = 64
            height, width = test_config['mask_size'], test_config['mask_size']
            
            # ダミー予測・正解データ（勾配計算可能に設定）
            predicted_masks = torch.randn(batch_size, num_queries, height, width, requires_grad=True)
            target_masks = torch.randint(0, 2, (batch_size, height, width)).float()
            query_embeds = torch.randn(batch_size, num_queries, 5120, requires_grad=True)
            text_embeds = torch.randn(batch_size, 5120, requires_grad=True)
            sam_prompts = torch.randn(batch_size, num_queries, 256, requires_grad=True)
            
            # デバイス移動
            device = next(self.model.parameters()).device
            predicted_masks = predicted_masks.to(device)
            target_masks = target_masks.to(device)
            query_embeds = query_embeds.to(device)
            text_embeds = text_embeds.to(device)
            sam_prompts = sam_prompts.to(device)
            
            logger.info(f"📊 損失計算用データ形状:")
            logger.info(f"  - 予測マスク: {predicted_masks.shape}")
            logger.info(f"  - 正解マスク: {target_masks.shape}")
            logger.info(f"  - クエリ埋め込み: {query_embeds.shape}")
            
            # 損失計算
            if hasattr(self.model, 'compute_loss'):
                losses = self.model.compute_loss(
                    predicted_masks=predicted_masks,
                    target_masks=target_masks,
                    query_embeds=query_embeds,
                    text_embeds=text_embeds,
                    sam_prompts=sam_prompts
                )
                
                logger.info(f"📊 計算された損失:")
                for loss_name, loss_value in losses.items():
                    logger.info(f"  - {loss_name}: {loss_value.item():.6f}")
                
                # 勾配計算テスト
                total_loss = losses['total_loss']
                total_loss.backward()
                logger.info("✅ 勾配計算成功")
                
            else:
                logger.warning("⚠️ compute_loss メソッドが見つかりません")
            
            self.test_results['loss_computation'] = True
            logger.info("✅ 損失計算テスト完了")
            
        except Exception as e:
            logger.error(f"❌ 損失計算テスト失敗: {e}")
            self.test_results['loss_computation'] = False
            traceback.print_exc()
    
    def _test_memory_usage(self):
        """メモリ使用量テスト"""
        logger.info("\n=== 8. メモリ使用量テスト ===")
        
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                    
                    logger.info(f"📊 GPU {i} メモリ使用量:")
                    logger.info(f"  - 現在割り当て: {allocated:.2f}GB")
                    logger.info(f"  - 予約済み: {cached:.2f}GB")
                    logger.info(f"  - 最大割り当て: {max_allocated:.2f}GB")
            
            self.test_results['memory_usage'] = True
            logger.info("✅ メモリ使用量テスト完了")
            
        except Exception as e:
            logger.error(f"❌ メモリ使用量テスト失敗: {e}")
            self.test_results['memory_usage'] = False
    
    def _test_device_placement(self):
        """デバイス配置テスト"""
        logger.info("\n=== 9. デバイス配置テスト ===")
        
        try:
            logger.info("📊 モデルコンポーネントのデバイス配置:")
            
            device_info = {}
            
            # Llama-4モデル
            if hasattr(self.model, 'llama_model'):
                device = next(self.model.llama_model.parameters()).device
                dtype = next(self.model.llama_model.parameters()).dtype
                device_info['llama4'] = f"{device}, {dtype}"
                logger.info(f"  - Llama-4: {device}, {dtype}")
            
            # Q-Former
            if hasattr(self.model, 'qformer'):
                device = next(self.model.qformer.parameters()).device
                dtype = next(self.model.qformer.parameters()).dtype
                device_info['qformer'] = f"{device}, {dtype}"
                logger.info(f"  - Q-Former: {device}, {dtype}")
            
            # 強化プロジェクター
            if hasattr(self.model, 'enhanced_sam_projector'):
                device = next(self.model.enhanced_sam_projector.parameters()).device
                dtype = next(self.model.enhanced_sam_projector.parameters()).dtype
                device_info['enhanced_projector'] = f"{device}, {dtype}"
                logger.info(f"  - 強化プロジェクター: {device}, {dtype}")
            
            # 損失関数
            if hasattr(self.model, 'loss_function') and hasattr(self.model.loss_function, 'parameters'):
                try:
                    device = next(self.model.loss_function.parameters()).device
                    dtype = next(self.model.loss_function.parameters()).dtype
                    device_info['loss_function'] = f"{device}, {dtype}"
                    logger.info(f"  - 損失関数: {device}, {dtype}")
                except StopIteration:
                    logger.info(f"  - 損失関数: パラメータなし（正常）")
            
            # デバイス統一性チェック
            devices = set()
            dtypes = set()
            for component, info in device_info.items():
                device_str, dtype_str = info.split(", ")
                devices.add(device_str)
                dtypes.add(dtype_str)
            
            if len(devices) == 1 and len(dtypes) == 1:
                logger.info(f"✅ 全コンポーネントが統一されています: {list(devices)[0]}, {list(dtypes)[0]}")
            else:
                logger.warning(f"⚠️ デバイス・データ型が不統一: デバイス={devices}, データ型={dtypes}")
            
            self.test_results['device_placement'] = True
            logger.info("✅ デバイス配置テスト完了")
            
        except Exception as e:
            logger.error(f"❌ デバイス配置テスト失敗: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['device_placement'] = False
    
    def _print_test_summary(self):
        """テスト結果サマリ表示"""
        logger.info("\n=== テスト結果サマリ ===")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        logger.info(f"📊 実行テスト数: {total_tests}")
        logger.info(f"✅ 成功: {passed_tests}")
        logger.info(f"❌ 失敗: {total_tests - passed_tests}")
        
        for test_name, result in self.test_results.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {test_name}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 全テスト成功！")
        else:
            logger.info(f"\n⚠️ {total_tests - passed_tests}個のテストで問題があります")

def main():
    """メイン実行関数"""
    print("=== LISA-Llama4-QFormer-SAM2 統合モデル単独テスト ===")
    print(f"実行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = LlamaQFormerSAM2StandaloneTest()
    success = tester.run_full_test()
    
    if success:
        print("\n✅ テスト完了 - 成功")
        return 0
    else:
        print("\n❌ テスト完了 - 失敗")
        return 1

if __name__ == "__main__":
    exit(main())