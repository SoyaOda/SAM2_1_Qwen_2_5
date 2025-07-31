#!/usr/bin/env python3
"""
DDP学習の段階的テストスクリプト
DeepSpeed移行に向けた準備として、DDPの動作を確認
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
import argparse
import time

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 動的設定管理機能を追加
def get_config():
    """動的設定読み込み（環境変数対応）"""
    config_path = os.environ.get('LISA_CONFIG_PATH', None)
    
    if config_path:
        # 環境変数で指定された設定ファイル
        try:
            config_module = __import__(config_path)
            print(f"✓ カスタム設定ファイルを使用: {config_path}")
            return config_module
        except ImportError:
            print(f"⚠️ カスタム設定ファイル {config_path} が見つかりません")
    
    # デフォルトの設定ファイル検索順序
    config_candidates = ['config_small_test', 'config_linux']
    
    for config_name in config_candidates:
        try:
            config_module = __import__(config_name)
            print(f"✓ 設定ファイルを使用: {config_name}")
            return config_module
        except ImportError:
            continue
    
    raise ImportError("利用可能な設定ファイルが見つかりません")

# 動的設定読み込み
config = get_config()

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model

def test_basic_imports():
    """Step 1: 基本的なインポートテスト"""
    print("=== Step 1: 基本インポートテスト ===")
    
    try:
        import torch.distributed as dist
        print("✅ torch.distributed インポート成功")
        
        from torch.nn.parallel import DistributedDataParallel as DDP
        print("✅ DDP インポート成功")
        
        import torch.multiprocessing as mp
        print("✅ multiprocessing インポート成功")
        
        print(f"✅ CUDA利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU数: {torch.cuda.device_count()}")
        
        return True
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        return False

def test_model_initialization():
    """Step 2: モデル初期化テスト"""
    print("\n=== Step 2: モデル初期化テスト ===")
    
    try:
        # Gemma プロセッサー
        gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
        print("✅ Gemmaプロセッサー初期化成功")
        
        # SEGトークン追加
        seg_token = "[SEG]"
        if seg_token not in gemma_processor.tokenizer.get_vocab():
            gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        print("✅ SEGトークン追加成功")
        
        # モデル設定
        model_config = LisaGemmaConfig(
            gemma_model_id=config.GEMMA_MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            seg_token_idx=gemma_processor.tokenizer.convert_tokens_to_ids(seg_token),
            gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),
            sam_prompt_embed_dim=getattr(config, 'SEG_PROJECTION_DIM', 256),
        )
        print("✅ モデル設定作成成功")
        
        # モデル初期化
        model = LisaGemmaForCausalLM(model_config)
        model.resize_token_embeddings(len(gemma_processor.tokenizer))
        print("✅ モデル初期化成功")
        
        # LoRA設定
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("✅ LoRA適用成功")
        
        # パラメータ確認
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ 総パラメータ: {total_params:,}")
        print(f"   訓練可能: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        return model, gemma_processor
        
    except Exception as e:
        print(f"❌ モデル初期化エラー: {e}")
        return None, None

def test_data_loading():
    """Step 3: データ読み込みテスト"""
    print("\n=== Step 3: データ読み込みテスト ===")
    
    try:
        # ダミープロセッサー（テスト用）
        from transformers import AutoProcessor
        gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
        seg_token = "[SEG]"
        if seg_token not in gemma_processor.tokenizer.get_vocab():
            gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        
        # 小さなデータセット
        dataset = HybridDataset(
            base_image_dir=config.DATASET_BASE_DIR,
            gemma_processor=gemma_processor,
            samples_per_epoch=10,  # 小さなサンプル数
            precision="bf16",
            gemma_image_size=config.GEMMA_IMAGE_SIZE,
            sam_image_size=config.SAM_IMAGE_SIZE,
            dataset="reason_seg",
            sample_rate=[1.0],
            reason_seg_data="ReasonSeg|train",
        )
        print(f"✅ データセット作成成功: {len(dataset)} サンプル")
        
        # データローダー
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # テスト用
        )
        print("✅ データローダー作成成功")
        
        # 1バッチ取得テスト
        batch = next(iter(dataloader))
        print(f"✅ バッチ取得成功")
        print(f"   images_for_gemma: {batch['images_for_gemma'].shape}")
        print(f"   images_for_sam: {batch['images_for_sam'].shape}")
        print(f"   input_ids: {batch['input_ids'].shape}")
        
        return dataloader
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return None

def test_single_gpu_forward():
    """Step 4: 単一GPU フォワードパステスト"""
    print("\n=== Step 4: 単一GPU フォワードテスト ===")
    
    model, gemma_processor = test_model_initialization()
    if model is None:
        return False
        
    dataloader = test_data_loading()
    if dataloader is None:
        return False
    
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ モデルを{device}に移動")
        
        # 1バッチでフォワードパス
        batch = next(iter(dataloader))
        
        # バッチをGPUに移動
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            print("✅ フォワードパス成功")
            
            loss_fn = CompositeLoss(
                ce_loss_weight=config.CE_LOSS_WEIGHT,
                dice_loss_weight=config.DICE_LOSS_WEIGHT,
                bce_loss_weight=config.BCE_LOSS_WEIGHT
            )
            
            loss_dict = loss_fn(outputs, batch)
            print(f"✅ 損失計算成功: {loss_dict['total_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ フォワードパスエラー: {e}")
        return False

def test_distributed_setup():
    """Step 5: 分散環境セットアップテスト"""
    print("\n=== Step 5: 分散環境セットアップテスト ===")
    
    # 環境変数の確認
    required_env_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    
    if missing_vars:
        print(f"⚠️ 必要な環境変数が設定されていません: {missing_vars}")
        print("シングルプロセス環境でテストを続行します")
        return test_single_gpu_forward()
    
    try:
        # 分散環境の初期化
        dist.init_process_group(backend='nccl')
        print("✅ 分散環境初期化成功")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"✅ ランク: {rank}, ワールドサイズ: {world_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分散環境セットアップエラー: {e}")
        return False

def run_ddp_train_test():
    """Step 6: 実際のDDP学習テスト"""
    print("\n=== Step 6: DDP学習テスト ===")
    
    try:
        cmd = [
            sys.executable, "train_ddp.py",
            "--batch_size", "2",
            "--grad_accumulation_steps", "1", 
            "--steps_per_epoch", "5",
            "--epochs", "1",
            "--exp_name", "ddp_test",
            "--world_size", "1",
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ DDP学習テスト成功")
            print("出力の最後の部分:")
            print(result.stdout.split('\n')[-10:])
            return True
        else:
            print("❌ DDP学習テスト失敗")
            print("エラー:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ DDP学習テスト: タイムアウト")
        return False
    except Exception as e:
        print(f"❌ DDP学習テストエラー: {e}")
        return False

def main():
    """メインテストフロー"""
    print("🚀 LISA-Gemma3 DDP段階的テスト開始")
    print("=" * 60)
    
    test_results = []
    
    # Step 1: 基本インポート
    result1 = test_basic_imports()
    test_results.append(("基本インポート", result1))
    
    if not result1:
        print("❌ 基本インポートが失敗したため、テストを中断します")
        return
    
    # Step 2: モデル初期化
    model, processor = test_model_initialization()
    test_results.append(("モデル初期化", model is not None))
    
    # Step 3: データ読み込み
    dataloader = test_data_loading()
    test_results.append(("データ読み込み", dataloader is not None))
    
    # Step 4: 単一GPU フォワード
    result4 = test_single_gpu_forward()
    test_results.append(("単一GPU フォワード", result4))
    
    # Step 5: 分散セットアップ
    result5 = test_distributed_setup()
    test_results.append(("分散セットアップ", result5))
    
    # Step 6: DDP学習（train_ddp.pyが存在する場合）
    if os.path.exists("train_ddp.py"):
        result6 = run_ddp_train_test()
        test_results.append(("DDP学習", result6))
    else:
        print("⚠️  train_ddp.py が見つかりません。先に作成してください。")
        test_results.append(("DDP学習", False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print("=" * 60)
    
    for test_name, result in test_results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name:20} : {status}")
    
    # 成功率計算
    success_count = sum(1 for _, result in test_results if result)
    total_count = len(test_results)
    success_rate = success_count / total_count * 100
    
    print(f"\n成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 DDP実装準備が整いました！")
        print("次のステップ: train_ddp.py で本格学習を開始してください")
    else:
        print("⚠️  いくつかの問題があります。上記の失敗項目を確認してください")

if __name__ == "__main__":
    main() 