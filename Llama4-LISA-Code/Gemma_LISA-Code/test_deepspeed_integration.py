#!/usr/bin/env python3
"""
DeepSpeed統合テストスクリプト
フェーズ2.3: DeepSpeed ZeRO Stage 2統合の基本機能検証

ローカル環境でDeepSpeedの基本的な機能をテストし、
Lambda Cloud A100*8での実行前に潜在的な問題を検出する。
"""

import os
import sys
import json
import torch
import deepspeed
from datetime import datetime

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_deepspeed_imports():
    """DeepSpeed関連のインポートテスト"""
    print("🔍 Test 1: DeepSpeed関連インポート")
    try:
        import deepspeed
        from deepspeed.utils import logger
        print(f"✅ DeepSpeed バージョン: {deepspeed.__version__}")
        print(f"✅ DeepSpeed インポート成功")
        return True
    except ImportError as e:
        print(f"❌ DeepSpeed インポートエラー: {e}")
        return False

def test_config_file_exists():
    """設定ファイル存在テスト"""
    print("\n🔍 Test 2: 設定ファイル存在確認")
    
    required_files = [
        "deepspeed_zero2_config.json",
        "config_linux.py",
        "train_deepspeed.py",
        "launch_deepspeed.sh"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - 存在")
        else:
            print(f"❌ {file} - 見つかりません")
            all_exist = False
    
    return all_exist

def test_deepspeed_config_format():
    """DeepSpeed設定ファイルの形式テスト"""
    print("\n🔍 Test 3: DeepSpeed設定ファイル形式確認")
    
    config_file = "deepspeed_zero2_config.json"
    if not os.path.exists(config_file):
        print(f"❌ 設定ファイルが見つかりません: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 必須キーの確認
        required_keys = [
            "bf16", "optimizer", "scheduler", "zero_optimization",
            "gradient_accumulation_steps", "train_batch_size"
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ 必須キーが不足: {missing_keys}")
            return False
        
        print("✅ DeepSpeed設定ファイル形式正常")
        print(f"   - ZeRO Stage: {config['zero_optimization']['stage']}")
        print(f"   - 混合精度: {'bf16' if config.get('bf16', {}).get('enabled', False) else 'fp32'}")
        print(f"   - オプティマイザー: {config['optimizer']['type']}")
        print(f"   - スケジューラー: {config['scheduler']['type']}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON形式エラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return False

def test_model_imports():
    """プロジェクトモデルのインポートテスト"""
    print("\n🔍 Test 4: プロジェクトモデルインポート")
    
    try:
        from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
        from model.losses import CompositeLoss
        from utils.dataset import HybridDataset, collate_fn
        print("✅ プロジェクトモデルインポート成功")
        return True
    except ImportError as e:
        print(f"❌ プロジェクトモデルインポートエラー: {e}")
        return False

def test_config_linux_access():
    """config_linux.pyアクセステスト"""
    print("\n🔍 Test 5: config_linux.py設定アクセス")
    
    try:
        import config_linux as config
        
        # 重要な設定値の確認
        required_attrs = [
            'GEMMA_MODEL_ID', 'LEARNING_RATE', 'BATCH_SIZE_PER_GPU',
            'LORA_R', 'LORA_ALPHA', 'LORA_TARGET_MODULES'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"❌ 必須設定が不足: {missing_attrs}")
            return False
        
        print("✅ config_linux.py アクセス成功")
        print(f"   - Gemmaモデル: {config.GEMMA_MODEL_ID}")
        print(f"   - 学習率: {config.LEARNING_RATE}")
        print(f"   - バッチサイズ: {config.BATCH_SIZE_PER_GPU}")
        print(f"   - LoRA rank: {config.LORA_R}")
        
        return True
        
    except ImportError as e:
        print(f"❌ config_linux.py インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ config_linux.py アクセスエラー: {e}")
        return False

def test_cuda_environment():
    """CUDA環境テスト"""
    print("\n🔍 Test 6: CUDA環境確認")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA利用不可 - CPU環境で実行")
        return True  # CPUでも基本テストは通す
    
    try:
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA利用可能")
        print(f"   - 検出されたGPU数: {device_count}")
        
        for i in range(device_count):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / (1024**3)
            print(f"   - GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA環境エラー: {e}")
        return False

def test_wandb_availability():
    """WandB利用可能性テスト"""
    print("\n🔍 Test 7: WandB利用可能性確認")
    
    try:
        import wandb
        print("✅ WandB インポート成功")
        
        # API キーの確認（詳細は表示しない）
        api_key = wandb.api.api_key
        if api_key:
            print("✅ WandB API キー設定済み")
        else:
            print("⚠️ WandB API キー未設定 - 手動ログインが必要")
        
        return True
        
    except ImportError as e:
        print(f"❌ WandB インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"⚠️ WandB設定警告: {e}")
        return True  # WandBエラーでも学習は継続可能

def test_argument_parsing():
    """引数解析テスト"""
    print("\n🔍 Test 8: 引数解析機能確認")
    
    try:
        # train_deepspeed.pyから関数をインポート
        from train_deepspeed import parse_args, get_config
        
        # 設定取得テスト
        config = get_config()
        print("✅ get_config() 成功")
        
        # 引数解析テスト（デフォルト値で）
        import sys
        original_argv = sys.argv
        sys.argv = ["train_deepspeed.py"]  # デフォルト引数でテスト
        
        try:
            args = parse_args()
            print("✅ parse_args() 成功")
            print(f"   - エポック数: {args.epochs}")
            print(f"   - ステップ/エポック: {args.steps_per_epoch}")
            print(f"   - 学習率: {args.learning_rate}")
        finally:
            sys.argv = original_argv  # 元に戻す
        
        return True
        
    except Exception as e:
        print(f"❌ 引数解析エラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("=" * 80)
    print("🧪 DeepSpeed統合テストスイート")
    print("フェーズ2.3: DeepSpeed ZeRO Stage 2統合")
    print("=" * 80)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # テスト実行
    tests = [
        test_deepspeed_imports,
        test_config_file_exists,
        test_deepspeed_config_format,
        test_model_imports,
        test_config_linux_access,
        test_cuda_environment,
        test_wandb_availability,
        test_argument_parsing,
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ テスト実行中にエラー: {e}")
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("🎯 テスト結果サマリー")
    print("=" * 80)
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"合格テスト数: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if passed_tests == total_tests:
        print("✅ 全テスト合格 - DeepSpeed統合準備完了")
        print("   → Lambda Cloud A100*8での実行準備が整いました")
        print("   → 実行コマンド: ./launch_deepspeed.sh")
        exit_code = 0
    elif passed_tests >= total_tests * 0.8:  # 80%以上
        print("⚠️ 部分的合格 - 一部警告あり")
        print("   → 多くの機能は正常ですが、いくつかの問題があります")
        print("   → 問題を修正後、再テストを推奨します")
        exit_code = 1
    else:
        print("❌ 複数のテスト失敗 - 修正が必要")
        print("   → 重要な問題が検出されました")
        print("   → Lambda Cloud実行前にすべての問題を修正してください")
        exit_code = 2
    
    print("\n🔧 推奨事項:")
    if success_rate < 100:
        print("  1. 失敗したテストの問題を修正")
        print("  2. 必要なファイルの存在確認")
        print("  3. 依存ライブラリの再インストール")
    print("  4. Lambda Cloud A100*8環境での実行")
    print("  5. WandBでの学習進捗監視")
    
    print("=" * 80)
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 