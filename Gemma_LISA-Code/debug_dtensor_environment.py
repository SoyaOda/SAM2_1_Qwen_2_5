#!/usr/bin/env python3
"""
DTensor問題詳細診断スクリプト - 根本原因特定版
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

def print_banner(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def safe_import_test(module_name, import_statement):
    """安全なインポートテスト"""
    try:
        exec(import_statement)
        return "✅ SUCCESS"
    except Exception as e:
        return f"❌ FAILED: {str(e)}"

def main():
    print_banner("DTensor問題 包括的環境診断")
    print(f"実行時刻: {datetime.now()}")
    
    # Step 1: 基本環境情報
    print_banner("Step 1: 基本環境情報")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Step 2: PyTorchバージョン詳細
    print_banner("Step 2: PyTorchバージョン詳細確認")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
        
        # PyTorch 2.6の重要チェック
        torch_version = torch.__version__
        major, minor = torch_version.split('.')[:2]
        is_pytorch_26_plus = int(major) >= 2 and int(minor) >= 6
        print(f"✅ PyTorch 2.6+: {is_pytorch_26_plus}")
        
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return
    
    # Step 3: DTensorインポート互換性テスト（重要）
    print_banner("Step 3: DTensorインポート互換性診断")
    
    # NEW API (PyTorch 2.5+)
    new_api_result = safe_import_test(
        "torch.distributed.tensor", 
        "from torch.distributed.tensor import DTensor, distribute_tensor, Replicate"
    )
    print(f"New API (torch.distributed.tensor): {new_api_result}")
    
    # OLD API (PyTorch 2.4-)
    old_api_result = safe_import_test(
        "torch.distributed._tensor",
        "from torch.distributed._tensor import DTensor, distribute_tensor, Replicate"
    )
    print(f"Old API (torch.distributed._tensor): {old_api_result}")
    
    # Step 4: transformersライブラリ詳細
    print_banner("Step 4: transformersライブラリ診断")
    try:
        import transformers
        print(f"✅ transformers version: {transformers.__version__}")
        
        # transformers 4.52.4の問題チェック
        if transformers.__version__.startswith('4.52'):
            print("⚠️  WARNING: transformers 4.52.x has known DTensor issues with PyTorch 2.6+")
        
        # モデルのresize_token_embeddingsテスト
        try:
            from transformers import AutoTokenizer, AutoModel
            print("✅ transformers core imports: SUCCESS")
        except Exception as e:
            print(f"❌ transformers imports failed: {e}")
            
    except Exception as e:
        print(f"❌ transformers import failed: {e}")
    
    # Step 5: accelerateライブラリ診断
    print_banner("Step 5: accelerateライブラリ診断")
    try:
        import accelerate
        print(f"✅ accelerate version: {accelerate.__version__}")
        
        # accelerate設定確認
        try:
            from accelerate.utils import find_executable_batch_size
            from accelerate import Accelerator
            print("✅ accelerate core imports: SUCCESS")
        except Exception as e:
            print(f"❌ accelerate imports failed: {e}")
            
    except Exception as e:
        print(f"❌ accelerate import failed: {e}")
    
    # Step 6: 分散環境変数確認
    print_banner("Step 6: 分散環境変数診断")
    dist_env_vars = [
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
        'CUDA_VISIBLE_DEVICES', 'TORCH_DISTRIBUTED_DEBUG',
        'TORCH_DTENSOR_ALLOW_IMPLICIT_REPLICATION'
    ]
    
    for var in dist_env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")
    
    # Step 7: DTensor環境実際のテスト
    print_banner("Step 7: DTensor環境実際テスト")
    
    try:
        # DTensorインポート（バージョン互換）
        dtensor_imported = False
        dtensor_api_version = "unknown"
        
        try:
            from torch.distributed.tensor import DTensor
            dtensor_imported = True
            dtensor_api_version = "new_api_2.5+"
            print("✅ DTensor import (new API): SUCCESS")
        except ImportError:
            try:
                from torch.distributed._tensor import DTensor
                dtensor_imported = True
                dtensor_api_version = "old_api_2.4-"
                print("✅ DTensor import (old API): SUCCESS")
            except ImportError as e:
                print(f"❌ DTensor import failed: {e}")
        
        if dtensor_imported:
            print(f"✅ DTensor API version: {dtensor_api_version}")
            
            # 分散環境初期化テスト
            try:
                if torch.distributed.is_available():
                    print("✅ torch.distributed available")
                    if torch.distributed.is_initialized():
                        print("✅ torch.distributed initialized")
                    else:
                        print("ℹ️  torch.distributed not initialized (normal for single process)")
                else:
                    print("❌ torch.distributed not available")
            except Exception as e:
                print(f"❌ Distributed check failed: {e}")
        
    except Exception as e:
        print(f"❌ DTensor test failed: {e}")
        traceback.print_exc()
    
    # Step 8: resize_token_embeddings問題の直接テスト
    print_banner("Step 8: resize_token_embeddings問題診断")
    
    try:
        # 小さなモデルでテスト
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # 最小構成でのテスト
        config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
        config.vocab_size = 1000  # 小さくしてメモリ節約
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        
        print("✅ Config created")
        
        # モデル作成テスト
        model = AutoModelForCausalLM.from_config(config)
        print(f"✅ Model created, vocab_size: {model.config.vocab_size}")
        
        # resize_token_embeddingsテスト
        original_vocab_size = model.config.vocab_size
        new_vocab_size = original_vocab_size + 10
        
        try:
            model.resize_token_embeddings(new_vocab_size)
            print(f"✅ resize_token_embeddings: {original_vocab_size} → {new_vocab_size} SUCCESS")
        except Exception as e:
            print(f"❌ resize_token_embeddings FAILED: {e}")
            print(f"   Error type: {type(e).__name__}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        traceback.print_exc()
    
    # Step 9: PyTorch 2.6の重要変更確認
    print_banner("Step 9: PyTorch 2.6重要変更確認")
    
    # torch.loadのweights_onlyデフォルト確認
    try:
        import tempfile
        import torch
        
        # テストテンソル保存
        test_tensor = torch.randn(2, 2)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(test_tensor, f.name)
            
            # weights_onlyデフォルト動作確認
            try:
                loaded = torch.load(f.name)  # デフォルト動作
                print("✅ torch.load default behavior: SUCCESS")
            except Exception as e:
                print(f"⚠️  torch.load default failed: {e}")
                print("   This indicates PyTorch 2.6 weights_only=True default")
            
            # 明示的にweights_only=False
            try:
                loaded = torch.load(f.name, weights_only=False)
                print("✅ torch.load(weights_only=False): SUCCESS")
            except Exception as e:
                print(f"❌ torch.load(weights_only=False) failed: {e}")
        
        os.unlink(f.name)
        
    except Exception as e:
        print(f"❌ torch.load test failed: {e}")
    
    # Step 10: 推奨解決策
    print_banner("Step 10: 根本原因と推奨解決策")
    
    print("🎯 根本原因分析:")
    print("1. PyTorch 2.6.0の破壊的変更:")
    print("   - DTensorインポートパス変更")
    print("   - torch.loadデフォルトweights_only=True")
    print("2. transformers 4.52.4の非互換性:")
    print("   - DTensor新APIに未対応")
    print("   - resize_token_embeddingsでDTensor/Tensor混在")
    
    print("\n🚀 推奨解決策:")
    print("Option A: バージョンダウングレード")
    print("   pip install torch==2.4.1 transformers==4.45.0")
    print("Option B: 上位バージョンへのアップグレード")
    print("   pip install torch>=2.6.1 transformers>=4.55.0")
    print("Option C: DeepSpeed直接使用（accelerate bypass）")
    print("   deepspeed --num_gpus=8 train_ds.py")
    
    print(f"\n診断完了: {datetime.now()}")

if __name__ == "__main__":
    main() 