#!/usr/bin/env python3
"""
最小限のCUDA初期化テスト
"""

import os
import sys

# 最大限のCUDA Error 802対策
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'  # NVML無効化
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

print("🔧 最小限CUDA初期化テスト...")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

try:
    import torch
    print(f"✅ PyTorch版: {torch.__version__}")
    print(f"✅ CUDA版: {torch.version.cuda}")
    
    # 最小限のCUDA確認
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("CUDA利用可能 - 基本テスト実行...")
        
        # 単純なテンソル作成
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"CPU tensor: {x}")
        
        # GPU転送テスト
        try:
            x_gpu = x.cuda()
            print(f"GPU tensor: {x_gpu}")
            print("✅ GPU転送成功")
            
            # 簡単な計算
            result = x_gpu * 2
            print(f"GPU計算結果: {result}")
            print("✅ GPU計算成功")
            
        except Exception as gpu_error:
            print(f"❌ GPU操作エラー: {gpu_error}")
            
    else:
        print("❌ CUDA利用不可")
        
except Exception as e:
    print(f"❌ PyTorchエラー: {e}")
    import traceback
    traceback.print_exc()

print("テスト完了")