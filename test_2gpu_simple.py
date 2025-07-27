#!/usr/bin/env python3
"""
2GPU環境での簡単なテスト
"""

import os
import sys

# CUDA Error 802対策
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

print("🔧 2GPU環境簡単テスト開始...")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch
import time

def main():
    """メイン実行"""
    print("🚀 2GPU環境テスト開始")
    print("=" * 40)
    
    # 基本情報
    print(f"PyTorch版: {torch.__version__}")
    print(f"CUDA版: {torch.version.cuda}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA利用不可")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"GPU数: {device_count}")
    
    # 各GPU情報
    for i in range(device_count):
        try:
            name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {name}")
        except Exception as e:
            print(f"  GPU {i}: エラー - {e}")
    
    # 各GPUでの簡単な計算
    print("\n📊 各GPUでの計算テスト:")
    for i in range(device_count):
        try:
            device = f"cuda:{i}"
            x = torch.randn(100, 100, device=device)
            y = x * 2
            result = y.sum().item()
            print(f"  GPU {i}: 計算成功 (結果: {result:.2f})")
        except Exception as e:
            print(f"  GPU {i}: 計算エラー - {e}")
    
    # GPU間コピーテスト
    if device_count >= 2:
        print("\n📊 GPU間コピーテスト:")
        try:
            x = torch.randn(100, 100, device='cuda:0')
            y = x.to('cuda:1')
            print("✅ GPU 0→1コピー成功")
            
            z = y.to('cuda:0')
            print("✅ GPU 1→0コピー成功")
            
        except Exception as e:
            print(f"❌ GPU間コピーエラー: {e}")
    
    print("\n" + "=" * 40)
    print("🎯 2GPU環境テスト完了")
    print("=" * 40)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)