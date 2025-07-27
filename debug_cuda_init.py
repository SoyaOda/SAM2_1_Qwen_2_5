#!/usr/bin/env python3
"""
1x H100 SXM5でのCUDA初期化デバッグスクリプト
Lambda Cloud環境での基本的なCUDA初期化問題を特定
"""

import sys
import os
import subprocess
import time

def debug_cuda_environment():
    """CUDA環境の詳細デバッグ"""
    print("🔍 1x H100 SXM5 CUDA環境デバッグ開始")
    print("=" * 60)
    
    # 1. 基本システム情報
    print("\n📊 基本システム情報:")
    try:
        result = subprocess.run(['uname', '-a'], capture_output=True, text=True)
        print(f"  システム: {result.stdout.strip()}")
        
        result = subprocess.run(['cat', '/etc/os-release'], capture_output=True, text=True)
        os_info = [line for line in result.stdout.split('\n') if 'PRETTY_NAME' in line]
        if os_info:
            os_name = os_info[0].split('=')[1].strip('"')
            print(f"  OS: {os_name}")
    except Exception as e:
        print(f"  ❌ システム情報取得エラー: {e}")
    
    # 2. NVIDIA GPU情報
    print("\n📊 NVIDIA GPU情報:")
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  GPU: {result.stdout.strip()}")
        else:
            print(f"  ❌ nvidia-smi -L 失敗: {result.stderr}")
    except Exception as e:
        print(f"  ❌ nvidia-smi実行エラー: {e}")
    
    # 3. NVIDIAドライバー情報
    print("\n📊 NVIDIAドライバー情報:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,cuda_version,name', '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  詳細: {result.stdout.strip()}")
        else:
            print(f"  ❌ ドライバー情報取得失敗: {result.stderr}")
    except Exception as e:
        print(f"  ❌ ドライバー情報エラー: {e}")
    
    # 4. CUDA環境変数確認
    print("\n📊 CUDA環境変数:")
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTORCH_NVML_BASED_CUDA_CHECK', 'CUDA_LAUNCH_BLOCKING']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # 5. PyTorch未インポート時のCUDAテスト
    print("\n📊 PyTorch未インポート時のCUDAテスト:")
    try:
        # ctypes経由でCUDAランタイムテスト
        import ctypes
        for lib_name in ['libcudart.so.12', 'libcudart.so.11', 'libcudart.so']:
            try:
                cudart = ctypes.CDLL(lib_name)
                print(f"  ✅ {lib_name} ロード成功")
                
                # デバイス数取得テスト
                device_count = ctypes.c_int()
                result = cudart.cudaGetDeviceCount(ctypes.byref(device_count))
                if result == 0:
                    print(f"  ✅ CUDA デバイス数: {device_count.value}")
                else:
                    print(f"  ❌ cudaGetDeviceCount失敗: エラーコード {result}")
                break
            except OSError as e:
                print(f"  ⚠️ {lib_name} ロード失敗: {e}")
    except Exception as e:
        print(f"  ❌ CUDAランタイムテストエラー: {e}")

def debug_pytorch_cuda_init():
    """PyTorch CUDA初期化デバッグ"""
    print("\n🔍 PyTorch CUDA初期化デバッグ")
    print("=" * 60)
    
    # Step 1: 環境変数設定
    print("\n📊 CUDA Error 802対策環境変数設定:")
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    for key, value in os.environ.items():
        if 'CUDA' in key or 'PYTORCH' in key:
            print(f"  {key}: {value}")
    
    # Step 2: PyTorchインポート
    print("\n📊 PyTorchインポート:")
    try:
        import torch
        print(f"  ✅ PyTorch バージョン: {torch.__version__}")
        print(f"  ✅ CUDA バージョン: {torch.version.cuda}")
        print(f"  ✅ cuDNN バージョン: {torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"  ❌ PyTorchインポートエラー: {e}")
        return False
    
    # Step 3: CUDA利用可能性確認
    print("\n📊 CUDA利用可能性確認:")
    try:
        is_available = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {is_available}")
        
        if is_available:
            device_count = torch.cuda.device_count()
            print(f"  torch.cuda.device_count(): {device_count}")
            
            current_device = torch.cuda.current_device()
            print(f"  torch.cuda.current_device(): {current_device}")
            
            device_name = torch.cuda.get_device_name(0)
            print(f"  torch.cuda.get_device_name(0): {device_name}")
            
            # 実際のGPU操作テスト
            print("\n📊 GPU操作テスト:")
            test_tensor = torch.randn(10, 10, device='cuda')
            result = test_tensor.sum()
            print(f"  ✅ GPU計算テスト成功: {result.item():.4f}")
            
            # メモリ情報
            allocated = torch.cuda.memory_allocated() / 1024**2
            cached = torch.cuda.memory_reserved() / 1024**2
            print(f"  GPU メモリ使用量: {allocated:.1f}MB (割当) / {cached:.1f}MB (予約)")
            
            return True
        else:
            print("  ❌ CUDA利用不可")
            return False
            
    except Exception as e:
        print(f"  ❌ CUDA確認エラー: {e}")
        return False

def debug_fabric_manager():
    """Fabric Manager状態確認（1x H100 SXM5では不要だが確認）"""
    print("\n🔍 Fabric Manager状態確認")
    print("=" * 60)
    
    try:
        result = subprocess.run(['systemctl', 'is-active', 'nvidia-fabricmanager'], capture_output=True, text=True)
        if result.returncode == 0:
            status = result.stdout.strip()
            print(f"  Fabric Manager状態: {status}")
        else:
            print("  Fabric Manager: 未インストールまたは非対応")
            
        # NVSwitch確認
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True)
        if result.returncode == 0:
            if 'NV' in result.stdout:
                print("  NVSwitch: 検出")
            else:
                print("  NVSwitch: 未検出（1x H100 SXM5では正常）")
        else:
            print("  トポロジー情報取得失敗")
            
    except Exception as e:
        print(f"  ❌ Fabric Manager確認エラー: {e}")

def main():
    """メイン実行"""
    print("🚀 1x H100 SXM5 CUDA初期化デバッグ開始")
    print(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 環境デバッグ
    debug_cuda_environment()
    
    # Fabric Manager確認
    debug_fabric_manager()
    
    # PyTorch CUDA初期化
    pytorch_success = debug_pytorch_cuda_init()
    
    print("\n" + "=" * 60)
    print("🎯 デバッグ結果サマリー:")
    if pytorch_success:
        print("  ✅ PyTorch CUDA初期化成功")
        print("  ✅ 1x H100 SXM5環境正常")
        print("  🚀 訓練スクリプト実行可能")
    else:
        print("  ❌ PyTorch CUDA初期化失敗")
        print("  🔧 追加調査が必要")
    
    print("=" * 60)
    
    return pytorch_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)