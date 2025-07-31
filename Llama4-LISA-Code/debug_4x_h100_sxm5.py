#!/usr/bin/env python3
"""
4x H100 SXM5 + NVSwitch専用CUDA初期化デバッグスクリプト
Fabric Manager問題の詳細調査
"""

import sys
import os
import subprocess
import time
import ctypes

def check_nvidia_services():
    """NVIDIA関連サービスの詳細確認"""
    print("🔍 NVIDIA関連サービス詳細確認")
    print("=" * 60)
    
    services = [
        'nvidia-fabricmanager',
        'nvidia-persistenced',
        'nvidia-modeset',
        'nvidia-drm'
    ]
    
    for service in services:
        try:
            result = subprocess.run(['systemctl', 'is-active', service], 
                                  capture_output=True, text=True)
            status = result.stdout.strip()
            print(f"  {service}: {status}")
            
            if service == 'nvidia-fabricmanager':
                # Fabric Manager詳細確認
                result = subprocess.run(['systemctl', 'status', service], 
                                      capture_output=True, text=True)
                print(f"    詳細: {result.stdout.split('Active:')[1].split('Main PID:')[0].strip() if 'Active:' in result.stdout else 'N/A'}")
                
        except Exception as e:
            print(f"  {service}: エラー - {e}")

def check_nvswitch_status():
    """NVSwitch状態の詳細確認"""
    print("\n🔍 NVSwitch状態詳細確認")
    print("=" * 60)
    
    try:
        # NVSwitch情報
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  NVLink状態:")
            print(result.stdout[:500])  # 最初の500文字
        else:
            print(f"  ❌ NVLink状態確認失敗: {result.stderr}")
            
        # GPU間通信確認
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  GPU間通信:")
            print(result.stdout[:500])  # 最初の500文字
        else:
            print(f"  ❌ GPU間通信確認失敗: {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ NVSwitch確認エラー: {e}")

def manual_fabric_manager_fix():
    """Fabric Manager手動修復試行"""
    print("\n🔧 Fabric Manager手動修復試行")
    print("=" * 60)
    
    commands = [
        # 1. 既存プロセス停止
        ['sudo', 'systemctl', 'stop', 'nvidia-fabricmanager'],
        ['sudo', 'pkill', '-f', 'nvidia-fabricmanager'],
        
        # 2. NVIDIAドライバーリロード
        ['sudo', 'rmmod', 'nvidia_uvm'],
        ['sudo', 'rmmod', 'nvidia_drm'],
        ['sudo', 'rmmod', 'nvidia_modeset'],
        ['sudo', 'rmmod', 'nvidia'],
        
        # 3. ドライバー再ロード
        ['sudo', 'modprobe', 'nvidia'],
        ['sudo', 'modprobe', 'nvidia_modeset'],
        ['sudo', 'modprobe', 'nvidia_drm'],
        ['sudo', 'modprobe', 'nvidia_uvm'],
        
        # 4. Fabric Manager再起動
        ['sudo', 'systemctl', 'start', 'nvidia-fabricmanager'],
        ['sudo', 'systemctl', 'enable', 'nvidia-fabricmanager']
    ]
    
    for cmd in commands:
        try:
            print(f"  実行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"    ✅ 成功")
            else:
                print(f"    ⚠️ 警告: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"    ⏰ タイムアウト")
        except Exception as e:
            print(f"    ❌ エラー: {e}")
    
    # 最終確認
    print("\n📊 修復後状態確認:")
    try:
        result = subprocess.run(['systemctl', 'is-active', 'nvidia-fabricmanager'], 
                              capture_output=True, text=True)
        print(f"  Fabric Manager: {result.stdout.strip()}")
        
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  GPU検出: 成功")
        else:
            print(f"  GPU検出: 失敗 - {result.stderr}")
            
    except Exception as e:
        print(f"  確認エラー: {e}")

def test_pytorch_with_env_variations():
    """異なる環境変数パターンでPyTorchテスト"""
    print("\n🔧 異なる環境変数パターンでPyTorchテスト")
    print("=" * 60)
    
    # 環境変数パターン
    patterns = [
        {
            'name': 'パターン1: 基本設定',
            'env': {
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'PYTORCH_NVML_BASED_CUDA_CHECK': '1',
                'CUDA_VISIBLE_DEVICES': '0'
            }
        },
        {
            'name': 'パターン2: 全GPU',
            'env': {
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'PYTORCH_NVML_BASED_CUDA_CHECK': '1',
                'CUDA_VISIBLE_DEVICES': '0,1,2,3'
            }
        },
        {
            'name': 'パターン3: NVSwitchモード',
            'env': {
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'PYTORCH_NVML_BASED_CUDA_CHECK': '1',
                'CUDA_VISIBLE_DEVICES': '0,1,2,3',
                'NCCL_NVLS_ENABLE': '1',
                'NCCL_P2P_DISABLE': '0'
            }
        },
        {
            'name': 'パターン4: 単一GPU強制',
            'env': {
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'PYTORCH_NVML_BASED_CUDA_CHECK': '0',
                'CUDA_VISIBLE_DEVICES': '0',
                'CUDA_LAUNCH_BLOCKING': '1'
            }
        }
    ]
    
    for pattern in patterns:
        print(f"\n📊 {pattern['name']}:")
        
        # 環境変数設定
        for key, value in pattern['env'].items():
            os.environ[key] = value
            print(f"  {key}: {value}")
        
        try:
            # PyTorch再インポート（新しいプロセスで実行）
            test_script = f"""
import os
{'; '.join([f"os.environ['{k}'] = '{v}'" for k, v in pattern['env'].items()])}

import torch
print(f"  torch.cuda.is_available(): {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"  torch.cuda.device_count(): {{torch.cuda.device_count()}}")
    try:
        test_tensor = torch.randn(10, 10, device='cuda:0')
        result = test_tensor.sum()
        print(f"  GPU計算テスト: 成功 ({{result.item():.4f}})")
    except Exception as e:
        print(f"  GPU計算テスト: 失敗 - {{e}}")
"""
            
            result = subprocess.run([sys.executable, '-c', test_script], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"  ❌ テスト失敗: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ テストタイムアウト")
        except Exception as e:
            print(f"  ❌ テストエラー: {e}")

def main():
    """メイン実行"""
    print("🚀 4x H100 SXM5 + NVSwitch詳細デバッグ開始")
    print(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. NVIDIA関連サービス確認
    check_nvidia_services()
    
    # 2. NVSwitch状態確認
    check_nvswitch_status()
    
    # 3. Fabric Manager手動修復
    manual_fabric_manager_fix()
    
    # 4. 異なる環境変数パターンでテスト
    test_pytorch_with_env_variations()
    
    print("\n" + "=" * 60)
    print("🎯 4x H100 SXM5デバッグ完了")
    print("=" * 60)

if __name__ == "__main__":
    main()