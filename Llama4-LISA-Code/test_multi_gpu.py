#!/usr/bin/env python3
"""
マルチGPU環境テスト（2x H100対応）
"""

import os
import sys

# CUDA Error 802対策
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

# 2GPU環境設定
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

print("🔧 マルチGPU環境テスト開始...")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch
import torch.nn as nn
import time

def test_multi_gpu():
    """マルチGPU動作テスト"""
    print("\n📊 マルチGPU動作テスト:")
    
    if not torch.cuda.is_available():
        print("❌ CUDA利用不可")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✅ 検出されたGPU数: {device_count}")
    
    # 各GPUの情報表示
    for i in range(device_count):
        try:
            device_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {device_name} ({memory_total:.1f}GB)")
        except Exception as e:
            print(f"  GPU {i}: エラー - {e}")
    
    # 各GPUでの基本計算テスト
    print("\n📊 各GPUでの基本計算テスト:")
    for i in range(device_count):
        try:
            device = f"cuda:{i}"
            
            # テンソル作成と計算
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"  GPU {i}: 行列乗算 {gpu_time:.4f}秒")
            
            # メモリ使用量確認
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            cached = torch.cuda.memory_reserved(i) / 1024**2
            print(f"  GPU {i}: メモリ {allocated:.1f}MB / {cached:.1f}MB")
            
        except Exception as e:
            print(f"  GPU {i}: エラー - {e}")
    
    return True

def test_data_parallel():
    """データ並列処理テスト"""
    print("\n📊 データ並列処理テスト:")
    
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("⚠️ GPU数が不足（2台以上必要）")
        return False
    
    try:
        # 簡単なモデル定義
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1000, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 10)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # モデルをDataParallelで包装
        model = SimpleModel()
        model = nn.DataParallel(model)
        model = model.cuda()
        
        print(f"✅ DataParallel設定完了 (GPU数: {device_count})")
        
        # テストデータ
        batch_size = 64
        test_input = torch.randn(batch_size, 1000).cuda()
        
        # 推論テスト
        with torch.no_grad():
            start_time = time.time()
            output = model(test_input)
            inference_time = time.time() - start_time
        
        print(f"✅ DataParallel推論: {inference_time:.4f}秒")
        print(f"✅ 出力形状: {output.shape}")
        
        # 学習テスト
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        target = torch.randint(0, 10, (batch_size,)).cuda()
        
        start_time = time.time()
        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_time = time.time() - start_time
        
        print(f"✅ DataParallel学習: {train_time:.4f}秒")
        print(f"✅ 損失: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataParallel テストエラー: {e}")
        return False

def test_gpu_to_gpu_communication():
    """GPU間通信テスト"""
    print("\n📊 GPU間通信テスト:")
    
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("⚠️ GPU数が不足（2台以上必要）")
        return False
    
    try:
        # GPU 0でテンソル作成
        x = torch.randn(1000, 1000, device='cuda:0')
        print(f"✅ GPU 0でテンソル作成: {x.shape}")
        
        # GPU 1にコピー
        start_time = time.time()
        y = x.to('cuda:1')
        copy_time = time.time() - start_time
        
        print(f"✅ GPU 0→1コピー: {copy_time:.4f}秒")
        
        # GPU 1で計算
        start_time = time.time()
        z = y * 2
        calc_time = time.time() - start_time
        
        print(f"✅ GPU 1で計算: {calc_time:.4f}秒")
        
        # GPU 0に結果を戻す
        start_time = time.time()
        result = z.to('cuda:0')
        back_time = time.time() - start_time
        
        print(f"✅ GPU 1→0コピー: {back_time:.4f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU間通信エラー: {e}")
        return False

def main():
    """メイン実行"""
    print("🚀 マルチGPU環境テスト開始（2x H100対応）")
    print("=" * 60)
    
    # 1. マルチGPU動作テスト
    multi_gpu_success = test_multi_gpu()
    
    # 2. データ並列処理テスト
    data_parallel_success = test_data_parallel()
    
    # 3. GPU間通信テスト
    communication_success = test_gpu_to_gpu_communication()
    
    print("\n" + "=" * 60)
    print("🎯 マルチGPU環境テスト結果:")
    print(f"  ✅ マルチGPU動作: {'成功' if multi_gpu_success else '失敗'}")
    print(f"  ✅ データ並列処理: {'成功' if data_parallel_success else '失敗'}")
    print(f"  ✅ GPU間通信: {'成功' if communication_success else '失敗'}")
    
    overall_success = multi_gpu_success and data_parallel_success and communication_success
    print(f"  🚀 総合評価: {'成功' if overall_success else '失敗'}")
    print("=" * 60)
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)