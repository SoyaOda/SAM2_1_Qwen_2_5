#!/usr/bin/env python3
"""
軽量Phase 3B統合テスト（1x H100 SXM5対応）
Llama4ロード前の基本的な統合テスト
"""

# 🔥 PyTorchインポート前の環境準備（CUDA Error 802対策）
import sys
import os

# Step 1: CUDA Error 802対策用環境変数設定（PyTorchインポート前）
print("🔧 CUDA Error 802対策：PyTorchインポート前環境設定...")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

# Step 2: CUDA_VISIBLE_DEVICESが未設定の場合のみ設定
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("✅ 環境変数設定完了 - PyTorchインポート開始...")

# Step 3: PyTorchインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from PIL import Image
import numpy as np

# ロギング設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_cuda_functionality():
    """基本的なCUDA機能テスト"""
    print("\n🔧 基本的なCUDA機能テスト...")
    
    # 1. デバイス確認
    if not torch.cuda.is_available():
        print("❌ CUDA利用不可")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA デバイス数: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {device_name}")
    
    # 2. 基本的なテンソル操作
    device = torch.device("cuda:0")
    
    # テンソル作成とGPU転送
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    # 行列乗算
    start_time = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"✅ GPU行列乗算: {gpu_time:.4f}秒")
    
    # メモリ使用量確認
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    print(f"✅ GPU メモリ: {allocated:.1f}MB (割当) / {cached:.1f}MB (予約)")
    
    return True

def test_lightweight_model_components():
    """軽量モデルコンポーネントテスト"""
    print("\n🔧 軽量モデルコンポーネントテスト...")
    
    device = torch.device("cuda:0")
    batch_size = 1
    
    # 1. 軽量CNN（SAM2代替）
    class LightweightCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 256)
        
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # 2. 軽量Transformer（Q-Former代替）
    class LightweightTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
                num_layers=2
            )
            self.output_proj = nn.Linear(256, 512)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.output_proj(x)
            return x
    
    # 3. 統合モデル
    class LightweightIntegrationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = LightweightCNN()
            self.transformer = LightweightTransformer()
            self.fusion = nn.Linear(256 + 512, 1024)
            self.output = nn.Linear(1024, 1000)  # 1000クラス分類
        
        def forward(self, image, text):
            # 画像処理
            image_feat = self.cnn(image)
            
            # テキスト処理
            text_feat = self.transformer(text)
            text_feat = text_feat.mean(dim=1)  # 平均プーリング
            
            # 特徴融合
            combined = torch.cat([image_feat, text_feat], dim=1)
            fused = F.relu(self.fusion(combined))
            output = self.output(fused)
            
            return output
    
    try:
        # モデル初期化
        model = LightweightIntegrationModel().to(device)
        
        # テストデータ
        test_image = torch.randn(batch_size, 3, 224, 224).to(device)
        test_text = torch.randint(0, 1000, (batch_size, 32)).to(device)
        
        # 推論テスト
        with torch.no_grad():
            start_time = time.time()
            output = model(test_image, test_text)
            inference_time = time.time() - start_time
        
        print(f"✅ 軽量統合モデル推論: {inference_time:.4f}秒")
        print(f"✅ 出力形状: {output.shape}")
        
        # 学習テスト
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        target = torch.randint(0, 1000, (batch_size,)).to(device)
        
        start_time = time.time()
        optimizer.zero_grad()
        output = model(test_image, test_text)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_time = time.time() - start_time
        
        print(f"✅ 軽量統合モデル学習: {train_time:.4f}秒")
        print(f"✅ 損失: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 軽量モデルテストエラー: {e}")
        return False

def test_phase3b_core_functions():
    """Phase 3B核心機能の軽量テスト"""
    print("\n🔧 Phase 3B核心機能軽量テスト...")
    
    device = torch.device("cuda:0")
    batch_size = 1
    
    # 1. 多重解像度特徴融合（簡易版）
    class SimpleMultiResolutionFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(256, 512, 3, padding=1)
            self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
            self.conv3 = nn.Conv2d(1024, 512, 3, padding=1)
            self.output_conv = nn.Conv2d(512, 64, 3, padding=1)
        
        def forward(self, features):
            # 3つの解像度の特徴を処理
            f1 = F.relu(self.conv1(features[0]))  # 高解像度
            f2 = F.relu(self.conv2(features[1]))  # 中解像度
            f3 = F.relu(self.conv3(features[2]))  # 低解像度
            
            # 解像度統一
            f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
            f3_up = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=False)
            
            # 特徴融合
            fused = f1 + f2_up + f3_up
            output = self.output_conv(fused)
            
            return output
    
    # 2. デュアルパスウェイデコーダ（簡易版）
    class SimpleDualPathwayDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.main_path = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1)
            )
            self.aux_path = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1)
            )
            self.fusion_weight = nn.Parameter(torch.tensor(0.7))
        
        def forward(self, x):
            main_out = self.main_path(x)
            aux_out = self.aux_path(x)
            
            # 重み付き融合
            fused = self.fusion_weight * main_out + (1 - self.fusion_weight) * aux_out
            
            return {
                'main_output': main_out,
                'aux_output': aux_out,
                'fused_output': fused
            }
    
    # 3. OHEM損失（簡易版）
    class SimpleOHEMLoss(nn.Module):
        def __init__(self, hard_ratio=0.25):
            super().__init__()
            self.hard_ratio = hard_ratio
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        def forward(self, pred, target):
            # 基本損失計算
            loss = self.bce_loss(pred, target)
            
            # 困難例選択
            batch_size = pred.size(0)
            num_hard = int(batch_size * self.hard_ratio)
            
            if num_hard > 0:
                loss_flat = loss.view(batch_size, -1).mean(dim=1)
                _, hard_indices = torch.topk(loss_flat, num_hard)
                hard_loss = loss_flat[hard_indices].mean()
            else:
                hard_loss = loss.mean()
            
            return hard_loss
    
    try:
        # 統合テスト
        multiresolution_fusion = SimpleMultiResolutionFusion().to(device)
        dual_pathway_decoder = SimpleDualPathwayDecoder().to(device)
        ohem_loss = SimpleOHEMLoss().to(device)
        
        # テストデータ
        test_features = [
            torch.randn(batch_size, 256, 64, 64).to(device),   # 高解像度
            torch.randn(batch_size, 512, 32, 32).to(device),   # 中解像度
            torch.randn(batch_size, 1024, 16, 16).to(device)   # 低解像度
        ]
        
        # 推論テスト
        with torch.no_grad():
            # 多重解像度特徴融合
            fused_features = multiresolution_fusion(test_features)
            print(f"✅ 多重解像度特徴融合: {fused_features.shape}")
            
            # デュアルパスウェイデコーダ
            decoder_output = dual_pathway_decoder(fused_features)
            print(f"✅ デュアルパスウェイデコーダ: {decoder_output['fused_output'].shape}")
            
            # OHEM損失
            target = torch.randint(0, 2, decoder_output['fused_output'].shape).float().to(device)
            loss = ohem_loss(decoder_output['fused_output'], target)
            print(f"✅ OHEM損失: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 3B核心機能テストエラー: {e}")
        return False

def main():
    """メイン実行"""
    print("🚀 軽量Phase 3B統合テスト開始（1x H100 SXM5対応）")
    print("=" * 60)
    
    # 1. 基本CUDA機能テスト
    cuda_success = test_basic_cuda_functionality()
    if not cuda_success:
        print("❌ 基本CUDA機能テスト失敗")
        return False
    
    # 2. 軽量モデルコンポーネントテスト
    model_success = test_lightweight_model_components()
    if not model_success:
        print("❌ 軽量モデルコンポーネントテスト失敗")
        return False
    
    # 3. Phase 3B核心機能テスト
    phase3b_success = test_phase3b_core_functions()
    if not phase3b_success:
        print("❌ Phase 3B核心機能テスト失敗")
        return False
    
    print("\n" + "=" * 60)
    print("🎯 軽量Phase 3B統合テスト結果:")
    print("  ✅ 基本CUDA機能: 正常")
    print("  ✅ 軽量モデルコンポーネント: 正常")
    print("  ✅ Phase 3B核心機能: 正常")
    print("  🚀 1x H100 SXM5環境での動作確認完了")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)