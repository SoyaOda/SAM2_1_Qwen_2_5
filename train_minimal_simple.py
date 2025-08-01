#!/usr/bin/env python3
"""
SAM2.1 + Qwen2.5-VL統合モデル最小学習スクリプト（超簡易版）
NaN問題を回避するための最小実装
"""
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# プロジェクトルートとsam2パスを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 超簡易モデル（テスト用）
class SimpleSegmentationModel(nn.Module):
    """最小学習テスト用の簡易セグメンテーションモデル"""
    
    def __init__(self, hidden_dim=256, img_size=224):
        super().__init__()
        self.img_size = img_size
        
        # 簡易エンコーダ（画像全体の特徴を抽出）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B, 64, 1, 1]
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, hidden_dim)  # [B, 256]
        )
        
        # マスクデコーダ（特徴からマスクを生成）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, img_size * img_size),
        )
        
        # 重みの初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みを小さく初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, image_tensor):
        """
        Args:
            image_tensor: [B, 3, H, W]
        Returns:
            mask_logits: [B, H, W]
        """
        # エンコード
        features = self.encoder(image_tensor)  # [B, 256]
        
        # デコード
        mask_logits = self.decoder(features)  # [B, H*W]
        mask_logits = mask_logits.view(-1, self.img_size, self.img_size)  # [B, H, W]
        
        return mask_logits


def create_dummy_data(num_samples: int = 3, img_size: int = 224) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    ダミーデータセットの作成（テンソル版）
    
    Returns:
        [(画像テンソル, マスクテンソル), ...]のリスト
    """
    data = []
    
    for i in range(num_samples):
        # 画像生成（黒背景に白い円）
        image = Image.new('RGB', (img_size, img_size), color='black')
        draw = ImageDraw.Draw(image)
        
        # 中央に円を描画
        radius = 30 + i * 10
        center = img_size // 2
        draw.ellipse([center - radius, center - radius, 
                     center + radius, center + radius], 
                    fill='white')
        
        # 画像をテンソルに変換
        img_array = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
        img_tensor = torch.FloatTensor(img_array) / 255.0  # 0-1に正規化
        
        # 正解マスク生成（円形）
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        y, x = np.ogrid[:img_size, :img_size]
        circle_mask = (x - center)**2 + (y - center)**2 <= radius**2
        mask[circle_mask] = 1.0
        mask_tensor = torch.FloatTensor(mask)
        
        data.append((img_tensor, mask_tensor))
    
    return data


def main():
    """メイン学習ループ（超簡易版）"""
    print("=" * 70)
    print("🚀 超簡易セグメンテーション学習テスト")
    print("=" * 70)
    
    # 設定
    img_size = 224
    num_epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ダミーデータ生成
    print("\n📦 ダミーデータ生成中...")
    dummy_data = create_dummy_data(num_samples=3, img_size=img_size)
    print(f"✅ {len(dummy_data)}個のダミーデータを生成")
    
    # モデル初期化
    print("\n🤖 簡易モデル初期化中...")
    model = SimpleSegmentationModel(hidden_dim=256, img_size=img_size).to(device)
    model.train()
    print(f"✅ モデル初期化完了 (デバイス: {device})")
    
    # 損失関数とオプティマイザ
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学習履歴
    training_history = []
    
    print(f"\n🎯 学習開始 (エポック数: {num_epochs})")
    print("-" * 50)
    
    # 学習ループ
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for img_tensor, mask_tensor in dummy_data:
            # デバイスに転送
            img = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
            mask = mask_tensor.unsqueeze(0).to(device)  # [1, H, W]
            
            # Forward pass
            mask_logits = model(img)  # [1, H, W]
            
            # 損失計算
            loss = criterion(mask_logits, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # エポック平均
        avg_loss = epoch_loss / len(dummy_data)
        training_history.append(avg_loss)
        
        # 進捗表示
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    
    print("-" * 50)
    print("✅ 学習完了！")
    
    # 最終評価
    print("\n📊 最終評価:")
    model.eval()
    
    with torch.no_grad():
        total_iou = 0.0
        
        for i, (img_tensor, mask_tensor) in enumerate(dummy_data):
            img = img_tensor.unsqueeze(0).to(device)
            mask = mask_tensor.to(device)
            
            # 推論
            mask_logits = model(img).squeeze(0)
            mask_pred = (torch.sigmoid(mask_logits) > 0.5).float()
            
            # IoU計算
            intersection = (mask_pred * mask).sum()
            union = mask_pred.sum() + mask.sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
            total_iou += iou
            
            print(f"  サンプル{i+1}: IoU={iou:.3f}")
    
    avg_iou = total_iou / len(dummy_data)
    print(f"  平均IoU: {avg_iou:.3f}")
    
    # 学習曲線の可視化
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), training_history, 'b-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    # 結果保存
    results_dir = os.path.join(project_root, "training_results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.savefig(os.path.join(results_dir, f"simple_training_{timestamp}.png"))
    plt.close()
    
    print(f"\n🎉 超簡易学習テスト完了！")
    
    # 損失が減少したかチェック
    if len(training_history) > 1:
        initial_loss = training_history[0]
        final_loss = training_history[-1]
        
        if final_loss < initial_loss:
            print(f"✅ 損失が減少: {initial_loss:.4f} → {final_loss:.4f} (改善率: {(1-final_loss/initial_loss)*100:.1f}%)")
        else:
            print(f"⚠️ 損失が増加: {initial_loss:.4f} → {final_loss:.4f}")


if __name__ == "__main__":
    main()