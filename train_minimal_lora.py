#!/usr/bin/env python3
"""
SAM2.1 + Qwen2.5-VL統合モデル最小学習スクリプト（LoRA/QLoRA版）
o3_spec4.mdに基づくダミーデータでの学習検証
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
sam2_path = os.path.join(project_root, 'sam2')
if os.path.exists(sam2_path):
    sys.path.insert(0, sam2_path)

from config_qwen_sam import get_config
from model.sam_qwen_model import create_sam_qwen_model
from model.lora_config import create_lora_manager


def create_dummy_data(num_samples: int = 3, img_size: int = 224) -> List[Tuple[Image.Image, str, np.ndarray]]:
    """
    ダミーデータセットの作成
    
    Args:
        num_samples: サンプル数
        img_size: 画像サイズ
        
    Returns:
        [(画像, 指示文, 正解マスク), ...]のリスト
    """
    data = []
    
    # パターン1: 白い四角形
    for i in range(num_samples // 3):
        # 画像生成（黒背景に白い四角形）
        image = Image.new('RGB', (img_size, img_size), color='black')
        draw = ImageDraw.Draw(image)
        
        # ランダムな位置に四角形を描画
        size = 50 + i * 10
        x1 = (img_size - size) // 2 + i * 20
        y1 = (img_size - size) // 2
        x2 = x1 + size
        y2 = y1 + size
        draw.rectangle([x1, y1, x2, y2], fill='white')
        
        # 正解マスク生成
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        
        # 指示文
        instruction = "この画像の白い四角形をセグメントしてください。<SEG>"
        
        data.append((image, instruction, mask))
    
    # パターン2: 赤い円
    for i in range(num_samples // 3):
        # 画像生成（白背景に赤い円）
        image = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(image)
        
        # 円を描画
        radius = 40 + i * 10
        center_x = img_size // 2 + i * 15
        center_y = img_size // 2
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], 
                    fill='red', outline='darkred', width=2)
        
        # 正解マスク生成（円形）
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        y, x = np.ogrid[:img_size, :img_size]
        circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[circle_mask] = 1.0
        
        # 指示文
        instruction = "画像内の赤い円形の物体を検出してください。<SEG>"
        
        data.append((image, instruction, mask))
    
    # パターン3: 青い三角形
    remaining = num_samples - 2 * (num_samples // 3)
    for i in range(remaining):
        # 画像生成（グレー背景に青い三角形）
        image = Image.new('RGB', (img_size, img_size), color='gray')
        draw = ImageDraw.Draw(image)
        
        # 三角形を描画
        size = 60 + i * 10
        cx = img_size // 2
        cy = img_size // 2
        points = [
            (cx, cy - size//2),  # 上
            (cx - size//2, cy + size//2),  # 左下
            (cx + size//2, cy + size//2)   # 右下
        ]
        draw.polygon(points, fill='blue', outline='darkblue')
        
        # 正解マスク生成（三角形）
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        mask_img = Image.new('L', (img_size, img_size), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.polygon(points, fill=255)
        mask = np.array(mask_img) / 255.0
        
        # 指示文
        instruction = "青い三角形の領域をセグメンテーションしてください。<SEG>"
        
        data.append((image, instruction, mask))
    
    return data


def calculate_metrics(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, float]:
    """
    IoUとDice係数を計算
    
    Args:
        pred_mask: 予測マスク
        target_mask: 正解マスク
        
    Returns:
        メトリクス辞書
    """
    # 二値化（0.5閾値）
    pred_binary = (pred_mask > 0.5).float()
    target_binary = target_mask.float()
    
    # IoU計算
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection / (union + 1e-8)).item()
    
    # Dice係数計算
    dice = (2 * intersection / (pred_binary.sum() + target_binary.sum() + 1e-8)).item()
    
    return {'iou': iou, 'dice': dice}


def visualize_training_progress(epoch_data: List[Dict], save_path: str):
    """
    学習進捗の可視化
    
    Args:
        epoch_data: エポックごとのデータ
        save_path: 保存パス
    """
    epochs = [d['epoch'] for d in epoch_data]
    losses = [d['loss'] for d in epoch_data]
    ious = [d['iou'] for d in epoch_data]
    dices = [d['dice'] for d in epoch_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 損失のプロット
    ax1.plot(epochs, losses, 'b-', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # メトリクスのプロット
    ax2.plot(epochs, ious, 'g-', marker='s', label='IoU')
    ax2.plot(epochs, dices, 'r-', marker='^', label='Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Segmentation Metrics')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 学習進捗を保存: {save_path}")


def main():
    """メイン学習ループ（LoRA版）"""
    print("=" * 70)
    print("🚀 SAM2.1 + Qwen2.5-VL最小学習開始（LoRA版）")
    print("=" * 70)
    
    # 設定
    config = get_config('development')
    model_config = config.get_model_config()
    
    # LoRA設定の作成
    lora_manager = create_lora_manager(
        use_qlora=False,  # 通常のLoRA（QLoRAは簡略化のため無効）
        qwen_r=16,        # Qwen LoRAランク
        sam_r=16,         # SAM LoRAランク
        qwen_lora_dropout=0.05,
        sam_lora_dropout=0.1
    )
    
    # ダミーデータ生成（段階的テスト）
    print("\n📦 ダミーデータ生成中...")
    dummy_data = create_dummy_data(num_samples=3, img_size=224)  # 3つのサンプルでテスト
    print(f"✅ {len(dummy_data)}個のダミーデータを生成")
    
    # モデル初期化（LoRA設定を渡す）
    print("\n🤖 モデル初期化中（LoRA有効）...")
    model = create_sam_qwen_model(model_config, lora_config=lora_manager)
    model.train()
    
    # 学習用設定（LoRA前提のシンプル版）
    model.configure_for_training(
        freeze_sam_encoder=True  # SAMエンコーダは凍結
    )
    
    # オプティマイザ設定（LoRA用の学習率）
    trainable_params = model.get_trainable_parameters()
    optimizer = optim.Adam(trainable_params, lr=2e-4, eps=1e-8, weight_decay=1e-4)  # LoRA推奨学習率
    
    # 学習設定（段階的拡張テスト）
    num_epochs = 5  # LoRAでは高速に収束するため少なめ
    device = model.device
    
    # 結果保存用
    results_dir = os.path.join(project_root, "training_results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 学習履歴
    training_history = []
    
    print(f"\n🎯 学習開始 (エポック数: {num_epochs}, LoRA有効)")
    print("-" * 50)
    
    # mask_headはモデル初期化時に作成済み
    print("✅ mask_headは初期化済み")
    
    # 学習ループ
    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        epoch_metrics = {'iou': 0.0, 'dice': 0.0}
        
        # 各サンプルで学習（完全独立処理）
        accumulated_loss = 0.0
        for i, (image, instruction, target_mask) in enumerate(dummy_data):
            # オプティマイザの勾配をクリア
            optimizer.zero_grad()
            
            # メッセージ形式に変換
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }]
            
            # ターゲットマスクをテンソルに変換
            target_mask_tensor = torch.from_numpy(target_mask).float().to(device)
            
            
            # Forward pass（既存のmask_headを使用）
            try:
                results = model.forward_train(
                    images=image,
                    messages=messages,
                    target_masks=target_mask_tensor,
                    max_new_tokens=32
                )
            except Exception as e:
                print(f"\n❌ Forward pass中にエラー: {str(e)}")
                print(f"エポック {epoch+1}, サンプル {i+1}")
                return
            
            # 損失取得
            loss = results['total_loss']
            
            
            # NaN検出時の早期停止
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ エポック {epoch+1}, サンプル {i+1} でNaN/Inf検出: {loss}")
                print("早期停止します。")
                return
            
            # Backward pass
            loss.backward()
            
            # 勾配クリッピング（LoRAでは小さめの値）
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            
            optimizer.step()
            
            # 損失を記録
            epoch_loss += loss.item()
            
            # メトリクス計算（推論モードで）
            with torch.no_grad():
                if results['has_masks'] and len(results['masks']) > 0:
                    pred_mask = results['masks'][0]
                    if isinstance(pred_mask, np.ndarray):
                        pred_mask = torch.from_numpy(pred_mask).float()
                    pred_mask = pred_mask.to(device)
                    
                    metrics = calculate_metrics(pred_mask, target_mask_tensor)
                    epoch_metrics['iou'] += metrics['iou']
                    epoch_metrics['dice'] += metrics['dice']
        
        # エポック平均を計算
        avg_loss = epoch_loss / len(dummy_data)
        avg_iou = epoch_metrics['iou'] / len(dummy_data)
        avg_dice = epoch_metrics['dice'] / len(dummy_data)
        
        # 結果を記録
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice
        })
        
        # 進捗表示
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | IoU: {avg_iou:.3f} | Dice: {avg_dice:.3f}")
    
    print("-" * 50)
    print("✅ 学習完了！")
    
    # LoRAパラメータのみの統計を表示
    if hasattr(model, 'lora_config'):
        print("\n📊 LoRAパラメータ統計:")
        lora_params = 0
        qwen_lora_params = 0
        sam_lora_params = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params += param.numel()
                if 'qwen' in name:
                    qwen_lora_params += param.numel()
                elif 'sam' in name or 'mask_decoder' in name:
                    sam_lora_params += param.numel()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - 総LoRAパラメータ数: {lora_params:,}")
        print(f"    - Qwen LoRA: {qwen_lora_params:,}")
        print(f"    - SAM LoRA: {sam_lora_params:,}")
        print(f"  - 総パラメータ数: {total_params:,}")
        print(f"  - LoRAパラメータ比率: {lora_params/total_params*100:.2f}%")
    
    # 最終評価
    print("\n📊 最終評価:")
    model.eval()
    
    with torch.no_grad():
        for i, (image, instruction, target_mask) in enumerate(dummy_data):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }]
            
            # 推論実行
            results = model.forward_with_segmentation(
                images=image,
                messages=messages,
                max_new_tokens=32
            )
            
            if results['has_masks'] and len(results['masks']) > 0:
                pred_mask = results['masks'][0]
                if isinstance(pred_mask, np.ndarray):
                    pred_mask = torch.from_numpy(pred_mask).float()
                
                target_mask_tensor = torch.from_numpy(target_mask).float()
                metrics = calculate_metrics(pred_mask, target_mask_tensor)
                
                print(f"  サンプル{i+1}: IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}")
                
                # マスク可視化（最初のサンプルのみ）
                if i == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    
                    # 元画像
                    axes[0].imshow(image)
                    axes[0].set_title("Input Image")
                    axes[0].axis('off')
                    
                    # 正解マスク
                    axes[1].imshow(target_mask, cmap='gray')
                    axes[1].set_title("Ground Truth")
                    axes[1].axis('off')
                    
                    # 予測マスク
                    axes[2].imshow(pred_mask.cpu().numpy(), cmap='gray')
                    axes[2].set_title(f"Prediction (IoU={metrics['iou']:.3f})")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    vis_path = os.path.join(results_dir, f"result_lora_{timestamp}.png")
                    plt.savefig(vis_path, dpi=150)
                    plt.close()
                    print(f"\n📸 結果を保存: {vis_path}")
    
    # 学習履歴の可視化
    if len(training_history) > 0:
        progress_path = os.path.join(results_dir, f"training_progress_lora_{timestamp}.png")
        visualize_training_progress(training_history, progress_path)
    
    # 学習履歴をテキストで保存
    history_path = os.path.join(results_dir, f"training_history_lora_{timestamp}.txt")
    with open(history_path, 'w') as f:
        f.write("Epoch\tLoss\tIoU\tDice\n")
        for h in training_history:
            f.write(f"{h['epoch']}\t{h['loss']:.4f}\t{h['iou']:.3f}\t{h['dice']:.3f}\n")
    print(f"📄 学習履歴を保存: {history_path}")
    
    print("\n🎉 最小学習テスト（LoRA版）完了！")
    
    # 損失が減少したかチェック
    if len(training_history) > 1:
        initial_loss = training_history[0]['loss']
        final_loss = training_history[-1]['loss']
        
        if final_loss < initial_loss:
            print(f"✅ 損失が減少: {initial_loss:.4f} → {final_loss:.4f} (改善率: {(1-final_loss/initial_loss)*100:.1f}%)")
        else:
            print(f"⚠️ 損失が増加: {initial_loss:.4f} → {final_loss:.4f}")


if __name__ == "__main__":
    main()