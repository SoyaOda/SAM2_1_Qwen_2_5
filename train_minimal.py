#!/usr/bin/env python3
"""
SAM2.1 + Qwen2.5-VL統合モデル最小学習スクリプト
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
    """メイン学習ループ"""
    print("=" * 70)
    print("🚀 SAM2.1 + Qwen2.5-VL最小学習開始")
    print("=" * 70)
    
    # 設定
    config = get_config('development')
    model_config = config.get_model_config()
    
    # ダミーデータ生成（段階的テスト）
    print("\n📦 ダミーデータ生成中...")
    dummy_data = create_dummy_data(num_samples=3, img_size=224)  # 3つのサンプルでテスト
    print(f"✅ {len(dummy_data)}個のダミーデータを生成")
    
    # モデル初期化
    print("\n🤖 モデル初期化中...")
    model = create_sam_qwen_model(model_config)
    model.train()
    
    # 学習用設定
    model.configure_for_training(
        freeze_sam_encoder=True,  # SAMエンコーダは凍結
        freeze_qwen_vision=True   # Qwenビジョンエンコーダも凍結
    )
    
    # オプティマイザ設定（最も安全な学習率）
    trainable_params = model.get_trainable_parameters()
    optimizer = optim.Adam(trainable_params, lr=1e-7, eps=1e-8, weight_decay=1e-5)  # 極小学習率と重み減衰
    
    # 学習設定（段階的拡張テスト）
    num_epochs = 3  # 複数エポックテスト
    device = model.device
    
    # 結果保存用
    results_dir = os.path.join(project_root, "training_results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 学習履歴
    training_history = []
    
    print(f"\n🎯 学習開始 (エポック数: {num_epochs})")
    print("-" * 50)
    
    # mask_headを事前に作成して固定（NaN問題対策）
    print("🔧 mask_head初期化中...")
    with torch.no_grad():
        # ダミーの前向き計算でmask_headを作成
        dummy_image = dummy_data[0][0]
        dummy_messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": dummy_data[0][1]}
            ]
        }]
        # 推論モードで一度実行してmask_headを初期化
        model.eval()
        _ = model.forward_with_segmentation(
            images=dummy_image,
            messages=dummy_messages,
            max_new_tokens=32
        )
        model.train()
    print("✅ mask_head初期化完了（動的作成を回避）")
    
    # 学習ループ
    for epoch in range(num_epochs):
        # エポック間でモデル状態をリセット（NaN対策）
        if epoch > 0:
            print(f"\n🔄 エポック {epoch+1} 開始: モデル状態リセット中...")
            model.eval()
            torch.cuda.empty_cache()  # GPU キャッシュクリア
            model.train()
            # mask_headを再初期化
            with torch.no_grad():
                dummy_image = dummy_data[0][0]
                dummy_messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": dummy_data[0][1]}
                    ]
                }]
                model.eval()
                _ = model.forward_with_segmentation(
                    images=dummy_image,
                    messages=dummy_messages,
                    max_new_tokens=32
                )
                model.train()
            print("✅ モデル状態リセット完了")
        
        epoch_loss = 0.0
        epoch_metrics = {'iou': 0.0, 'dice': 0.0}
        
        # 各サンプルで学習（完全独立処理）
        accumulated_loss = 0.0
        for i, (image, instruction, target_mask) in enumerate(dummy_data):
            # 各サンプル毎にmask_headを再初期化（NaN完全回避）
            print(f"\n🔧 サンプル {i+1}: mask_head再初期化中...")
            model.eval()
            torch.cuda.empty_cache()  # GPU キャッシュクリア
            with torch.no_grad():
                _ = model.forward_with_segmentation(
                    images=image,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction}
                        ]
                    }],
                    max_new_tokens=32
                )
            model.train()
            optimizer.zero_grad()
            print(f"✅ サンプル {i+1} mask_head再初期化完了")
            
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
            
            # サンプル処理前のデバッグ情報
            print(f"\n--- サンプル {i+1} 処理開始 ---")
            print(f"  - 画像タイプ: {type(image)}")
            print(f"  - ターゲットマスク統計: min={target_mask.min():.6f}, max={target_mask.max():.6f}, mean={target_mask.mean():.6f}")
            print(f"  - ターゲットマスクNaN/Inf: NaN={np.isnan(target_mask).sum()}, Inf={np.isinf(target_mask).sum()}")
            
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
            
            # デバッグ: 損失の状態を確認
            if i == 0:  # 各エポックの最初のサンプルでデバッグ
                print(f"\nエポック {epoch+1} デバッグ情報:")
                print(f"  - loss type: {type(loss)}")
                print(f"  - loss value: {loss}")
                print(f"  - loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'N/A'}")
                print(f"  - has_masks: {results.get('has_masks', False)}")
                print(f"  - losses dict: {results.get('losses', {})}")
                
                # より詳細な損失成分の確認
                if 'losses' in results:
                    for loss_name, loss_value in results['losses'].items():
                        if torch.isnan(loss_value) or torch.isinf(loss_value):
                            print(f"  ❌ NaN/Inf検出: {loss_name} = {loss_value}")
                        else:
                            print(f"  ✅ 正常: {loss_name} = {loss_value:.6f}")
                
                # マスクの統計情報
                if results.get('has_masks') and len(results['masks']) > 0:
                    pred_mask = results['masks'][0]
                    if isinstance(pred_mask, np.ndarray):
                        pred_mask = torch.from_numpy(pred_mask)
                    print(f"  - マスク統計: min={pred_mask.min():.6f}, max={pred_mask.max():.6f}, mean={pred_mask.mean():.6f}")
                    print(f"  - マスク形状: {pred_mask.shape}")
                    print(f"  - NaN/Inf含有: NaN={torch.isnan(pred_mask).sum()}, Inf={torch.isinf(pred_mask).sum()}")
                
                # 投影層の出力確認
                if 'seg_query' in results:
                    seg_query = results['seg_query']
                    print(f"  - seg_query統計: min={seg_query.min():.6f}, max={seg_query.max():.6f}, mean={seg_query.mean():.6f}")
                    print(f"  - seg_query NaN/Inf: NaN={torch.isnan(seg_query).sum()}, Inf={torch.isinf(seg_query).sum()}")
            
            # NaN検出時の早期停止
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ エポック {epoch+1}, サンプル {i+1} でNaN/Inf検出: {loss}")
                print("早期停止します。")
                return
            
            # Backward pass
            
            # 勾配計算前のパラメータ状態確認
            if i == 0 and epoch == 0:
                print(f"\n勾配計算前パラメータチェック:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print(f"  ❌ パラメータ異常 {name}: NaN={torch.isnan(param).sum()}, Inf={torch.isinf(param).sum()}")
            
            loss.backward()
            
            # 勾配の詳細な監視
            max_grad_norm = 0.0
            nan_grad_params = []
            large_grad_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_grad_params.append((name, grad_norm))
                    elif grad_norm > 1.0:
                        large_grad_params.append((name, grad_norm))
            
            # デバッグ出力
            if i == 0:
                print(f"  - 最大勾配ノルム: {max_grad_norm:.6f}")
                if nan_grad_params:
                    print(f"  ❌ NaN/Inf勾配: {[f'{name}:{norm:.4f}' for name, norm in nan_grad_params]}")
                if large_grad_params:
                    print(f"  ⚠️ 大きな勾配: {[f'{name}:{norm:.4f}' for name, norm in large_grad_params[:3]]}")  # 最初の3つのみ表示
            
            # NaN勾配検出時の早期停止
            if nan_grad_params:
                print(f"\n❌ NaN勾配検出により学習を停止します")
                return
            
            # 極めて厳格な勾配クリッピング（NaN対策）
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.01)
            
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
                    vis_path = os.path.join(results_dir, f"result_{timestamp}.png")
                    plt.savefig(vis_path, dpi=150)
                    plt.close()
                    print(f"\n📸 結果を保存: {vis_path}")
    
    # 学習履歴の可視化
    if len(training_history) > 0:
        progress_path = os.path.join(results_dir, f"training_progress_{timestamp}.png")
        visualize_training_progress(training_history, progress_path)
    
    # 学習履歴をテキストで保存
    history_path = os.path.join(results_dir, f"training_history_{timestamp}.txt")
    with open(history_path, 'w') as f:
        f.write("Epoch\tLoss\tIoU\tDice\n")
        for h in training_history:
            f.write(f"{h['epoch']}\t{h['loss']:.4f}\t{h['iou']:.3f}\t{h['dice']:.3f}\n")
    print(f"📄 学習履歴を保存: {history_path}")
    
    print("\n🎉 最小学習テスト完了！")
    
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