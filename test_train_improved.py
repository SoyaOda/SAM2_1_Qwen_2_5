"""
改善されたNaN対策を含む訓練テスト
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from model import create_sam_qwen_model, create_lora_manager

def create_dummy_data(num_samples=3, img_size=224):
    """ダミーデータセットを作成"""
    dummy_data = []
    
    for i in range(num_samples):
        # 簡単な画像を生成
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        
        if i == 0:
            # 白い四角形
            img[50:150, 50:150] = 255
            mask[50:150, 50:150] = 1.0
            instruction = "この画像の白い四角形をセグメントしてください。<SEG>"
        elif i == 1:
            # 赤い円
            center = (img_size // 2, img_size // 2)
            radius = 40
            y, x = np.ogrid[:img_size, :img_size]
            circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[circle_mask] = [255, 0, 0]
            mask[circle_mask] = 1.0
            instruction = "この画像の赤い円をセグメントしてください。<SEG>"
        else:
            # 青い三角形
            triangle = np.array([(100, 50), (50, 150), (150, 150)])
            from PIL import ImageDraw
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            draw.polygon([tuple(p) for p in triangle], fill=(0, 0, 255))
            img = np.array(pil_img)
            
            # 三角形のマスク
            pil_mask = Image.new('L', (img_size, img_size), 0)
            draw_mask = ImageDraw.Draw(pil_mask)
            draw_mask.polygon([tuple(p) for p in triangle], fill=255)
            mask = np.array(pil_mask).astype(np.float32) / 255.0
            instruction = "この画像の青い三角形をセグメントしてください。<SEG>"
        
        dummy_data.append((Image.fromarray(img), instruction, mask))
    
    return dummy_data

def test_improved_training():
    """改善された訓練のテスト"""
    print("=" * 80)
    print("🚀 改善された訓練テスト開始")
    print("=" * 80)
    
    # モデル設定
    lora_config = create_lora_manager(use_qlora=False)
    model_config = {
        'model_name': "Qwen/Qwen2.5-VL-3B-Instruct",
        'sam_model_name': "sam2_hiera_large.pt",
        'sam_config_name': "sam2_hiera_l.yaml",
        'torch_dtype': torch.float16,
        'device_map': "auto"
    }
    
    # モデル初期化
    print("\n🤖 モデル初期化中...")
    model = create_sam_qwen_model(model_config, lora_config)
    model.configure_for_training(freeze_sam_encoder=True)
    
    # 勾配蓄積の設定（メモリ効率化）
    gradient_accumulation_steps = 2
    
    # オプティマイザ（改善された設定）
    trainable_params = model.get_trainable_parameters()
    optimizer = optim.AdamW(
        trainable_params,
        lr=1e-5,  # さらに小さい学習率
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # 学習率スケジューラー（Warmup付き）
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    
    # ダミーデータ生成
    print("\n📦 ダミーデータ生成中...")
    dummy_data = create_dummy_data(num_samples=3)
    
    # 訓練ループ
    print("\n🎯 訓練開始...")
    device = model.device
    model.train()
    
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\n[エポック {epoch+1}/{num_epochs}]")
        epoch_loss = 0.0
        
        for i, (image, instruction, target_mask) in enumerate(dummy_data):
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
            
            try:
                # Forward pass
                results = model.forward_train(
                    images=image,
                    messages=messages,
                    target_masks=target_mask_tensor,
                    max_new_tokens=32
                )
                
                loss = results['total_loss']
                
                # NaN検出
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  ❌ サンプル {i+1} でNaN/Inf検出!")
                    # 詳細なデバッグ情報
                    print(f"     Loss: {loss.item()}")
                    if 'losses' in results:
                        for k, v in results['losses'].items():
                            if isinstance(v, torch.Tensor):
                                print(f"     {k}: {v.item()}")
                    
                    # パラメータの状態確認
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"     NaN/Inf in grad of {name}: norm={grad_norm}")
                    
                    # スキップ
                    optimizer.zero_grad()
                    continue
                
                # 勾配蓄積のためのスケーリング
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # 勾配蓄積
                if (i + 1) % gradient_accumulation_steps == 0:
                    # 勾配クリッピング
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    # オプティマイザステップ
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    print(f"  ✓ サンプル {i+1}: loss={loss.item() * gradient_accumulation_steps:.4f}, "
                          f"grad_norm={grad_norm:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
            except Exception as e:
                print(f"  ❌ サンプル {i+1} でエラー: {str(e)}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                continue
        
        avg_epoch_loss = epoch_loss / len(dummy_data)
        print(f"\nエポック平均損失: {avg_epoch_loss:.4f}")
    
    print("\n✅ 改善された訓練テスト完了！")

if __name__ == "__main__":
    test_improved_training()