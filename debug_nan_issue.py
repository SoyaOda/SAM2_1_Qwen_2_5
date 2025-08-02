"""
NaN問題のデバッグスクリプト
"""
import torch
import numpy as np
from PIL import Image
import os
from model import create_sam_qwen_model, create_lora_manager

def create_dummy_data(num_samples=1, img_size=224):
    """ダミーデータセットを作成"""
    dummy_data = []
    
    for i in range(num_samples):
        # 簡単な画像を生成（白い背景に図形）
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

def debug_forward_pass(model, image, instruction, target_mask, sample_idx):
    """Forward passをデバッグ"""
    print(f"\n{'='*60}")
    print(f"サンプル {sample_idx} のデバッグ")
    print(f"{'='*60}")
    
    # メッセージ形式に変換
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]
    }]
    
    # ターゲットマスクをテンソルに変換
    device = model.device
    target_mask_tensor = torch.from_numpy(target_mask).float().to(device)
    
    # Forward pass
    try:
        # 各層の出力を監視するためのフック
        activation_stats = {}
        gradient_stats = {}
        
        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item()
                    }
                    if activation_stats[name]['has_nan'] or activation_stats[name]['has_inf']:
                        print(f"⚠️ {name}でNaN/Inf検出!")
                        print(f"   統計: {activation_stats[name]}")
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad = grad_output[0]
                    gradient_stats[name] = {
                        'mean': grad.mean().item() if not torch.isnan(grad).all() else float('nan'),
                        'std': grad.std().item() if not torch.isnan(grad).all() else float('nan'),
                        'min': grad.min().item() if not torch.isnan(grad).all() else float('nan'),
                        'max': grad.max().item() if not torch.isnan(grad).all() else float('nan'),
                        'has_nan': torch.isnan(grad).any().item(),
                        'has_inf': torch.isinf(grad).any().item()
                    }
                    if gradient_stats[name]['has_nan'] or gradient_stats[name]['has_inf']:
                        print(f"⚠️ {name}の勾配でNaN/Inf検出!")
                        print(f"   統計: {gradient_stats[name]}")
            return hook
        
        # フックを登録
        handles = []
        if hasattr(model, 'vision_to_llm_projector'):
            for i, layer in enumerate(model.vision_to_llm_projector):
                if isinstance(layer, torch.nn.Module):
                    h1 = layer.register_forward_hook(forward_hook(f'vision_to_llm_projector.{i}'))
                    h2 = layer.register_backward_hook(backward_hook(f'vision_to_llm_projector.{i}'))
                    handles.extend([h1, h2])
        
        if hasattr(model, 'seg_to_sam_projector'):
            for i, layer in enumerate(model.seg_to_sam_projector):
                if isinstance(layer, torch.nn.Module):
                    h1 = layer.register_forward_hook(forward_hook(f'seg_to_sam_projector.{i}'))
                    h2 = layer.register_backward_hook(backward_hook(f'seg_to_sam_projector.{i}'))
                    handles.extend([h1, h2])
        
        # Forward pass
        results = model.forward_train(
            images=image,
            messages=messages,
            target_masks=target_mask_tensor,
            max_new_tokens=32
        )
        
        # 損失情報
        loss = results['total_loss']
        print(f"\n損失値: {loss.item():.6f}")
        print(f"NaN: {torch.isnan(loss).item()}, Inf: {torch.isinf(loss).item()}")
        
        # 活性化の統計を表示
        print("\n活性化統計:")
        for name, stats in activation_stats.items():
            print(f"  {name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                  f"min={stats['min']:.6f}, max={stats['max']:.6f}")
        
        # Backward pass
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            
            # 勾配統計を表示
            print("\n勾配統計:")
            for name, stats in gradient_stats.items():
                if name in gradient_stats:
                    print(f"  {name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                          f"min={stats['min']:.6f}, max={stats['max']:.6f}")
        
        # フックを削除
        for h in handles:
            h.remove()
        
        return True
        
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("NaN問題デバッグ")
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
    model.configure_for_training()
    model.train()
    
    # ダミーデータ生成
    print("\n📦 ダミーデータ生成中...")
    dummy_data = create_dummy_data(num_samples=2)
    
    # 各サンプルでテスト
    for i, (image, instruction, target_mask) in enumerate(dummy_data):
        success = debug_forward_pass(model, image, instruction, target_mask, i+1)
        if not success:
            print(f"\nサンプル {i+1} で失敗")
            break
        
        # オプティマイザステップをシミュレート
        optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-5)
        optimizer.step()
        optimizer.zero_grad()
        print(f"\n✅ サンプル {i+1} 完了")

if __name__ == "__main__":
    main()