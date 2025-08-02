"""
NaN問題の詳細デバッグ - seg_hidden状態の確認
"""
import torch
import numpy as np
from PIL import Image
from model import create_sam_qwen_model, create_lora_manager

def check_tensor_stats(tensor, name):
    """テンソルの統計情報を表示"""
    if tensor is None:
        print(f"{name}: None")
        return
    
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  mean: {tensor.mean().item():.6f}")
    print(f"  std: {tensor.std().item():.6f}")
    print(f"  min: {tensor.min().item():.6f}")
    print(f"  max: {tensor.max().item():.6f}")
    print(f"  has_nan: {torch.isnan(tensor).any().item()}")
    print(f"  has_inf: {torch.isinf(tensor).any().item()}")
    
    # NaNの位置を特定
    if torch.isnan(tensor).any():
        nan_positions = torch.isnan(tensor).nonzero()
        print(f"  NaN positions: {nan_positions.tolist()[:5]}...")  # 最初の5個のみ

def create_dummy_image(img_size=224):
    """ダミー画像を作成"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    img[50:150, 50:150] = 255
    return Image.fromarray(img)

def main():
    print("="*80)
    print("seg_hidden状態の詳細デバッグ")
    print("="*80)
    
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
    model.train()
    
    # seg_to_sam_projectorの初期状態を確認
    print("\n📊 seg_to_sam_projectorの初期状態:")
    for i, layer in enumerate(model.seg_to_sam_projector):
        if hasattr(layer, 'weight'):
            check_tensor_stats(layer.weight, f"Layer {i} weight")
            if hasattr(layer, 'bias') and layer.bias is not None:
                check_tensor_stats(layer.bias, f"Layer {i} bias")
    
    # カスタムフックを追加
    class DebugHook:
        def __init__(self, name):
            self.name = name
            self.call_count = 0
        
        def __call__(self, module, input, output):
            self.call_count += 1
            print(f"\n[Hook {self.name}] Call #{self.call_count}")
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        check_tensor_stats(inp, f"Input {i}")
            if isinstance(output, torch.Tensor):
                check_tensor_stats(output, "Output")
    
    # フックを登録
    hooks = []
    
    # seg_hiddenが処理される部分にフックを追加
    if hasattr(model, '_seg_hidden_norm'):
        hook = model._seg_hidden_norm.register_forward_hook(DebugHook("seg_hidden_norm"))
        hooks.append(hook)
    
    # seg_to_sam_projectorの各層にフックを追加
    for i, layer in enumerate(model.seg_to_sam_projector):
        if isinstance(layer, torch.nn.Module):
            hook = layer.register_forward_hook(DebugHook(f"seg_to_sam_projector[{i}]"))
            hooks.append(hook)
    
    # ダミーデータでテスト
    image = create_dummy_image()
    instruction = "この画像の白い四角形をセグメントしてください。<SEG>"
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]
    }]
    
    try:
        print("\n🔄 Forward pass実行...")
        results = model.forward_with_segmentation(
            images=image,
            messages=messages,
            max_new_tokens=32
        )
        
        print("\n✅ Forward pass成功")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # フックを削除
        for hook in hooks:
            hook.remove()

if __name__ == "__main__":
    main()