"""
Option Aモデルの基本動作テスト
Qwen ViT主体アーキテクチャの動作確認
"""
import torch
import numpy as np
from PIL import Image
import os
from model import create_sam_qwen_model_option_a, create_lora_manager

def create_dummy_image(size=224):
    """ダミー画像を作成"""
    # 白い四角形を含む画像
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[50:150, 50:150] = 255  # 白い四角形
    return Image.fromarray(img)

def test_option_a_model():
    """Option Aモデルの基本テスト"""
    print("=" * 80)
    print("Option Aモデル（Qwen ViT主体）基本動作テスト")
    print("=" * 80)
    
    # モデル設定
    model_config = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "sam_checkpoint": "sam2_hiera_large.pt",
        "sam_config": "sam2_hiera_l.yaml",
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    
    # LoRA設定（オプション）
    # lora_config = create_lora_manager(use_qlora=False)
    # model_config["lora_config"] = lora_config
    # 一旦LoRAなしでテスト
    model_config["lora_config"] = None
    
    try:
        # モデル作成
        print("\n🚀 Option Aモデル初期化中...")
        model = create_sam_qwen_model_option_a(model_config)
        model.eval()
        
        # デバイス確認
        device = model.device
        print(f"\n📍 デバイス: {device}")
        
        # テスト画像準備
        print("\n🖼️ テスト画像準備中...")
        test_image = create_dummy_image(224)
        
        # メッセージ準備（<SEG>トークンを含む）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},  # PILイメージを直接指定
                    {"type": "text", "text": "この画像の白い四角形をセグメントしてください。<SEG>"}
                ]
            }
        ]
        
        # 推論実行
        print("\n🔮 推論実行中...")
        with torch.no_grad():
            results = model.forward_with_segmentation(
                images=[test_image],
                messages=messages,
                max_new_tokens=128
            )
        
        # 結果表示
        print("\n📊 推論結果:")
        print(f"  - 生成テキスト: {results['generated_text']}")
        print(f"  - <SEG>トークン検出: {results['has_seg_token']}")
        
        if results['mask'] is not None:
            mask_shape = results['mask'].shape
            mask_sum = results['mask'].sum().item()
            print(f"  - マスク形状: {mask_shape}")
            print(f"  - マスク非ゼロ要素数: {mask_sum}")
        else:
            print("  - マスク: なし")
        
        # アーキテクチャ確認
        print("\n🏗️ アーキテクチャ確認:")
        print(f"  - Qwen Vision Hidden Size: {model.vision_hidden_size}")
        print(f"  - SAM Embed Dim: {model.sam_embed_dim}")
        print(f"  - Vision to SAM Projector: {model.vision_hidden_size} → {model.sam_embed_dim}")
        
        # メモリ使用量
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n💾 GPU メモリ使用量:")
            print(f"  - 確保済み: {allocated:.2f} GB")
            print(f"  - 予約済み: {reserved:.2f} GB")
        
        print("\n✅ Option Aモデル基本動作テスト完了！")
        return True
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # CUDA使用可能性チェック
    if torch.cuda.is_available():
        print(f"🎮 CUDA利用可能: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA利用不可 - CPUモードで実行")
    
    # テスト実行
    success = test_option_a_model()
    exit(0 if success else 1)