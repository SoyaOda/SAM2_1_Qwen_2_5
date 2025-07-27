#!/usr/bin/env python3
"""
マルチモーダル入力準備のデバッグテスト
"""

import sys
import torch
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.insert(0, '.')

from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config

def test_multimodal_input():
    """マルチモーダル入力準備のテスト"""
    print("=== マルチモーダル入力デバッグテスト ===")
    
    # 設定
    config = LisaLlama4Config(
        llama_model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        sam_checkpoint_path=None,
        attn_implementation="eager",
        torch_dtype="bfloat16"
    )
    
    print("モデル初期化中...")
    torch.compiler.disable()
    model = LisaLlama4ForCausalLM(config)
    
    print("マルチモーダル入力テスト実行...")
    test_image = Image.new('RGB', (224, 224), color='red')
    test_prompt = "この画像を説明してください。"
    
    try:
        inputs = model.prepare_multimodal_input(test_image, test_prompt, for_training=False)
        print(f"✅ 成功: type={type(inputs)}")
        if isinstance(inputs, dict):
            print(f"Keys: {list(inputs.keys())}")
            for k, v in inputs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
        else:
            print(f"❌ 辞書ではない: {inputs}")
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multimodal_input()
    sys.exit(0 if success else 1) 