#!/usr/bin/env python3
"""
Gemma-3の正しい使用方法のテスト
公式ドキュメントに基づく実装
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gemma3_official_way():
    """Gemma-3の公式な使用方法でテスト"""
    print("=== Gemma-3 公式使用方法テスト ===")
    
    try:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        
        # モデルID (仕様書通り)
        model_id = "google/gemma-3-4b-it"
        
        print(f"モデルロード中: {model_id}")
        
        # 公式推奨の方法でモデルとプロセッサーをロード
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        print("✅ モデルとプロセッサーが正常にロードされました")
        
        # ダミー画像を作成
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # 公式ドキュメント通りのメッセージ形式
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},  # 画像をここで指定
                    {"type": "text", "text": "What do you see in this image?"}
                ]
            }
        ]
        
        print("チャットテンプレート適用中...")
        
        # プロセッサーのチャットテンプレート機能を使用
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        print(f"入力テンソル形状: {inputs['input_ids'].shape}")
        if 'pixel_values' in inputs:
            print(f"画像テンソル形状: {inputs['pixel_values'].shape}")
        
        print("推論実行中...")
        
        # 推論実行
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        print("✅ 推論が正常に完了しました")
        
        # 結果をデコード
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = generation[0][input_len:]
        decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"生成されたテキスト: {decoded_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_only_mode():
    """テキストのみでのGemma-3テスト"""
    print("\n=== テキストのみモードテスト ===")
    
    try:
        from transformers import AutoTokenizer, Gemma3ForCausalLM
        
        model_id = "google/gemma-3-4b-it"
        
        print("テキストのみモデルをロード中...")
        
        # テキストのみの場合のクラスを使用
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # シンプルなテキスト入力
        prompt = "What is the meaning of life?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("テキスト推論実行中...")
        
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ テキスト生成結果: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ テキストモードエラー: {e}")
        return False

if __name__ == "__main__":
    print("Gemma-3 正しい使用方法のテスト開始\n")
    
    # まずテキストのみをテスト
    text_success = test_text_only_mode()
    
    # 次にマルチモーダルをテスト
    multimodal_success = test_gemma3_official_way()
    
    print(f"\n=== テスト結果 ===")
    print(f"テキストのみ: {'✅ 成功' if text_success else '❌ 失敗'}")
    print(f"マルチモーダル: {'✅ 成功' if multimodal_success else '❌ 失敗'}")
    
    if text_success and multimodal_success:
        print("🎉 すべてのテストが成功しました！")
    else:
        print("⚠️  一部のテストが失敗しました。") 