#!/usr/bin/env python3
"""
LISA-Llama4 Inference Pipeline テスト

【重要】Training時とInference時のAPI使用法の違い:
1. Training時: 直接tensor -> model(input_ids=..., pixel_values=..., labels=...)
2. Inference時: apply_chat_template -> model.generate()

このスクリプトは正しいInference APIの使用法を示します。
"""

import torch
import os
from PIL import Image
import numpy as np
from pathlib import Path

# プロジェクトのパスを追加
import sys
sys.path.append('/Users/odasoya/LISA-Gemma-Linux_claude_code')

from model.llama4_lisa import LisaLlama4ForCausalLM
from config_linux import load_config

def test_llama4_inference():
    """
    Llama4のInference Pipeline テスト
    """
    print("\n" + "="*60)
    print("🚀 LISA-Llama4 Inference Pipeline テスト開始")
    print("="*60)
    
    # 設定読み込み
    print("\n📝 Step 1: 設定読み込み")
    config = load_config()
    print(f"✅ 設定読み込み完了")
    print(f"   - Model: {config.llama_model_id}")
    print(f"   - Hidden Size: {config.llama_hidden_size}")
    print(f"   - SEG Token: {config.seg_token}")
    
    # デバイス確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - Device: {device}")
    
    # モデル初期化
    print("\n📝 Step 2: モデル初期化")
    try:
        model = LisaLlama4ForCausalLM.from_config_file('config_linux.py')
        model = model.to(device)
        model.eval()  # Inference モード
        print("✅ モデル初期化成功")
        
        # パラメータ情報表示
        param_info = model.get_trainable_parameters_info()
        total_params = param_info["total_parameters"]
        trainable_params = param_info["trainable_parameters"]
        print(f"   - 総パラメータ: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   - 学習可能: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        
    except Exception as e:
        print(f"❌ モデル初期化エラー: {e}")
        return False
    
    # テスト画像準備
    print("\n📝 Step 3: テスト画像準備")
    try:
        # ダミー画像作成（Lambda Cloudには画像がない可能性）
        test_image = Image.new('RGB', (512, 512), color='red')
        print("✅ テスト画像準備完了")
    except Exception as e:
        print(f"❌ 画像準備エラー: {e}")
        return False
    
    # テストケース実行
    print("\n📝 Step 4: Inference テスト実行")
    test_cases = [
        {
            "name": "基本テスト（SEGなし）",
            "prompt": "この画像には何が写っていますか？",
            "expect_seg": False
        },
        {
            "name": "セグメンテーションテスト（SEG要求）", 
            "prompt": f"この画像の赤い部分を{config.seg_token}してください。",
            "expect_seg": True
        },
        {
            "name": "複雑な指示テスト",
            "prompt": f"画像を分析して、主要なオブジェクトを{config.seg_token}で示してください。",
            "expect_seg": True
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎯 テスト {i}: {test_case['name']}")
        print(f"   プロンプト: {test_case['prompt']}")
        
        try:
            # Inference実行
            with torch.no_grad():  # メモリ効率化
                result = model.generate_with_segmentation(
                    image=test_image,
                    text_prompt=test_case['prompt'],
                    max_new_tokens=100
                )
            
            # 結果確認
            if "error" in result:
                print(f"   ❌ エラー: {result['error']}")
                results.append({"test": test_case['name'], "status": "error", "error": result['error']})
                continue
            
            generated_text = result.get('generated_text', '')
            predicted_masks = result.get('predicted_masks')
            
            # SEGトークン検出確認
            has_seg = config.seg_token in generated_text
            mask_generated = predicted_masks is not None
            
            print(f"   ✅ 生成テキスト: {generated_text[:150]}...")
            print(f"   📊 SEGトークン検出: {has_seg}")
            print(f"   📊 マスク生成: {mask_generated}")
            
            # 期待結果との比較
            if test_case['expect_seg']:
                if has_seg and mask_generated:
                    print(f"   🎉 期待通り: SEGトークン検出＆マスク生成")
                    status = "success"
                elif has_seg and not mask_generated:
                    print(f"   ⚠️  部分成功: SEGトークン検出、マスク生成失敗")
                    status = "partial"
        else:
                    print(f"   ❌ 期待外: SEGトークン未検出")
                    status = "unexpected"
            else:
                if not has_seg:
                    print(f"   🎉 期待通り: SEGトークンなし")
                    status = "success"
        else:
                    print(f"   ⚠️  期待外: SEGトークン検出")
                    status = "unexpected"
            
            results.append({
                "test": test_case['name'],
                "status": status,
                "generated_text": generated_text,
                "has_seg": has_seg,
                "mask_generated": mask_generated
            })
            
    except Exception as e:
            print(f"   ❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
            results.append({"test": test_case['name'], "status": "error", "error": str(e)})
    
    # 結果サマリー
    print("\n" + "="*60)
    print("📊 テスト結果サマリー")
    print("="*60)
    
    success_count = sum(1 for r in results if r.get('status') == 'success')
    total_count = len(results)
    
    print(f"✅ 成功: {success_count}/{total_count}")
    
    for result in results:
        status_icon = {
            'success': '✅',
            'partial': '⚠️',
            'unexpected': '❓', 
            'error': '❌'
        }.get(result.get('status'), '❓')
        
        print(f"{status_icon} {result['test']}: {result.get('status', 'unknown')}")
        if 'error' in result:
            print(f"    エラー: {result['error']}")
    
    # Training/Inference API違いの説明
    print("\n" + "="*60)
    print("📚 Training vs Inference API 使用法の違い")
    print("="*60)
    print("""
🔹 Training時:
   - 直接tensor入力: model(input_ids=..., pixel_values=..., labels=...)
   - apply_chat_templateは使用しない
   - 損失計算が主目的

🔹 Inference時:
   - apply_chat_template使用: processor.apply_chat_template(messages, ...)
   - model.generate()でテキスト生成
   - SEGトークン検出でマスク生成

⚠️  重要: Training時にapply_chat_templateを使うとエラーの原因になる！
""")
    
    return success_count == total_count

if __name__ == "__main__":
    success = test_llama4_inference()
    
    print(f"\n🏁 テスト完了: {'SUCCESS' if success else 'FAILURE'}")
    
    if success:
        print("🎉 すべてのInferenceテストが成功しました！")
    else:
        print("⚠️  一部のテストで問題が発生しました。ログを確認してください。")
