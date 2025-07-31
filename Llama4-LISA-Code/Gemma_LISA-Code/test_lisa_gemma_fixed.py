#!/usr/bin/env python3
"""
LISA-Gemma Fixed の動作テスト
Gemma-3公式API準拠の新しいアーキテクチャ
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa_fixed import LisaGemmaFixedConfig, LisaGemmaFixed
import config_linux

def test_lisa_gemma_fixed():
    """修正されたLISA-Gemmaモデルのテスト"""
    print("=== LISA-Gemma Fixed テスト ===")
    
    try:
        # 設定を作成（SAMチェックポイントなしでテスト）
        config = LisaGemmaFixedConfig(
            gemma_model_id=config_linux.GEMMA_MODEL_ID,
            sam_checkpoint_path="",  # SAMなしでテスト
            seg_token="<SEG>",
            gemma_hidden_size=2560,
            sam_prompt_embed_dim=256,
        )
        
        print("LISA-Gemmaモデルを初期化中...")
        model = LisaGemmaFixed(config)
        
        print("✅ モデル初期化成功")
        
        # テスト用の画像を作成
        test_image = Image.new('RGB', (400, 400), color='red')
        
        # テスト1: 基本的なテキスト生成
        print("\n--- テスト1: 基本的なテキスト生成 ---")
        test_prompt = "この画像について説明してください。"
        
        result = model.generate_with_segmentation(
            image=test_image,
            text_prompt=test_prompt,
            max_new_tokens=50
        )
        
        print(f"生成されたテキスト: {result['generated_text']}")
        print(f"マスク生成: {'✅ あり' if result['predicted_masks'] is not None else '❌ なし'}")
        
        # テスト2: セグメンテーションを要求するプロンプト
        print("\n--- テスト2: セグメンテーション要求 ---")
        seg_prompt = "赤い部分をセグメンテーションしてください。<SEG>"
        
        result_seg = model.generate_with_segmentation(
            image=test_image,
            text_prompt=seg_prompt,
            max_new_tokens=50
        )
        
        print(f"生成されたテキスト: {result_seg['generated_text']}")
        print(f"マスク生成: {'✅ あり' if result_seg['predicted_masks'] is not None else '❌ なし'}")
        
        # テスト3: フォワードパスの直接テスト
        print("\n--- テスト3: フォワードパス直接テスト ---")
        forward_result = model.forward(
            image=test_image,
            text_prompt="この画像の内容を教えてください",
            generate_mask=False  # SAMなしなのでマスク生成はしない
        )
        
        if "error" in forward_result:
            print(f"❌ フォワードパスエラー: {forward_result['error']}")
        else:
            print("✅ フォワードパス成功")
            print(f"出力形状: {forward_result['gemma_logits'].shape}")
        
        print("\n=== すべてのテスト完了 ===")
        return True
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer_seg_token():
    """SEGトークンの追加をテスト"""
    print("\n=== SEGトークン追加テスト ===")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        
        print(f"元の語彙サイズ: {len(tokenizer)}")
        
        # SEGトークンを追加
        tokenizer.add_tokens(["<SEG>"], special_tokens=True)
        
        print(f"追加後の語彙サイズ: {len(tokenizer)}")
        
        # トークン化テスト
        test_text = "この画像の赤い部分を<SEG>してください"
        tokens = tokenizer.tokenize(test_text)
        print(f"トークン化結果: {tokens}")
        
        seg_token_id = tokenizer.convert_tokens_to_ids("<SEG>")
        print(f"<SEG>トークンID: {seg_token_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ SEGトークンテストエラー: {e}")
        return False

if __name__ == "__main__":
    print("LISA-Gemma Fixed テスト開始\n")
    
    # まずSEGトークンのテスト
    token_success = test_tokenizer_seg_token()
    
    # 次にモデル全体のテスト
    model_success = test_lisa_gemma_fixed()
    
    print(f"\n=== 最終結果 ===")
    print(f"SEGトークン: {'✅ 成功' if token_success else '❌ 失敗'}")
    print(f"モデルテスト: {'✅ 成功' if model_success else '❌ 失敗'}")
    
    if token_success and model_success:
        print("🎉 全テスト成功！修正されたLISA-Gemmaアーキテクチャが動作しています！")
    else:
        print("⚠️ 一部のテストが失敗しました。") 