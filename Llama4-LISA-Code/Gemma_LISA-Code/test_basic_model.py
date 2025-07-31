#!/usr/bin/env python3
"""
基本的なモデル動作確認テスト
仕様書第6章「段階的検証ガイド」に従った検証を実行（Gemma-3対応版）
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaConfig, LisaGemmaForCausalLM
import config_linux

def test_model_initialization():
    """Step 1: モデルの初期化テスト"""
    print("=== Step 1: モデル初期化テスト ===")
    
    try:
        # SAMチェックポイントなしでテスト（まだダウンロードしていないため）
        config = LisaGemmaConfig(
            gemma_model_id=config_linux.GEMMA_MODEL_ID,
            sam_checkpoint_path="",  # 空文字でダミーを使用
            seg_token="<SEG>",  # セグメンテーショントークン
            gemma_hidden_size=2560,  # Gemma 3 4B
            sam_prompt_embed_dim=256,
        )
        
        print(f"設定: Gemmaモデル = {config.gemma_model_id}")
        print(f"設定: Hidden size = {config.gemma_hidden_size}")
        print(f"設定: SAM projection dim = {config.sam_prompt_embed_dim}")
        
        print("モデルを初期化中...")
        model = LisaGemmaForCausalLM(config)
        print("✓ モデル初期化成功！")
        
        return model, config
        
    except Exception as e:
        print(f"✗ モデル初期化失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_forward_pass_text_only(model):
    """Step 2: テキストのみのフォワードパステスト"""
    print("\n=== Step 2: テキストのみフォワードパステスト ===")
    
    try:
        # ダミー画像とテキストプロンプト
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        text_prompt = "What is in this image?"
        
        print(f"入力: {text_prompt}")
        
        # 新しいAPIでフォワードパス
        print("フォワードパス実行中...")
        outputs = model.forward(
            image=dummy_image,
            text_prompt=text_prompt,
            generate_mask=False,  # SAMコンポーネントがないのでマスク生成なし
        )
        
        print(f"✓ フォワードパス成功!")
        if "gemma_logits" in outputs:
            print(f"  - logits形状: {outputs['gemma_logits'].shape}")
        if "hidden_states" in outputs:
            print(f"  - 隠れ状態層数: {len(outputs['hidden_states'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ フォワードパス失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generate_with_segmentation(model):
    """Step 3: テキスト生成テスト"""
    print("\n=== Step 3: テキスト生成テスト ===")
    
    try:
        # ダミー画像とテキストプロンプト
        dummy_image = Image.new('RGB', (224, 224), color='green')
        text_prompt = "Describe this image in detail."
        
        print(f"入力プロンプト: {text_prompt}")
        
        # テキスト生成実行
        print("テキスト生成実行中...")
        results = model.generate_with_segmentation(
            image=dummy_image,
            text_prompt=text_prompt,
            max_new_tokens=50
        )
        
        print(f"✓ テキスト生成成功!")
        if "generated_text" in results:
            print(f"  - 生成されたテキスト: {results['generated_text']}")
        if "predicted_masks" in results:
            print(f"  - マスク予測: {results['predicted_masks']}")
        
        return True
        
    except Exception as e:
        print(f"✗ テキスト生成失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainable_parameters(model):
    """Step 4: 訓練可能パラメータの確認"""
    print("\n=== Step 4: 訓練可能パラメータ確認 ===")
    
    try:
        param_info = model.get_trainable_parameters_info()
        
        print(f"総パラメータ数: {param_info['total_parameters']:,}")
        print(f"訓練可能パラメータ数: {param_info['trainable_parameters']:,}")
        print(f"訓練可能な割合: {param_info['trainable_percentage']:.2f}%")
        
        # 詳細な分析
        trainable_components = []
        frozen_components = []
        
        for name, param in model.named_parameters():
            component = name.split('.')[0]  # 最初の部分を取得
            if param.requires_grad:
                if component not in trainable_components:
                    trainable_components.append(component)
            else:
                if component not in frozen_components:
                    frozen_components.append(component)
        
        print(f"訓練可能コンポーネント: {trainable_components}")
        print(f"凍結コンポーネント: {frozen_components}")
        
        # 期待する設定の確認
        if param_info['trainable_parameters'] > 0:
            print("✓ 訓練可能パラメータが正しく設定されています")
            return True
        else:
            print("⚠️ 訓練可能パラメータが設定されていません")
            return False
            
    except Exception as e:
        print(f"✗ パラメータ確認失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("LISA-Gemma基本動作確認テスト開始")
    print("=" * 50)
    
    # Step 1: モデル初期化
    model, config = test_model_initialization()
    if model is None:
        print("モデル初期化に失敗したため、テストを終了します。")
        return False
    
    # Step 2: テキストのみフォワードパス
    if not test_forward_pass_text_only(model):
        print("テキストのみフォワードパスに失敗しました。")
        return False
    
    # Step 3: テキスト生成テスト
    if not test_generate_with_segmentation(model):
        print("テキスト生成に失敗しました。")
        return False
    
    # Step 4: 訓練可能パラメータ確認
    if not test_trainable_parameters(model):
        print("訓練可能パラメータの設定に問題があります。")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 全ての基本テストが成功しました！")
    print("LISA-Gemmaモデルが正しく動作しています。")
    print("次のステップ: データセットパイプラインの実装に進むことができます。")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 