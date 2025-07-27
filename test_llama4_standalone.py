#!/usr/bin/env python3
"""
Llama4-Scout-17B単独動作テスト（SAM統合準備版）
===========================================

Web研究で判明した実証済みコードを基に、確実に動作する
Llama4単独スクリプトを作成し、後のSAM統合に備える

参考: Medium記事「Llama-4-Scout: Hands-On with Meta's Cutting-Edge Multimodal AI」
https://medium.com/@agentic-ai/llama-4-scout-hands-on-with-metas-cutting-edge-multimodal-ai-cf5c2097b756
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import traceback
import gc
import psutil
from PIL import Image
import numpy as np

print("🚀 Llama4-Scout-17B Standalone Test (SAM Integration Ready)")

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_memory_optimizations():
    """メモリ最適化設定"""
    print("\n🧠 メモリ最適化設定を適用中...")
    
    # CUDA メモリ最適化環境変数
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    print("  ✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128")
    
    if torch.cuda.is_available():
        # TensorFloat-32の有効化（高速化）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  ✅ TensorFloat-32有効化")
        
        # CUDAメモリクリア
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        print("  ✅ 全GPUメモリクリア完了")

def check_gpu_environment():
    """GPU環境の確認"""
    print("\n💻 GPU環境の確認:")
    print(f"  PyTorch バージョン: {torch.__version__}")
    print(f"  CUDA 対応: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  利用可能GPU数: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}, メモリ: {memory_gb:.1f}GB")
        
        if gpu_count >= 8:
            print("  ✅ A100*8環境確認完了")
        else:
            print(f"  ⚠️  予期したGPU数(8)より少ない: {gpu_count}")
    else:
        print("  ❌ CUDA環境が利用できません")

def print_memory_usage(prefix=""):
    """メモリ使用量を表示"""
    if torch.cuda.is_available():
        print(f"\n{prefix}📊 メモリ使用量:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB 使用中, {reserved:.2f}GB 予約済み / {total:.2f}GB")

def load_llama4_model():
    """
    実証済み設定でLlama4モデルを読み込み
    Web研究で判明したベストプラクティスを適用
    """
    print("\n📥 Llama4-Scout-17B-16E-Instruct読み込み中...")
    print("  ⚙️  実証済み設定を使用:")
    print("    - attn_implementation: eager (flex attentionのバグ回避)")
    print("    - device_map: auto (A100×8自動分散)")
    print("    - torch_dtype: bfloat16 (メモリ効率化)")
    
    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    
    try:
        # プロセッサの読み込み（テキスト + 画像処理）
        print("  📝 AutoProcessor読み込み中...")
        processor = AutoProcessor.from_pretrained(model_id)
        print("  ✅ AutoProcessor読み込み完了")
        
        # モデルの読み込み（実証済み設定）
        print("  🤖 Llama4ForConditionalGeneration読み込み中...")
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="eager",     # Web研究: flex attentionにバグ
            device_map="auto",               # A100×8自動分散
            torch_dtype=torch.bfloat16,     # メモリ効率化
            trust_remote_code=True
        )
        print("  ✅ Llama4モデル読み込み完了")
        
        return model, processor
        
    except Exception as e:
        print(f"  ❌ モデル読み込みエラー: {e}")
        traceback.print_exc()
        raise

def test_text_generation(model, processor):
    """
    テキスト生成テスト
    Web研究で判明した公式chat templateを使用
    """
    print("\n" + "="*60)
    print("🔤 テキスト生成テスト")
    print("="*60)
    
    # シンプルな数学問題（Web研究の例を簡略化）
    test_prompt = """Please solve this calculus problem step by step:
    
Find the derivative of f(x) = x³ * sin(2x) + e^(x²)
    
Show all intermediate steps and explain your reasoning."""
    
    # 公式chat template形式（Web研究より）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": test_prompt},
            ]
        },
    ]
    
    try:
        print("🔄 入力処理中...")
        # apply_chat_templateを使用（実証済み方法）
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        print("✅ 入力処理完了")
        print(f"  - input_ids shape: {inputs['input_ids'].shape}")
        
        print("🔄 テキスト生成中...")
        # 生成実行
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # メモリ節約のため制限
            do_sample=True,
            temperature=0.1,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        print("✅ テキスト生成完了")
        
        # レスポンスのデコード
        response = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        print("\n📄 生成結果:")
        print("-" * 40)
        print(response[0])
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ テキスト生成エラー: {e}")
        traceback.print_exc()
        return False

def create_dummy_image():
    """SAM統合準備：ダミー画像作成"""
    print("\n🎨 SAM統合準備：ダミー画像作成中...")
    
    # 448x448の色付きパターン画像（Llama4推奨サイズ）
    image = Image.new('RGB', (448, 448), color='white')
    
    # NumPy配列で簡単なパターンを作成
    img_array = np.ones((448, 448, 3), dtype=np.uint8) * 255
    
    # カラフルなパターンを追加
    for i in range(448):
        for j in range(448):
            img_array[i, j, 0] = (i * 255) // 448  # 赤のグラデーション
            img_array[i, j, 1] = (j * 255) // 448  # 緑のグラデーション
            img_array[i, j, 2] = 128  # 青は固定
    
    image = Image.fromarray(img_array)
    
    print("✅ ダミー画像作成完了")
    print(f"  - サイズ: {image.size}")
    print(f"  - モード: {image.mode}")
    
    return image

def test_multimodal_generation(model, processor):
    """
    マルチモーダル生成テスト（SAM統合準備）
    """
    print("\n" + "="*60)
    print("🖼️  マルチモーダル生成テスト（SAM統合準備）")
    print("="*60)
    
    # ダミー画像作成
    test_image = create_dummy_image()
    
    # マルチモーダルメッセージ（Web研究の形式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # 画像プレースホルダー
                {"type": "text", "text": "Describe this image in detail. What colors, patterns, and shapes do you see? This is a test image that will help us prepare for SAM integration."},
            ]
        },
    ]
    
    try:
        print("🔄 マルチモーダル入力処理中...")
        # 画像 + テキストの処理（正しいLlama4方法）
        inputs = processor(
            images=[test_image],
            text=processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False  # textとして取得
            ),
            return_tensors="pt"
        ).to(model.device)
        
        print("✅ マルチモーダル入力処理完了")
        print(f"  - input_ids shape: {inputs['input_ids'].shape}")
        print(f"  - pixel_values shape: {inputs['pixel_values'].shape}")
        
        print("🔄 マルチモーダル生成中...")
        # 生成実行
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # メモリ節約
            do_sample=True,
            temperature=0.1,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        print("✅ マルチモーダル生成完了")
        
        # レスポンスのデコード
        response = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        print("\n🖼️  画像説明結果:")
        print("-" * 40)
        print(response[0])
        print("-" * 40)
        
        # SAM統合用の画像データを保存（後で使用）
        return {
            "success": True,
            "pixel_values": inputs['pixel_values'],  # SAM統合で活用
            "image_size": test_image.size,
            "description": response[0]
        }
        
    except Exception as e:
        print(f"❌ マルチモーダル生成エラー: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def analyze_model_for_sam_integration(model, processor):
    """SAM統合のためのモデル分析"""
    print("\n" + "="*60)
    print("🔧 SAM統合準備：モデル分析")
    print("="*60)
    
    # パラメータ統計
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Llama4モデル統計:")
    print(f"  - 総パラメータ数: {total_params:,}")
    print(f"  - 学習可能パラメータ数: {trainable_params:,}")
    print(f"  - 学習可能率: {(trainable_params/total_params)*100:.2f}%")
    
    # ビジョン関連コンポーネントの確認
    print(f"\n🔍 ビジョンコンポーネント分析:")
    vision_components = []
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['vision', 'image', 'visual', 'pixel']):
            vision_components.append(name)
            print(f"  - {name}: {type(module).__name__}")
    
    print(f"\n✅ ビジョンコンポーネント数: {len(vision_components)}")
    
    # プロセッサ情報
    print(f"\n📝 Processor情報:")
    print(f"  - Processor type: {type(processor).__name__}")
    if hasattr(processor, 'image_processor'):
        print(f"  - Image processor: {type(processor.image_processor).__name__}")
    if hasattr(processor, 'tokenizer'):
        print(f"  - Tokenizer: {type(processor.tokenizer).__name__}")
        print(f"  - Vocabulary size: {len(processor.tokenizer)}")
    
    # SAM統合のための推奨事項
    print(f"\n💡 SAM統合推奨事項:")
    print(f"  1. pixel_valuesを直接SAMに渡すことが可能")
    print(f"  2. Llama4の隠れ状態（5120次元）をSAMプロンプト（256次元）に射影")
    print(f"  3. [SEG]トークンをtokenizerに追加")
    print(f"  4. SAMマスクデコーダーのみ学習可能に設定")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "vision_components": vision_components,
        "processor_info": {
            "type": type(processor).__name__,
            "has_image_processor": hasattr(processor, 'image_processor'),
            "vocab_size": len(processor.tokenizer) if hasattr(processor, 'tokenizer') else 0
        }
    }

def main():
    try:
        print("🚀 Llama4-Scout-17B Standalone Test (SAM Integration Ready)")
        print("="*80)
        print("実証済みコードによるLlama4単独動作確認")
        print("SAM統合準備を念頭に置いた設計")
        print("="*80)
        
        # メモリ最適化設定
        setup_memory_optimizations()
        
        # GPU環境確認
        check_gpu_environment()
        
        print_memory_usage("初期")
        
        # ===== Step 1: Llama4モデル読み込み =====
        print("\n" + "="*60)
        print("Step 1: Llama4モデル読み込み（実証済み設定）")
        print("="*60)
        
        model, processor = load_llama4_model()
        
        print_memory_usage("モデル読み込み後")
        
        # ===== Step 2: テキスト生成テスト =====
        print("\n" + "="*60)
        print("Step 2: テキスト生成テスト")
        print("="*60)
        
        text_success = test_text_generation(model, processor)
        
        print_memory_usage("テキスト生成後")
        
        # ===== Step 3: マルチモーダル生成テスト =====
        print("\n" + "="*60)
        print("Step 3: マルチモーダル生成テスト")
        print("="*60)
        
        multimodal_result = test_multimodal_generation(model, processor)
        
        print_memory_usage("マルチモーダル生成後")
        
        # ===== Step 4: SAM統合準備分析 =====
        print("\n" + "="*60)
        print("Step 4: SAM統合準備分析")
        print("="*60)
        
        analysis_result = analyze_model_for_sam_integration(model, processor)
        
        # ===== Step 5: 結果サマリー =====
        print("\n" + "="*60)
        print("Step 5: テスト結果サマリー")
        print("="*60)
        
        # 成功条件チェック
        success_criteria = {
            "model_loading": True,
            "text_generation": text_success,
            "multimodal_generation": multimodal_result["success"] if isinstance(multimodal_result, dict) else False,
            "sam_integration_ready": len(analysis_result["vision_components"]) > 0
        }
        
        print("🎯 テスト結果:")
        for criterion, success in success_criteria.items():
            status = "✅" if success else "❌"
            print(f"  {status} {criterion}")
        
        overall_success = all(success_criteria.values())
        
        if overall_success:
            print("\n🎉 全テストが成功しました！")
            print("   Llama4単独動作が確認できました。")
            print("   SAM統合の準備が整いました。")
        else:
            print("\n⚠️  一部のテストで問題が検出されました。")
            print("   問題を解決してからSAM統合に進んでください。")
        
        # 結果をJSONで保存（Tensorオブジェクトを除外）
        def convert_for_json(obj):
            """Tensorオブジェクトを文字列に変換してJSON保存可能にする"""
            if torch.is_tensor(obj):
                return f"Tensor{list(obj.shape)}"
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "success_criteria": success_criteria,
            "overall_success": overall_success,
            "model_analysis": analysis_result,
            "multimodal_test": convert_for_json(multimodal_result)
        }
        
        output_file = f"llama4_standalone_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 テスト結果を保存: {output_file}")
        
        print_memory_usage("最終")
        
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        traceback.print_exc()
    
    finally:
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 メモリクリーンアップ完了")

if __name__ == "__main__":
    main() 