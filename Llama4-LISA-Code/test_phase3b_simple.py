# test_phase3b_simple.py
"""
シンプルなテストスクリプト（マルチスケール無効）
"""

import torch
import torch.nn as nn
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_repo'))

# 設定
import config_linux

# Enhanced Model
from model.enhanced_llama4_qformer_sam2 import (
    EnhancedQFormerSegmentationBridge, 
    EnhancedLlamaQFormerSAM2Config
)

# Transformers (2025年正規実装準拠)
try:
    from transformers import Llama4ForConditionalGeneration, AutoProcessor
    LLAMA4_AVAILABLE = True
    print("✅ Llama4ForConditionalGeneration利用可能")
except ImportError:
    from transformers import AutoModelForCausalLM as Llama4ForConditionalGeneration, AutoProcessor
    LLAMA4_AVAILABLE = False
    print("⚠️ Llama4ForConditionalGeneration未対応、AutoModelForCausalLM使用")


def test_simple():
    """シンプルなテスト実行"""
    print("🚀 シンプルテスト開始...\n")
    
    # GPU確認
    if not torch.cuda.is_available():
        raise RuntimeError("GPU環境が必要です")
    
    print(f"✅ GPU利用可能: {torch.cuda.get_device_name()}")
    print(f"  - GPU数: {torch.cuda.device_count()}")
    
    # 1. Llama-4初期化 (2025年正規実装準拠)
    print("\n🧠 Llama-4初期化...")
    llama_model = Llama4ForConditionalGeneration.from_pretrained(
        config_linux.LLAMA_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=config_linux.ATTN_IMPLEMENTATION,  # config統一
        low_cpu_mem_usage=True
    )
    
    llama_processor = AutoProcessor.from_pretrained(
        config_linux.LLAMA_MODEL_ID,
        trust_remote_code=True
    )
    
    print("✅ Llama-4初期化完了 (2025年正規実装)")
    
    # 2. Enhanced Model初期化（マルチスケール無効）
    print("\n🚀 Enhanced Model初期化...")
    enhanced_config = EnhancedLlamaQFormerSAM2Config()
    
    model = EnhancedQFormerSegmentationBridge(
        config=enhanced_config,
        shared_llama_model=llama_model,
        shared_llama_processor=llama_processor,
        enable_lora=False,         # LoRA無効
        enable_multiscale=False,   # マルチスケール無効
        training_stage=1
    )
    
    print("✅ Enhanced Model初期化完了")
    
    # 3. テスト実行
    print("\n🧪 Forward処理テスト...")
    
    # テストデータ作成 (デュアルエンコーダー構成対応)
    batch_size = 1
    test_image = torch.randn(batch_size, 3, 448, 448, dtype=torch.bfloat16).cuda()      # Llama-4用 (正式仕様サイズ)
    test_sam_image = torch.randn(batch_size, 3, 1024, 1024, dtype=torch.bfloat16).cuda() # SAM2用
    test_text = ["This is a test image with objects"]
    test_labels = torch.zeros(batch_size, 1024, 1024, dtype=torch.long).cuda()  # SAM2解像度に合わせる
    
    print(f"  - Llama-4画像形状: {test_image.shape}")
    print(f"  - SAM2画像形状: {test_sam_image.shape}")
    print(f"  - テキスト: {test_text}")
    print(f"  - ラベル形状: {test_labels.shape}")
    
    # Forward実行
    try:
        with torch.no_grad():
            outputs = model(
                images=test_image,
                sam_images=test_sam_image,  # SAM2用画像を追加
                text_input=test_text,
                labels=test_labels,
                mode='itg'
            )
        
        print("\n✅ Forward成功！")
        print(f"  - マスク形状: {outputs['masks'].shape}")
        print(f"  - テキストロジット形状: {outputs['text_logits'].shape}")
        print(f"  - 視覚特徴形状: {outputs['visual_features'].shape}")
        
        if 'loss' in outputs:
            print(f"  - 損失値: {outputs['loss'].item():.4f}")
        
        # Q-Former出力確認
        if 'qformer_outputs' in outputs:
            qf_out = outputs['qformer_outputs']
            print(f"\n📊 Q-Former出力:")
            print(f"  - クエリ埋め込み: {qf_out['query_embeds'].shape}")
            print(f"  - LLM埋め込み: {qf_out['llm_embeds'].shape}")
            print(f"  - SAMプロンプト: {qf_out['sam_prompts'].shape}")
        
        print("\n🎉 テスト完了！")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # デバッグ情報
        print("\n🔍 デバッグ情報:")
        
        # SAM2エンコーダの出力を確認
        if hasattr(model.segmentation_head, 'sam_wrapper'):
            sam_wrapper = model.segmentation_head.sam_wrapper
            if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
                actual_model = sam_wrapper.predictor.model
                if hasattr(actual_model, 'image_encoder'):
                    image_encoder = actual_model.image_encoder
                    
                    with torch.no_grad():
                        encoded_output = image_encoder(test_image)
                    
                    if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
                        vision_features = encoded_output['vision_features']
                        print(f"  - SAM2出力形状: {vision_features.shape}")
                        print(f"  - SAM2出力dtype: {vision_features.dtype}")


if __name__ == "__main__":
    test_simple()