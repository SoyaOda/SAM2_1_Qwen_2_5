# debug_feature_flow.py
"""
Enhanced modelの特徴フロー全体をデバッグ
特に14次元問題の原因を特定
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('.')

from model.enhanced_llama4_qformer_sam2 import EnhancedQFormerSegmentationBridge, EnhancedLlamaQFormerSAM2Config
from transformers import AutoModelForCausalLM, AutoProcessor
import config_linux

def debug_feature_flow():
    """特徴の流れを詳細にトレース"""
    print("🔍 特徴フローデバッグ開始...\n")
    
    # 最小限のLlama初期化（必須）
    print("🧠 Llama-4初期化...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        config_linux.LLAMA_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    llama_processor = AutoProcessor.from_pretrained(
        config_linux.LLAMA_MODEL_ID,
        trust_remote_code=True
    )
    
    # Enhanced model設定と初期化
    enhanced_config = EnhancedLlamaQFormerSAM2Config()
    
    # テスト用に設定を確認
    print(f"\n📋 Q-Former設定:")
    print(f"  - encoder_hidden_size: {enhanced_config.qformer_config['encoder_hidden_size']}")
    
    # モデル初期化
    print("\n🚀 Enhanced Model初期化...")
    model = EnhancedQFormerSegmentationBridge(
        config=enhanced_config,
        shared_llama_model=llama_model,
        shared_llama_processor=llama_processor,
        enable_lora=False,  # LoRAは無効化してシンプルに
        enable_multiscale=False,  # マルチスケールも無効化
        training_stage=1
    )
    
    # テスト画像
    batch_size = 1
    test_image = torch.randn(batch_size, 3, 448, 448).cuda()
    print(f"\n📊 テスト画像形状: {test_image.shape}")
    
    # 画像エンコーダの出力を直接確認
    print("\n🔬 画像エンコーダの直接実行...")
    
    # SAM2エンコーダを取得
    if hasattr(model.segmentation_head, 'sam_wrapper'):
        sam_wrapper = model.segmentation_head.sam_wrapper
        if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
            actual_model = sam_wrapper.predictor.model
            if hasattr(actual_model, 'image_encoder'):
                image_encoder = actual_model.image_encoder
                
                # エンコーダ実行
                with torch.no_grad():
                    encoded_output = image_encoder(test_image)
                
                print(f"✅ エンコーダ実行成功")
                
                if isinstance(encoded_output, dict):
                    for key, value in encoded_output.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  - {key}: {value.shape}")
                    
                    if 'vision_features' in encoded_output:
                        vision_features = encoded_output['vision_features']
                        print(f"\n📐 vision_features詳細:")
                        print(f"  - 形状: {vision_features.shape}")
                        print(f"  - dtype: {vision_features.dtype}")
                        
                        # Enhanced modelのforward処理の一部を模擬
                        print(f"\n🔄 Enhanced modelのforward処理シミュレーション...")
                        
                        # _reshape_image_featuresの処理を確認
                        from model.enhanced_qformer import EnhancedQFormerModel
                        qformer = model.qformer
                        
                        # 直接_reshape_image_featuresを呼び出す
                        if hasattr(qformer, '_reshape_image_features'):
                            reshaped_features = qformer._reshape_image_features(vision_features)
                            print(f"  - リシェイプ後: {reshaped_features.shape}")
                            
                            # Q-Formerのforward処理の一部を実行
                            print(f"\n🔄 Q-Formerのforward処理...")
                            
                            # query_embedsを取得
                            query_embeds = qformer.query_tokens.expand(batch_size, -1, -1)
                            print(f"  - query_embeds: {query_embeds.shape}")
                            
                            # image_attsを作成
                            image_atts = torch.ones(reshaped_features.size()[:-1], dtype=torch.long, device=reshaped_features.device)
                            print(f"  - image_atts: {image_atts.shape}")
                            
                            # BLIP-2 Q-Formerのqformerを呼び出す前の準備
                            print(f"\n📋 BLIP-2 Q-Former呼び出し前の状態:")
                            print(f"  - query_embeds: {query_embeds.shape}")
                            print(f"  - encoder_hidden_states (reshaped_features): {reshaped_features.shape}")
                            print(f"  - encoder_attention_mask (image_atts): {image_atts.shape}")
                            
                            # 実際の問題が起きている場所を特定
                            if reshaped_features.shape[-1] != qformer.qformer.config.encoder_hidden_size:
                                print(f"\n❗ 次元不一致検出:")
                                print(f"  - 実際の特徴次元: {reshaped_features.shape[-1]}")
                                print(f"  - Q-Formerが期待する次元: {qformer.qformer.config.encoder_hidden_size}")
                                print(f"  - 一致するか: ❌")
                                
                                # なぜ14次元になっているか調査
                                print(f"\n🔍 14次元の原因調査...")
                                
                                # マルチスケール特徴抽出が有効な場合
                                if hasattr(model.segmentation_head, 'feature_extractor'):
                                    print(f"  - マルチスケール特徴抽出が有効")
                                    feature_extractor = model.segmentation_head.feature_extractor
                                    
                                    # フックが登録されているか確認
                                    if hasattr(feature_extractor, 'hooks') and feature_extractor.hooks:
                                        print(f"  - フック数: {len(feature_extractor.hooks)}")
                                        
                                        # 実際に特徴抽出を実行
                                        multiscale_features = feature_extractor(test_image, image_encoder)
                                        
                                        print(f"  - マルチスケール特徴:")
                                        for name, feat in multiscale_features.items():
                                            if isinstance(feat, torch.Tensor):
                                                print(f"    - {name}: {feat.shape}")
                                                
                                                # 14次元の特徴を探す
                                                if feat.shape[-1] == 14 or (feat.dim() == 4 and feat.shape[1] == 14):
                                                    print(f"    ⚠️ 14次元の特徴を発見！")


if __name__ == "__main__":
    debug_feature_flow()