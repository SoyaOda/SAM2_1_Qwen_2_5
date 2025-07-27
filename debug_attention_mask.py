# debug_attention_mask.py
"""
アテンションマスクのサイズ不一致問題をデバッグ
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('.')

from model.enhanced_qformer import EnhancedQFormerModel
from model.sam2_integration import get_sam2_wrapper
import config_linux

def debug_attention_mask():
    """アテンションマスク問題の詳細調査"""
    print("🔍 アテンションマスクデバッグ開始...\n")
    
    # SAM2ラッパー取得
    sam2_config = {
        'model_id': config_linux.SAM2_HF_MODEL_ID,
        'target_dtype': torch.bfloat16,
        'vos_optimized': True,
        'compile_model': False,
        'memory_pathways': 3,
        'mixed_precision': True,
    }
    
    sam_wrapper = get_sam2_wrapper(**sam2_config)
    
    # SAM2エンコーダを取得
    if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
        actual_model = sam_wrapper.predictor.model
        if hasattr(actual_model, 'image_encoder'):
            image_encoder = actual_model.image_encoder
            
            # テスト画像
            test_image = torch.randn(1, 3, 448, 448).cuda()
            
            # エンコーダ実行
            with torch.no_grad():
                encoded_output = image_encoder(test_image)
            
            if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
                vision_features = encoded_output['vision_features']
                print(f"✅ SAM2出力形状: {vision_features.shape}")
                print(f"  - dtype: {vision_features.dtype}")
                
                # Q-Former設定
                qformer_config = {
                    'num_queries': 32,
                    'hidden_size': 768,
                    'num_layers': 12,
                    'num_heads': 12,
                    'intermediate_size': 3072,
                    'dropout': 0.1,
                    'sam_prompt_dim': 256,
                    'encoder_hidden_size': 256,  # SAM2の実際の出力
                    'llm_hidden_size': 5120,
                    'max_txt_len': 64,
                }
                
                # Q-Formerモデル作成
                qformer = EnhancedQFormerModel(qformer_config)
                qformer = qformer.cuda()
                
                # リシェイプ処理を確認
                print("\n📐 リシェイプ処理の確認:")
                print(f"  - 入力形状: {vision_features.shape}")
                
                # _reshape_image_featuresを実行
                reshaped_features = qformer._reshape_image_features(vision_features)
                print(f"  - リシェイプ後: {reshaped_features.shape}")
                
                # アテンションマスクの作成方法を確認
                print("\n🎭 アテンションマスクの確認:")
                
                # 現在の実装
                image_atts_current = torch.ones(reshaped_features.size()[:-1], dtype=torch.long, device=vision_features.device)
                print(f"  - 現在の実装: {image_atts_current.shape}")
                
                # 正しい実装（リシェイプ後の形状に基づく）
                batch_size = reshaped_features.size(0)
                seq_len = reshaped_features.size(1)
                image_atts_correct = torch.ones((batch_size, seq_len), dtype=torch.long, device=vision_features.device)
                print(f"  - 正しい実装: {image_atts_correct.shape}")
                
                # 問題の原因を特定
                if vision_features.dim() == 4:
                    B, C, H, W = vision_features.shape
                    print(f"\n❗ 問題の原因:")
                    print(f"  - 元の空間サイズ: H={H}, W={W}")
                    print(f"  - リシェイプ後のシーケンス長: {H*W}")
                    print(f"  - アテンションマスクが元のHまたはWのサイズ（{H}）になっている可能性")
                
                # Q-Formerのforward処理を模擬
                print("\n🔄 Q-Formerのforward処理シミュレーション:")
                
                try:
                    # 現在の実装でforward実行（エラーが出るはず）
                    query_embeds = qformer.query_tokens.expand(batch_size, -1, -1)
                    
                    # ITGモードでforward
                    outputs = qformer.qformer(
                        query_embeds=query_embeds,
                        encoder_hidden_states=reshaped_features,
                        encoder_attention_mask=image_atts_current,
                        use_cache=False,
                        return_dict=True
                    )
                    print("  ✅ Forward成功（予期しない）")
                except Exception as e:
                    print(f"  ❌ Forward失敗（予期通り）: {str(e)}")
                    
                    # 正しいアテンションマスクで再試行
                    print("\n  🔄 正しいアテンションマスクで再試行:")
                    try:
                        outputs = qformer.qformer(
                            query_embeds=query_embeds,
                            encoder_hidden_states=reshaped_features,
                            encoder_attention_mask=image_atts_correct,
                            use_cache=False,
                            return_dict=True
                        )
                        print("  ✅ Forward成功！")
                        print(f"  - 出力形状: {outputs.last_hidden_state.shape}")
                    except Exception as e2:
                        print(f"  ❌ まだエラー: {str(e2)}")


if __name__ == "__main__":
    debug_attention_mask()