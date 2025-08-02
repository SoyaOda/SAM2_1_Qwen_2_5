"""
SAM2.1とQwen2.5-VLの実際の次元を確認するスクリプト
"""
import torch
import sys
import os

# プロジェクトルートとsam2パスを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sam2_path = os.path.join(project_root, 'sam2')
if os.path.exists(sam2_path):
    sys.path.insert(0, sam2_path)

from sam2.build_sam import build_sam2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig

def check_sam2_dimensions():
    """SAM2.1の実際の出力次元を確認"""
    print("="*80)
    print("🔍 SAM2.1の次元確認")
    print("="*80)
    
    # SAM2.1モデル構築
    config_name = "sam2_hiera_l.yaml"
    ckpt_path = os.path.join(project_root, 'checkpoints', 'sam2_hiera_large.pt')
    
    sam_model = build_sam2(config_name, ckpt_path, device="cuda")
    
    # ダミー画像でテスト
    dummy_image = torch.randn(1, 3, 1024, 1024).cuda()
    
    # image_encoderの出力を確認
    with torch.no_grad():
        backbone_out = sam_model.image_encoder(dummy_image)
    
    print("\n📊 SAM2.1 image_encoder出力:")
    print(f"  - backbone_out keys: {backbone_out.keys()}")
    
    if "vision_features" in backbone_out:
        vf = backbone_out["vision_features"]
        print(f"  - vision_features shape: {vf.shape}")
        print(f"  - vision_features dtype: {vf.dtype}")
    
    if "backbone_fpn" in backbone_out:
        print(f"\n  - backbone_fpn levels: {len(backbone_out['backbone_fpn'])}")
        for i, feat in enumerate(backbone_out['backbone_fpn']):
            print(f"    - Level {i}: {feat.shape}")
    
    # Hiera backboneの確認
    if hasattr(sam_model.image_encoder, 'neck'):
        neck = sam_model.image_encoder.neck
        print(f"\n  - Neck (FPN) info:")
        print(f"    - Type: {type(neck)}")
        if hasattr(neck, 'd_model'):
            print(f"    - d_model: {neck.d_model}")
        if hasattr(neck, 'fpn_dims'):
            print(f"    - fpn_dims: {getattr(neck, 'fpn_dims', 'N/A')}")
    
    # 実際のbackbone出力次元を確認
    if hasattr(sam_model.image_encoder, 'trunk'):
        trunk = sam_model.image_encoder.trunk
        print(f"\n  - Trunk (Hiera) info:")
        print(f"    - Type: {type(trunk)}")
        
        # Hieraの各ステージの出力次元を確認
        if hasattr(trunk, 'embed_dim'):
            print(f"    - embed_dim: {trunk.embed_dim}")
        if hasattr(trunk, 'stages'):
            print(f"    - num_stages: {len(trunk.stages)}")
            for i, stage in enumerate(trunk.stages):
                if hasattr(stage, 'dim'):
                    print(f"      - Stage {i} dim: {stage.dim}")
                elif hasattr(stage, 'out_channels'):
                    print(f"      - Stage {i} out_channels: {stage.out_channels}")

def check_qwen_dimensions():
    """Qwen2.5-VL-3Bの実際の次元を確認"""
    print("\n" + "="*80)
    print("🔍 Qwen2.5-VL-3Bの次元確認")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # 設定のみ読み込み（モデル全体は読み込まない）
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    print("\n📊 Qwen2.5-VL-3B設定:")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - vocab_size: {config.vocab_size}")
    
    if hasattr(config, 'vision_config'):
        vc = config.vision_config
        print(f"\n  - Vision config:")
        print(f"    - hidden_size: {getattr(vc, 'hidden_size', 'N/A')}")
        print(f"    - image_size: {getattr(vc, 'image_size', 'N/A')}")
        print(f"    - patch_size: {getattr(vc, 'patch_size', 'N/A')}")
        print(f"    - num_hidden_layers: {getattr(vc, 'num_hidden_layers', 'N/A')}")
    
    # モデルアーキテクチャの詳細
    print(f"\n  - Model architecture:")
    print(f"    - num_hidden_layers: {config.num_hidden_layers}")
    print(f"    - num_attention_heads: {config.num_attention_heads}")
    print(f"    - intermediate_size: {config.intermediate_size}")

def main():
    """メイン実行"""
    print("🚀 モデル次元確認開始...")
    
    # SAM2.1の次元確認
    check_sam2_dimensions()
    
    # Qwen2.5-VLの次元確認
    check_qwen_dimensions()
    
    print("\n" + "="*80)
    print("✅ 次元確認完了")
    print("="*80)
    
    print("\n📌 重要な発見:")
    print("1. SAM2.1のFPN出力は256次元に統一されている（o3_spec6.mdの1280次元は要確認）")
    print("2. Qwen2.5-VL-3Bのhidden_sizeを確認して正しい投影層を設計する必要がある")
    print("3. SAM2.1のHiera backboneの最終段階の次元が実際のViT出力次元")

if __name__ == "__main__":
    main()