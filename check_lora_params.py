#!/usr/bin/env python3
"""
LoRAパラメータの確認スクリプト
"""
import sys
import os

# プロジェクトルートとsam2パスを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sam2_path = os.path.join(project_root, 'sam2')
if os.path.exists(sam2_path):
    sys.path.insert(0, sam2_path)

from config_qwen_sam import get_config
from model.sam_qwen_model import create_sam_qwen_model
from model.lora_config import create_lora_manager


def main():
    print("=" * 70)
    print("🔍 LoRAパラメータ確認")
    print("=" * 70)
    
    # 設定
    config = get_config('development')
    model_config = config.get_model_config()
    
    # LoRA設定の作成
    lora_manager = create_lora_manager(
        use_qlora=False,
        qwen_r=16,
        sam_r=16,
    )
    
    # モデル初期化（LoRA設定を渡す）
    print("\n🤖 モデル初期化中（LoRA有効）...")
    model = create_sam_qwen_model(model_config, lora_config=lora_manager)
    
    # パラメータ確認
    print("\n📊 パラメータ分析:")
    
    # 全パラメータを確認
    total_params = 0
    trainable_params = 0
    lora_params = 0
    qwen_lora_params = 0
    sam_lora_params = 0
    
    print("\n[LoRA関連パラメータ]")
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
        # LoRA関連パラメータの検出
        if 'lora' in name.lower():
            lora_params += param.numel()
            print(f"  - {name}: {param.shape} (requires_grad={param.requires_grad})")
            
            if 'qwen' in name:
                qwen_lora_params += param.numel()
            elif 'sam' in name or 'mask_decoder' in name:
                sam_lora_params += param.numel()
    
    print(f"\n[統計]")
    print(f"  - 総パラメータ数: {total_params:,}")
    print(f"  - 学習可能パラメータ数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  - LoRAパラメータ数: {lora_params:,} ({lora_params/total_params*100:.4f}%)")
    print(f"    - Qwen LoRA: {qwen_lora_params:,}")
    print(f"    - SAM LoRA: {sam_lora_params:,}")
    
    # Qwenモデルの構造を確認
    print("\n[Qwenモデル内のLoRA対象モジュール]")
    for name, module in model.qwen_model.named_modules():
        if any(target in name for target in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            # PEFTが適用されているか確認
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                print(f"  ✅ {name}: LoRA適用済み")
            else:
                print(f"  ❌ {name}: LoRA未適用")
    
    # SAMモデルの構造を確認
    print("\n[SAMモデル内のLoRA対象モジュール]")
    if hasattr(model.sam_model, 'mask_decoder'):
        for name, module in model.sam_model.mask_decoder.named_modules():
            if any(target in name for target in ['q_proj', 'v_proj']):
                if hasattr(module, 'lora_layer'):
                    print(f"  ✅ {name}: LoRA適用済み（手動実装）")
                else:
                    print(f"  ❌ {name}: LoRA未適用")


if __name__ == "__main__":
    main()