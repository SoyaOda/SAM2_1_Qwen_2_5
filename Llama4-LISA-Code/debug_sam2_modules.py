#!/usr/bin/env python3
"""
SAM2のHiera architectureにおけるLoRA target_modulesの特定
SAM2モデルの実際のモジュール構造を調査し、LoRA対応可能なLinear層を特定する
"""

import torch
import torch.nn as nn
from transformers import Sam2Model, Sam2Processor, Sam2Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sam2_modules():
    """SAM2モデルの構造を分析し、LoRA target_modulesを特定"""
    
    print("=" * 60)
    print("SAM2 Hiera Architecture LoRA Target Modules Analysis")
    print("=" * 60)
    
    try:
        # SAM2モデルの設定とプロセッサーをロード
        print("\n1. Loading SAM2 model configuration...")
        model_id = "facebook/sam2-hiera-large"
        
        # CPU上でモデルをロード（メモリ節約のため）
        model = Sam2Model.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        processor = Sam2Processor.from_pretrained(model_id)
        
        print(f"✓ Model loaded: {model_id}")
        print(f"✓ Model config: {model.config}")
        
        # モデル全体の構造を表示
        print("\n2. Model architecture overview:")
        print(model)
        
        # state_dictのキーを分析
        print("\n3. Analyzing state_dict keys...")
        state_dict = model.state_dict()
        
        # すべてのキーを表示
        print(f"Total parameters: {len(state_dict)} keys")
        
        # Attention関連のモジュールを特定
        attention_modules = []
        linear_modules = []
        conv_modules = []
        
        for name, param in state_dict.items():
            print(f"{name}: {param.shape}")
            
            # Attention関連
            if any(keyword in name.lower() for keyword in ['attn', 'attention', 'query', 'key', 'value', 'qkv']):
                attention_modules.append(name)
            
            # Linear層関連
            if 'weight' in name and len(param.shape) == 2:
                linear_modules.append(name)
            
            # Conv層関連
            if 'weight' in name and len(param.shape) == 4:
                conv_modules.append(name)
        
        # 結果の分析と表示
        print("\n4. Attention-related modules:")
        print("-" * 40)
        for module in attention_modules:
            print(f"  - {module}")
        
        print(f"\nTotal attention modules: {len(attention_modules)}")
        
        print("\n5. Linear layer modules (potential LoRA targets):")
        print("-" * 40)
        for module in linear_modules[:20]:  # 最初の20個を表示
            print(f"  - {module}")
        
        if len(linear_modules) > 20:
            print(f"  ... and {len(linear_modules) - 20} more")
        
        print(f"\nTotal linear modules: {len(linear_modules)}")
        
        # Named modulesから直接構造を調査
        print("\n6. Direct module structure analysis:")
        print("-" * 40)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"Linear: {name} -> {module}")
            elif isinstance(module, nn.MultiheadAttention):
                print(f"MultiheadAttention: {name} -> {module}")
            elif 'attention' in name.lower() or 'attn' in name.lower():
                print(f"Attention-related: {name} -> {type(module)}")
        
        # LoRA target_modules の推奨リストを生成
        print("\n7. Recommended LoRA target_modules:")
        print("-" * 40)
        
        # SAM2 Hiera特有のattentionモジュール名パターンを特定
        target_modules = set()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # モジュール名の最後の部分を取得
                module_parts = name.split('.')
                if len(module_parts) > 0:
                    last_part = module_parts[-1]
                    
                    # 典型的なattention projection名をチェック
                    if any(pattern in last_part for pattern in ['q', 'k', 'v', 'query', 'key', 'value', 'qkv', 'proj']):
                        target_modules.add(last_part)
                    
                    # MLP layers
                    if any(pattern in last_part for pattern in ['fc', 'linear', 'mlp']):
                        target_modules.add(last_part)
        
        print("Identified module patterns:")
        for module in sorted(target_modules):
            print(f"  - {module}")
        
        # 実際のSAM2モジュール構造に基づく推奨設定
        print("\n8. Suggested LoRA configurations:")
        print("-" * 40)
        
        print("Option 1 - Attention layers only:")
        attention_targets = [m for m in target_modules if any(pattern in m for pattern in ['q', 'k', 'v', 'qkv', 'attn'])]
        print(f"  target_modules = {sorted(attention_targets)}")
        
        print("\nOption 2 - Attention + MLP:")
        all_targets = [m for m in target_modules if any(pattern in m for pattern in ['q', 'k', 'v', 'qkv', 'attn', 'fc', 'mlp'])]
        print(f"  target_modules = {sorted(all_targets)}")
        
        print("\nOption 3 - All linear layers:")
        print(f"  target_modules = {sorted(target_modules)}")
        
        return target_modules, attention_modules, linear_modules
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_lora_config_compatibility():
    """特定されたtarget_modulesでLoRA設定をテスト"""
    
    print("\n9. Testing LoRA configuration compatibility:")
    print("-" * 40)
    
    try:
        from peft import LoraConfig, get_peft_model
        
        # SAM2モデルを再ロード
        model_id = "facebook/sam2-hiera-large"
        model = Sam2Model.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # 異なるtarget_modules設定をテスト
        test_configs = [
            # Web調査で見つかった一般的なパターン
            ["qkv"],
            ["q", "v"],
            ["query", "key", "value"],
            ["proj"],
            # SAM特有の可能性があるパターン
            ["to_q", "to_k", "to_v"],
            ["in_proj"],
            ["out_proj"],
        ]
        
        for target_modules in test_configs:
            try:
                print(f"\nTesting target_modules: {target_modules}")
                
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=target_modules,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="FEATURE_EXTRACTION"
                )
                
                # LoRAモデルの作成を試行
                lora_model = get_peft_model(model, lora_config)
                print(f"  ✓ SUCCESS: {target_modules} worked!")
                
                # LoRAパラメータの数を確認
                total_params = sum(p.numel() for p in lora_model.parameters())
                trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
                print(f"  Total params: {total_params:,}")
                print(f"  Trainable params: {trainable_params:,}")
                print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
                
                break  # 成功したら終了
                
            except Exception as e:
                print(f"  ✗ FAILED: {target_modules} - {str(e)}")
                continue
        
    except ImportError:
        print("PEFT library not available, skipping LoRA compatibility test")
    except Exception as e:
        print(f"Error during LoRA testing: {e}")

if __name__ == "__main__":
    target_modules, attention_modules, linear_modules = analyze_sam2_modules()
    
    if target_modules is not None:
        test_lora_config_compatibility()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("SAM2のHiera architectureでLoRAを適用する場合:")
        print("1. まず 'qkv' パターンを試す")
        print("2. 失敗した場合は 'q', 'v' または 'query', 'key', 'value' を試す")
        print("3. SAM特有のモジュール名も考慮する必要がある")
        print("4. Web調査によると、SAM2でのLoRA効果は限定的な可能性がある")
        print("=" * 60)