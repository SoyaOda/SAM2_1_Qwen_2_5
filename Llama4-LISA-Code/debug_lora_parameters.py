#!/usr/bin/env python3
"""
LoRAパラメータ初期化デバッグスクリプト
Web調査結果に基づくPEFT標準準拠の詳細デバッグ
"""
import torch
import torch.nn as nn
import sys
import os

# プロジェクトルート設定
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_repo'))

# ローカル環境用の軽量設定
SAM2_TARGET_MODULES = [
    "attn.qkv",
    "attn.proj", 
    "mlp.layers.0",
    "mlp.layers.1"
]

def create_simple_model_for_debug():
    """デバッグ用の簡単なモデル作成"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # SAM2風のモジュール構造を模擬
            self.image_encoder = nn.Sequential(
                nn.Linear(256, 512, bias=True),  # attn.qkv
                nn.Linear(512, 256, bias=True),  # attn.proj
                nn.Linear(256, 1024, bias=True), # mlp.layers.0
                nn.Linear(1024, 256, bias=True), # mlp.layers.1
                nn.Linear(256, 128, bias=True),  # other.layer
            )
            
        def forward(self, x):
            return self.image_encoder(x)
    
    return SimpleModel()

def analyze_lora_parameters(model, verbose=True):
    """LoRAパラメータの詳細分析（Web調査準拠）"""
    
    print("\n" + "="*60)
    print("LoRAパラメータ詳細分析 (Web調査準拠PEFT準拠)")
    print("="*60)
    
    # 1. モジュール名検査（PEFT準拠named_modules探索）
    all_modules = {}
    lora_modules = {}
    linear_modules = {}
    
    for name, module in model.named_modules():
        all_modules[name] = type(module).__name__
        
        # LoRAキーワード検出
        if any(keyword in name.lower() for keyword in ['lora', 'expert']):
            lora_modules[name] = {
                'type': type(module).__name__,
                'module': module
            }
        
        # Linearレイヤー検出
        if isinstance(module, nn.Linear):
            linear_modules[name] = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None
            }
    
    print(f"📊 モジュール統計:")
    print(f"  - 全モジュール数: {len(all_modules)}")
    print(f"  - LoRA関連モジュール数: {len(lora_modules)}")
    print(f"  - Linearモジュール数: {len(linear_modules)}")
    
    # 2. LoRAパラメータ詳細確認
    total_lora_params = 0
    lora_param_details = []
    
    for name, info in lora_modules.items():
        module = info['module']
        module_params = 0
        param_breakdown = {}
        
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                module_params += param_count
                param_breakdown[param_name] = {
                    'shape': list(param.shape),
                    'count': param_count,
                    'requires_grad': param.requires_grad,
                    'device': param.device,
                    'dtype': param.dtype
                }
        
        total_lora_params += module_params
        lora_param_details.append({
            'name': name,
            'type': info['type'],
            'total_params': module_params,
            'param_breakdown': param_breakdown
        })
    
    print(f"\n🔍 LoRAパラメータ詳細:")
    print(f"  - 総LoRAパラメータ数: {total_lora_params:,}")
    
    # 3. パラメータ詳細表示（上位10個）
    if lora_param_details:
        print(f"\n📋 LoRAモジュール詳細（上位10個）:")
        sorted_details = sorted(lora_param_details, key=lambda x: x['total_params'], reverse=True)
        
        for i, detail in enumerate(sorted_details[:10]):
            print(f"  {i+1}. {detail['name']}")
            print(f"     - タイプ: {detail['type']}")
            print(f"     - パラメータ数: {detail['total_params']:,}")
            
            if verbose and detail['param_breakdown']:
                print(f"     - パラメータ詳細:")
                for param_name, param_info in detail['param_breakdown'].items():
                    print(f"       - {param_name}: {param_info['shape']} ({param_info['count']:,}個)")
            print()
    
    # 4. SAM2特化チェック
    sam2_modules = {}
    sam2_target_found = []
    
    for name, module in model.named_modules():
        if any(target in name for target in SAM2_TARGET_MODULES):
            sam2_modules[name] = {
                'type': type(module).__name__,
                'target_match': [t for t in SAM2_TARGET_MODULES if t in name]
            }
            sam2_target_found.extend(sam2_modules[name]['target_match'])
    
    print(f"🎯 SAM2ターゲットモジュール分析:")
    print(f"  - 設定ターゲット: {SAM2_TARGET_MODULES}")
    print(f"  - 発見モジュール数: {len(sam2_modules)}")
    print(f"  - マッチしたターゲット: {list(set(sam2_target_found))}")
    
    if sam2_modules and verbose:
        print(f"  - 発見されたSAM2モジュール:")
        for name, info in list(sam2_modules.items())[:5]:
            print(f"    - {name}: {info['type']} (マッチ: {info['target_match']})")
    
    # 5. 全体パラメータ統計
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 全体パラメータ統計:")
    print(f"  - 総パラメータ数: {total_params:,}")
    print(f"  - 学習可能パラメータ数: {trainable_params:,}")
    print(f"  - LoRAパラメータ数: {total_lora_params:,}")
    print(f"  - LoRA比率: {(total_lora_params/total_params*100) if total_params > 0 else 0:.4f}%")
    
    # 6. 推定問題診断
    print(f"\n🔍 問題診断:")
    
    if total_lora_params == 0:
        print(f"  ❌ LoRAパラメータが0個: 初期化に失敗している可能性")
        print(f"    - 原因候補1: target_modulesが実際のモジュール名と不一致")
        print(f"    - 原因候補2: LoRA注入処理でエラーが発生")
        print(f"    - 原因候補3: requires_grad=Falseに設定されている")
        
        if len(sam2_target_found) == 0:
            print(f"    🔧 修正提案: SAM2_TARGET_MODULESを実際のモジュール名に修正")
            print(f"    - 実際のLinearモジュール名（参考用）:")
            for name in list(linear_modules.keys())[:5]:
                print(f"      - {name}")
    else:
        print(f"  ✅ LoRAパラメータが正常に初期化されています")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params, 
        'lora_params': total_lora_params,
        'lora_modules_count': len(lora_modules),
        'sam2_modules_found': len(sam2_modules),
        'sam2_target_matches': list(set(sam2_target_found))
    }

def inject_simple_lora_to_model(model, target_modules, rank=16, alpha=16.0):
    """簡単なLoRA注入（ローカルデバッグ用）"""
    print(f"🔧 簡単なLoRA注入開始...")
    print(f"  - ターゲット: {target_modules}")
    
    injected_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name:
                    print(f"  - LoRA注入対象発見: {name}")
                    
                    # 簡単なLoRAパラメータを追加
                    in_features = module.in_features
                    out_features = module.out_features
                    
                    # LoRAパラメータを直接モジュールに追加
                    module.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
                    module.lora_B = nn.Parameter(torch.zeros(out_features, rank))
                    module.lora_scaling = alpha / rank
                    
                    print(f"    - LoRAパラメータ追加: A={module.lora_A.shape}, B={module.lora_B.shape}")
                    injected_count += 1
                    break
    
    print(f"✅ LoRA注入完了: {injected_count}個のモジュール")
    return model

def main():
    """メイン実行（ローカル版）"""
    print("🚀 LoRAパラメータデバッグ開始（ローカル版）...")
    
    # 1. 簡単なモデル作成
    print("\n📦 デバッグ用モデル作成...")
    model = create_simple_model_for_debug()
    
    # モジュール名を確認
    print("\n📋 作成されたモジュール一覧:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  - {name}: {type(module).__name__} ({module.in_features}→{module.out_features})")
    
    # 2. LoRA注入前の分析
    print("\n📊 LoRA注入前の分析:")
    pre_stats = analyze_lora_parameters(model, verbose=False)
    
    # 3. LoRA注入
    print("\n🔧 LoRA注入実行...")
    model = inject_simple_lora_to_model(model, SAM2_TARGET_MODULES)
    
    # 4. LoRA注入後の分析
    print("\n📊 LoRA注入後の分析:")
    post_stats = analyze_lora_parameters(model, verbose=True)
    
    # 5. 結果サマリー
    print("\n" + "="*60)
    print("📋 デバッグ結果サマリー")
    print("="*60)
    print(f"📊 注入前:")
    print(f"  - 総パラメータ数: {pre_stats['total_params']:,}")
    print(f"  - LoRAパラメータ数: {pre_stats['lora_params']:,}")
    print(f"📊 注入後:")
    print(f"  - 総パラメータ数: {post_stats['total_params']:,}")
    print(f"  - 学習可能パラメータ数: {post_stats['trainable_params']:,}")
    print(f"  - LoRAパラメータ数: {post_stats['lora_params']:,}")
    print(f"  - LoRAモジュール数: {post_stats['lora_modules_count']}")
    print(f"  - SAM2ターゲットマッチ: {post_stats['sam2_target_matches']}")
    
    # 診断結果
    if post_stats['lora_params'] > 0:
        print(f"\n✅ LoRAパラメータが正常に初期化されました")
        print(f"   - 増加パラメータ数: {post_stats['lora_params'] - pre_stats['lora_params']:,}")
        print(f"   - target_modules設定は有効です")
    else:
        print(f"\n❌ LoRAパラメータ初期化に失敗")
        print(f"   - target_modules設定に問題がある可能性があります")
    
    print("="*60)

if __name__ == "__main__":
    main()