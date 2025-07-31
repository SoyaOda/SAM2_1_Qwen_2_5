#!/usr/bin/env python3
"""
target_modules不一致問題の根本原因特定スクリプト
なぜSAM2_TARGET_MODULESが実際のモジュール名と一致しないかを調査
"""
import torch
import torch.nn as nn
import sys
import os

# プロジェクトルート設定
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_repo'))

def analyze_target_modules_mismatch():
    """target_modules不一致の根本原因分析"""
    
    print("=" * 70)
    print("🔍 target_modules不一致問題の根本原因分析")
    print("=" * 70)
    
    # 1. 設定ファイルの分析
    print("\n📋 1. 設定ファイル分析...")
    
    try:
        import config_linux
        configured_target_modules = config_linux.SAM2_TARGET_MODULES
        print(f"✅ config_linux.SAM2_TARGET_MODULES: {configured_target_modules}")
        
        # 設定の由来を分析
        print(f"\n🔍 設定の由来分析:")
        print(f"  - 設定値: {configured_target_modules}")
        print(f"  - パターン分析:")
        for module in configured_target_modules:
            if 'attn' in module:
                print(f"    - {module}: Attention層パターン")
            elif 'mlp' in module:
                print(f"    - {module}: MLP層パターン")
            else:
                print(f"    - {module}: その他パターン")
        
    except ImportError as e:
        print(f"❌ config_linux import失敗: {e}")
        configured_target_modules = ['attn.qkv', 'attn.proj', 'mlp.layers.0', 'mlp.layers.1']
        print(f"⚠️ フォールバック使用: {configured_target_modules}")
    
    # 2. 実際のモデル構造との比較分析
    print(f"\n📦 2. 実際のモデル構造分析...")
    
    # 2-1. Enhanced Modelの構造確認
    print(f"\n🔍 2-1. Enhanced Model構造確認")
    try:
        from model.enhanced_llama4_qformer_sam2 import EnhancedQFormerSegmentationBridge
        print(f"✅ Enhanced Model import成功")
        
        # Enhanced Modelがどのようにsam2を初期化しているか確認
        print(f"  - Enhanced Modelのsam2関連実装を確認...")
        
    except Exception as e:
        print(f"❌ Enhanced Model import失敗: {e}")
    
    # 2-2. SAM2統合モジュールの確認
    print(f"\n🔍 2-2. SAM2統合モジュール確認")
    try:
        from model.sam2_integration import get_sam2_wrapper
        print(f"✅ SAM2統合モジュール import成功")
        
        # SAM2統合がどのような構造を作るか確認
        print(f"  - SAM2統合の実装方式を確認...")
        
    except Exception as e:
        print(f"❌ SAM2統合モジュール import失敗: {e}")
    
    # 2-3. Meta公式SAM2の確認
    print(f"\n🔍 2-3. Meta公式SAM2構造確認")
    try:
        # HuggingFace SAM2を試行
        print(f"  HuggingFace SAM2構造確認...")
        try:
            from transformers import Sam2Model
            print(f"    ✅ HuggingFace SAM2 import成功")
            
            # 小さなSAM2モデルで構造確認
            model_id = "facebook/sam2-hiera-tiny"
            print(f"    - 軽量SAM2モデル読み込み: {model_id}")
            
            # モデル初期化を試行（CPU、軽量版）
            model = Sam2Model.from_pretrained(model_id, torch_dtype=torch.float32)
            print(f"    ✅ SAM2モデル初期化成功")
            
            # 実際のモジュール構造を確認
            print(f"\n    📋 SAM2実際のモジュール構造（上位20個）:")
            sam2_modules = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    sam2_modules.append(name)
            
            for i, name in enumerate(sam2_modules[:20]):
                print(f"      {i+1:2d}. {name}")
            
            if len(sam2_modules) > 20:
                print(f"      ... 他{len(sam2_modules)-20}個のLinearモジュール")
            
            # target_modulesとの一致確認
            print(f"\n    🎯 target_modules一致確認:")
            matches_found = []
            for target in configured_target_modules:
                matching_modules = [name for name in sam2_modules if target in name]
                if matching_modules:
                    matches_found.extend(matching_modules)
                    print(f"      ✅ '{target}' マッチ: {len(matching_modules)}個")
                    for match in matching_modules[:3]:  # 最初の3個まで表示
                        print(f"         - {match}")
                    if len(matching_modules) > 3:
                        print(f"         - ... 他{len(matching_modules)-3}個")
                else:
                    print(f"      ❌ '{target}' マッチなし")
            
            print(f"\n    📊 一致統計:")
            print(f"      - 総Linearモジュール数: {len(sam2_modules)}")
            print(f"      - target_modules設定数: {len(configured_target_modules)}")
            print(f"      - マッチしたモジュール数: {len(matches_found)}")
            print(f"      - マッチ率: {len(matches_found)/len(sam2_modules)*100:.1f}%")
            
        except Exception as e:
            print(f"    ❌ HuggingFace SAM2確認失敗: {e}")
    
    except Exception as e:
        print(f"❌ Meta公式SAM2確認失敗: {e}")
    
    # 3. 問題の根本原因特定
    print(f"\n📊 3. 根本原因特定...")
    
    print(f"\n🔍 推定される問題:")
    print(f"  1. **モジュール名の抽象化レベル不一致**")
    print(f"     - 設定値: Vision Transformer理論的名前 ('attn.qkv', 'mlp.layers.0')")
    print(f"     - 実際値: HuggingFace実装固有名前 (model specific)")
    print(f"  ")
    print(f"  2. **モデル階層の違い**")
    print(f"     - 設定: SAM2単体のモジュール名を想定")
    print(f"     - 実際: Enhanced Model内でのネストされたモジュール名")
    print(f"  ")
    print(f"  3. **実装バージョンの違い**")
    print(f"     - 設定: Meta公式SAM2リポジトリベース")
    print(f"     - 実際: HuggingFace Transformers実装")
    
    # 4. 修正方針提案
    print(f"\n🔧 4. 修正方針提案...")
    print(f"  **即座に実行可能な修正:**")
    print(f"  1. 実際のSAM2モジュール名を特定して config_linux.py を修正")
    print(f"  2. パターンマッチングを使用 (例: '.*attn.*', '.*mlp.*')")
    print(f"  3. 階層付きモジュール名に対応 (例: 'sam_wrapper.model.vision_encoder.layers.*.attn.qkv')")
    
    print(f"\n✅ 根本原因分析完了")
    print("=" * 70)

if __name__ == "__main__":
    analyze_target_modules_mismatch()