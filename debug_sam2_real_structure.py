#!/usr/bin/env python3
"""
SAM2実際のモジュール構造確認スクリプト
Enhanced Modelが実際にどのようなSAM2モジュール名を作成するかを調査
"""
import torch
import torch.nn as nn
import sys
import os

# プロジェクトルート設定
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_repo'))

def analyze_sam2_real_structure():
    """SAM2実際のモジュール構造分析"""
    
    print("=" * 70)
    print("🔍 SAM2実際のモジュール構造分析")
    print("=" * 70)
    
    # 1. SAM2統合モジュールの構造確認
    print("\n📦 1. SAM2統合モジュール構造確認...")
    
    try:
        from model.sam2_integration import get_sam2_wrapper, SAM2Wrapper
        print("✅ SAM2統合モジュール import成功")
        
        # 軽量SAM2Wrapperを作成
        print("\n🔧 軽量SAM2Wrapper作成...")
        
        # Meta公式SAM2が利用可能かチェック
        try:
            # CPU+軽量版でテスト初期化
            sam2_wrapper = SAM2Wrapper(
                model_id="facebook/sam2-hiera-tiny",  # 最軽量版
                device="cpu",  # CPUで安全に試行
                target_dtype=torch.float32,  # CPU互換
                debug_mode=True,
                compile_model=False  # コンパイル無効
            )
            print("✅ SAM2Wrapper初期化成功")
            
            # 実際のSAM2モデル構造を確認
            print("\n📋 SAM2実際のモジュール構造:")
            
            # SAM2Wrapper内のpredictorからモデルを取得
            if hasattr(sam2_wrapper, 'predictor') and hasattr(sam2_wrapper.predictor, 'model'):
                sam2_model = sam2_wrapper.predictor.model
                print(f"✅ SAM2モデル取得成功: {type(sam2_model)}")
                
                # 全モジュール名を収集
                all_modules = []
                linear_modules = []
                attention_modules = []
                mlp_modules = []
                
                for name, module in sam2_model.named_modules():
                    all_modules.append(name)
                    
                    if isinstance(module, nn.Linear):
                        linear_modules.append(name)
                        
                        # パターン分析
                        if any(keyword in name.lower() for keyword in ['attn', 'attention', 'qkv', 'proj']):
                            attention_modules.append(name)
                        elif any(keyword in name.lower() for keyword in ['mlp', 'feed', 'fc']):
                            mlp_modules.append(name)
                
                # 統計表示
                print(f"\n📊 SAM2モジュール統計:")
                print(f"  - 総モジュール数: {len(all_modules)}")
                print(f"  - Linearモジュール数: {len(linear_modules)}")
                print(f"  - Attentionモジュール数: {len(attention_modules)}")
                print(f"  - MLPモジュール数: {len(mlp_modules)}")
                
                # 実際のモジュール名（上位20個）
                print(f"\n📋 SAM2 Linearモジュール名（上位20個）:")
                for i, name in enumerate(linear_modules[:20]):
                    print(f"  {i+1:2d}. {name}")
                
                if len(linear_modules) > 20:
                    print(f"  ... 他{len(linear_modules)-20}個")
                
                # 設定されたtarget_modulesとの一致確認
                print(f"\n🎯 現在のtarget_modules一致確認:")
                
                configured_targets = ['attn.qkv', 'attn.proj', 'mlp.layers.0', 'mlp.layers.1']
                print(f"  設定値: {configured_targets}")
                
                matching_count = 0
                for target in configured_targets:
                    matches = [name for name in linear_modules if target in name]
                    if matches:
                        matching_count += len(matches)
                        print(f"  ✅ '{target}' マッチ: {len(matches)}個")
                        for match in matches[:3]:
                            print(f"     - {match}")
                    else:
                        print(f"  ❌ '{target}' マッチなし")
                
                print(f"\n📊 一致統計:")
                print(f"  - マッチしたモジュール数: {matching_count}")
                print(f"  - 総Linearモジュール数: {len(linear_modules)}")
                print(f"  - マッチ率: {matching_count/len(linear_modules)*100 if linear_modules else 0:.1f}%")
                
                # 推奨target_modules生成
                print(f"\n💡 推奨target_modules生成:")
                
                # パターンベースの候補
                attention_candidates = [name for name in linear_modules if any(kw in name.lower() for kw in ['qkv', 'q_proj', 'k_proj', 'v_proj', 'out_proj'])]
                mlp_candidates = [name for name in linear_modules if any(kw in name.lower() for kw in ['mlp', 'fc', 'feed_forward'])]
                
                print(f"  Attention系候補 ({len(attention_candidates)}個):")
                for name in attention_candidates[:5]:
                    print(f"    - {name}")
                if len(attention_candidates) > 5:
                    print(f"    - ... 他{len(attention_candidates)-5}個")
                
                print(f"  MLP系候補 ({len(mlp_candidates)}個):")
                for name in mlp_candidates[:5]:
                    print(f"    - {name}")
                if len(mlp_candidates) > 5:
                    print(f"    - ... 他{len(mlp_candidates)-5}個")
                
                # 最適なtarget_modules提案
                suggested_targets = []
                
                # 一般的なパターンを抽出
                common_patterns = []
                for name in linear_modules:
                    # 共通パターンを抽出（簡単な実装）
                    parts = name.split('.')
                    if len(parts) >= 2:
                        # 最後の2つの要素を使用
                        pattern = '.'.join(parts[-2:])
                        if pattern not in common_patterns:
                            common_patterns.append(pattern)
                
                print(f"\n🔧 修正提案:")
                print(f"  現在の設定を以下に変更することを推奨:")
                print(f"  ")
                print(f"  config_linux.py の SAM2_TARGET_MODULES = [")
                
                # 実際に存在するパターンを基に提案
                useful_patterns = []
                for pattern in common_patterns[:8]:  # 上位8個まで
                    matching_modules = [name for name in linear_modules if pattern in name]
                    if len(matching_modules) >= 2:  # 2個以上マッチするパターンのみ
                        useful_patterns.append(pattern)
                        print(f"      \"{pattern}\",  # {len(matching_modules)}個のモジュールにマッチ")
                
                print(f"  ]")
                
                return {
                    'total_modules': len(all_modules),
                    'linear_modules': len(linear_modules),
                    'attention_modules': len(attention_modules),
                    'mlp_modules': len(mlp_modules),
                    'matching_count': matching_count,
                    'suggested_patterns': useful_patterns,
                    'linear_module_names': linear_modules
                }
                
            else:
                print("❌ SAM2モデル取得失敗: predictor.modelが見つかりません")
                return None
                
        except Exception as e:
            print(f"❌ SAM2Wrapper初期化失敗: {e}")
            print(f"  - エラータイプ: {type(e).__name__}")
            
            # フォールバック: ダミー構造で推定
            print(f"\n🔧 フォールバック: 理論的SAM2構造推定...")
            print(f"  一般的なVision Transformerパターン:")
            print(f"    - encoder.layers.*.attn.qkv")
            print(f"    - encoder.layers.*.attn.proj") 
            print(f"    - encoder.layers.*.mlp.fc1")
            print(f"    - encoder.layers.*.mlp.fc2")
            print(f"    - image_encoder.blocks.*.attn.qkv")
            print(f"    - image_encoder.blocks.*.mlp.layers.0")
            
            return None
            
    except Exception as e:
        print(f"❌ SAM2統合モジュール import失敗: {e}")
        return None

def main():
    """メイン実行"""
    print("🚀 SAM2実際のモジュール構造分析開始...")
    
    result = analyze_sam2_real_structure()
    
    if result:
        print(f"\n✅ 分析完了: 有用な情報を取得しました")
        print(f"  - 推奨パターン数: {len(result['suggested_patterns'])}")
        print(f"  - 総Linearモジュール数: {result['linear_modules']}")
    else:
        print(f"\n⚠️ 分析不完全: Lambda Cloud環境での再実行を推奨")
    
    print("=" * 70)

if __name__ == "__main__":
    main()