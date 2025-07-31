#!/usr/bin/env python3
"""
Llama4ForCausalLMクラスの存在確認スクリプト
"""
import sys
import importlib.metadata

print("=== Llama4ForCausalLMクラス確認 ===")

# 1. transformersバージョン確認
try:
    transformers_version = importlib.metadata.version('transformers')
    print(f"✅ transformers version: {transformers_version}")
except:
    print("❌ transformersがインストールされていません")
    sys.exit(1)

# 2. Llama4ForCausalLMのインポート試行
print("\n🔍 Llama4ForCausalLMインポート試行...")
try:
    from transformers import Llama4ForCausalLM
    print("✅ Llama4ForCausalLM import成功！")
    print(f"  - クラス: {Llama4ForCausalLM}")
    print(f"  - モジュール: {Llama4ForCausalLM.__module__}")
except ImportError as e:
    print(f"❌ Llama4ForCausalLM import失敗: {e}")
    
    # 3. 代替案: AutoModelForCausalLM経由
    print("\n🔍 AutoModelForCausalLM経由での確認...")
    try:
        from transformers import AutoModelForCausalLM
        print("✅ AutoModelForCausalLM import成功")
        
        # モデルマッピング確認
        if hasattr(AutoModelForCausalLM, '_model_mapping'):
            print("📋 利用可能なCausalLMモデル:")
            for key in AutoModelForCausalLM._model_mapping._model_mapping.keys():
                if 'llama' in str(key).lower():
                    print(f"  - {key}")
    except Exception as e2:
        print(f"❌ AutoModelForCausalLM import失敗: {e2}")

# 4. 利用可能なLlamaクラス一覧
print("\n📋 利用可能なLlamaクラス検索...")
try:
    import transformers
    llama_classes = [attr for attr in dir(transformers) if 'Llama' in attr and 'CausalLM' in attr]
    if llama_classes:
        print("✅ 見つかったLlama CausalLMクラス:")
        for cls in llama_classes:
            print(f"  - {cls}")
    else:
        print("⚠️ Llama CausalLMクラスが見つかりません")
except Exception as e:
    print(f"❌ クラス検索エラー: {e}")

# 5. 推奨事項
print("\n💡 推奨事項:")
print("1. transformersを最新版に更新: pip install --upgrade transformers")
print("2. Llama4サポートバージョン確認: transformers>=4.45.0")
print("3. AutoModelForCausalLMを使用（自動クラス選択）")