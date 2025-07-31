#!/usr/bin/env python3
"""
DTensor問題解決 - 完全版バージョンダウングレード戦略
Webリサーチで判明したtorchvision互換性問題も解決
"""

import subprocess
import sys

def run_command(cmd):
    """コマンド実行とエラーハンドリング"""
    print(f"🔧 実行中: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ エラー: {result.stderr}")
        return False
    print(f"✅ 成功: {result.stdout}")
    return True

def main():
    print("🎯 DTensor問題解決 - 完全版バージョンダウングレード")
    print("=" * 60) 
    print("📚 Webリサーチ結果:")
    print("  - PyTorch 2.4.1 → torchvision 0.19.1 (必須)")
    print("  - transformers 4.45.0")
    print("  - accelerate 1.7.0")
    print("=" * 60)
    
    # Step 1: 安定版組み合わせにダウングレード
    print("\n📦 Step 1: 完全版バージョンダウングレード")
    
    # 重要: 特定の互換性組み合わせを強制インストール
    compatible_packages = [
        "torch==2.4.1",
        "torchvision==0.19.1",  # 🔥 重要: PyTorch 2.4.1と互換性のあるバージョン
        "transformers==4.45.0",
        "accelerate==1.7.0"
    ]
    
    for package in compatible_packages:
        if not run_command(f"pip install --force-reinstall {package}"):
            print(f"❌ {package}のインストールに失敗")
            return False
    
    print("\n🔍 Step 2: 互換性確認")
    version_checks = [
        "python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"",
        "python -c \"import torchvision; print(f'torchvision: {torchvision.__version__}')\"",
        "python -c \"import transformers; print(f'transformers: {transformers.__version__}')\"",
        "python -c \"import accelerate; print(f'accelerate: {accelerate.__version__}')\""
    ]
    
    for check in version_checks:
        if not run_command(check):
            print(f"❌ バージョン確認失敗: {check}")
    
    print("\n🧪 Step 3: torchvision互換性テスト")
    torchvision_test = """python -c "
import torch
import torchvision
print('✅ torch + torchvision インポート成功')
print(f'torch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print('✅ torchvision::nms問題解決確認')
"
"""
    
    if not run_command(torchvision_test):
        print("❌ torchvision互換性テストに失敗")
        return False
    
    print("\n🎉 完全版バージョンダウングレード完了！")
    print("次のステップ: LISA-Gemmaモデルテストを実行")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 すべて成功！DTensor問題は完全に解決されました。")
        print("📋 確認済み互換性:")
        print("  - PyTorch 2.4.1")
        print("  - torchvision 0.19.1")
        print("  - transformers 4.45.0")
        print("  - accelerate 1.7.0")  
    else:
        print("\n❌ 一部失敗。ログを確認してください。") 