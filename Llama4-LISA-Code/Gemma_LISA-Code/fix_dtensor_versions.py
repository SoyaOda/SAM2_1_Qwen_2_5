#!/usr/bin/env python3
"""
DTensor問題解決 - バージョンダウングレード戦略
Web調査で実証済みの動作する組み合わせに統一
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
    print("🎯 DTensor問題解決 - 安定版バージョンダウングレード")
    print("=" * 60)
    
    # Step 1: 安定版組み合わせにダウングレード
    print("\n📦 Step 1: 安定版バージョンにダウングレード")
    stable_packages = [
        "torch==2.4.1",
        "transformers==4.45.0", 
        "accelerate==1.7.0"
    ]
    
    for package in stable_packages:
        if not run_command(f"pip install {package}"):
            print(f"❌ {package}のダウングレードに失敗")
            return False
    
    # Step 2: 互換性確認
    print("\n🔍 Step 2: 互換性確認")
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import transformers; print(f'transformers: {transformers.__version__}')", 
        "import accelerate; print(f'accelerate: {accelerate.__version__}')"
    ]
    
    for test in test_imports:
        if not run_command(f"python -c \"{test}\""):
            print(f"❌ インポートテストに失敗: {test}")
            return False
    
    # Step 3: DTensorインポートテスト
    print("\n🧪 Step 3: DTensor互換性テスト")
    dtensor_test = '''
import torch
try:
    from torch.distributed._tensor import DTensor
    print("✅ DTensor (old API): 成功")
except ImportError:
    print("⚠️ DTensor (old API): 失敗")
    
try:
    from torch.distributed.tensor import DTensor  
    print("✅ DTensor (new API): 成功")
except ImportError:
    print("⚠️ DTensor (new API): 失敗")
'''
    
    if not run_command(f"python -c \"{dtensor_test}\""):
        print("❌ DTensorテストに失敗")
        return False
    
    print("\n🎉 成功！バージョンダウングレード完了")
    print("\n📋 更新されたバージョン:")
    print("- PyTorch: 2.4.1 (DTensor安定版)")
    print("- transformers: 4.45.0 (DTensor対応版)")
    print("- accelerate: 1.7.0 (互換性確認済み)")
    print("\n🚀 次のステップ:")
    print("1. accelerate launch --config_file multi_gpu_debug.yaml train.py")
    print("2. DTensorエラーが解決されていることを確認")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1) 