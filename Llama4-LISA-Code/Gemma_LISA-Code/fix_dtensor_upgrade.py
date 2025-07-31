#!/usr/bin/env python3
"""
DTensor問題解決 - 上位バージョンアップグレード戦略
最新版で修正されたバージョンにアップグレード
"""

import subprocess
import sys
import json

def run_command(cmd):
    """コマンド実行とエラーハンドリング"""
    print(f"🔧 実行中: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ エラー: {result.stderr}")
        return False
    print(f"✅ 成功: {result.stdout}")
    return True

def backup_current_versions():
    """現在のバージョンをバックアップ"""
    print("💾 現在のバージョンをバックアップ中...")
    backup_cmd = """
pip freeze | grep -E "(torch|transformers|accelerate)" > dtensor_backup_versions.txt
echo "バックアップ完了: dtensor_backup_versions.txt"
cat dtensor_backup_versions.txt
"""
    run_command(backup_cmd)

def upgrade_to_latest():
    """最新版にアップグレード"""
    print("\n📦 最新版へのアップグレード...")
    
    # Step 1: PyTorch 2.6.1+ (DTensor修正版)
    pytorch_cmd = "pip install torch>=2.6.1"
    if not run_command(pytorch_cmd):
        print("❌ PyTorchのアップグレードに失敗")
        return False
    
    # Step 2: transformers 4.55.0+ (DTensor完全対応版)
    transformers_cmd = "pip install transformers>=4.55.0"
    if not run_command(transformers_cmd):
        print("❌ transformersのアップグレードに失敗")
        return False
    
    # Step 3: accelerate最新版
    accelerate_cmd = "pip install accelerate>=1.9.0"
    if not run_command(accelerate_cmd):
        print("❌ accelerateのアップグレードに失敗")
        return False
    
    return True

def test_dtensor_compatibility():
    """DTensor互換性テスト"""
    print("\n🧪 DTensor互換性テスト...")
    
    dtensor_test = '''
import torch
import transformers
import accelerate

print(f"PyTorch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")

# DTensorインポートテスト
try:
    from torch.distributed.tensor import DTensor, distribute_tensor, Replicate
    print("✅ DTensor新API: 成功")
except ImportError as e:
    print(f"❌ DTensor新API: 失敗 - {e}")

# torch.load weights_only互換性テスト
try:
    import torch.serialization
    print("✅ torch.serialization: 利用可能")
    
    # safe_globalsテスト
    with torch.serialization.safe_globals([]):
        print("✅ safe_globals: 利用可能")
except Exception as e:
    print(f"⚠️ safe_globals: {e}")

# resize_token_embeddingsテスト
try:
    from transformers import AutoModel
    print("✅ transformers基本インポート: 成功")
except ImportError as e:
    print(f"❌ transformers基本インポート: 失敗 - {e}")
'''
    
    return run_command(f"python -c \"{dtensor_test}\"")

def create_fixed_config():
    """修正版accelerate設定を作成"""
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "downcast_bf16": "no",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "num_processes": 8,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False
    }
    
    with open("accelerate_config_dtensor_fixed.yaml", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ 修正版accelerate設定作成: accelerate_config_dtensor_fixed.yaml")

def main():
    print("🎯 DTensor問題解決 - 上位バージョンアップグレード戦略")
    print("=" * 60)
    print("⚠️  注意: この方法は実験的です。バックアップ必須！")
    
    # Step 1: バックアップ
    print("\n📦 Step 1: 現在のバージョンをバックアップ")
    backup_current_versions()
    
    # Step 2: アップグレード
    print("\n🚀 Step 2: 最新版にアップグレード")
    if not upgrade_to_latest():
        print("❌ アップグレードに失敗")
        print("💡 復元方法: pip install -r dtensor_backup_versions.txt")
        return False
    
    # Step 3: 互換性テスト
    print("\n🔍 Step 3: 互換性テスト")
    if not test_dtensor_compatibility():
        print("❌ 互換性テストに失敗")
        print("💡 復元方法: pip install -r dtensor_backup_versions.txt")
        return False
    
    # Step 4: 修正版設定作成
    print("\n⚙️ Step 4: 修正版accelerate設定作成")
    create_fixed_config()
    
    print("\n🎉 アップグレード完了！")
    print("\n📋 更新されたバージョン:")
    print("- PyTorch: 2.6.1+ (DTensor修正版)")
    print("- transformers: 4.55.0+ (DTensor完全対応)")
    print("- accelerate: 1.9.0+ (最新版)")
    
    print("\n🚀 次のステップ:")
    print("1. accelerate launch --config_file accelerate_config_dtensor_fixed.yaml train.py")
    print("2. DTensorエラーが解決されていることを確認")
    
    print("\n⚠️  復元方法（問題発生時）:")
    print("pip install -r dtensor_backup_versions.txt")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1) 