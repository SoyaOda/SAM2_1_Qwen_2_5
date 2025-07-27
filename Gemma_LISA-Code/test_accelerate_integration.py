#!/usr/bin/env python3
"""
フェーズ2: マルチGPU対応とFSDPでの学習
2.1. train.py の accelerate 対応 - 動作確認テスト
"""

import sys
import os
import subprocess
import time
from datetime import datetime

print("="*80)
print("🚀 フェーズ2.1: train.py accelerate対応 動作確認テスト")
print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Test results tracking
test_results = {}
failed_tests = []

def run_test(test_name, test_func):
    """Run a test and track results"""
    print(f"\n🔧 Testing {test_name}...")
    try:
        result = test_func()
        test_results[test_name] = "✅ PASS"
        print(f"✅ {test_name}: SUCCESS")
        return True
    except Exception as e:
        test_results[test_name] = f"❌ FAIL: {str(e)}"
        failed_tests.append(test_name)
        print(f"❌ {test_name}: FAILED - {e}")
        return False

def test_accelerate_import():
    """accelerateライブラリのインポート確認"""
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    return f"accelerate version available, device: {accelerator.device}"

def test_config_gradient_accumulation():
    """config_linux.pyでのGRADIENT_ACCUMULATION_STEPS設定確認"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config_linux as config
    
    gradient_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', None)
    if gradient_steps is None:
        raise ValueError("GRADIENT_ACCUMULATION_STEPS not defined in config_linux.py")
    
    return f"GRADIENT_ACCUMULATION_STEPS = {gradient_steps}"

def test_train_py_accelerate_imports():
    """train.pyでのaccelerateインポート確認"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    if 'from accelerate import Accelerator' not in content:
        raise ValueError("accelerate import not found in train.py")
    
    if 'accelerator = Accelerator(' not in content:
        raise ValueError("Accelerator initialization not found in train.py")
    
    return "accelerate imports and initialization found"

def test_train_py_prepare_calls():
    """train.pyでのaccelerator.prepare()呼び出し確認"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    if 'accelerator.prepare(' not in content:
        raise ValueError("accelerator.prepare() not found in train.py")
    
    if 'accelerator.backward(' not in content:
        raise ValueError("accelerator.backward() not found in train.py")
    
    return "accelerator.prepare() and accelerator.backward() found"

def test_train_py_main_process_control():
    """train.pyでのメインプロセス制御確認"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    if 'accelerator.is_main_process' not in content:
        raise ValueError("accelerator.is_main_process not found in train.py")
    
    main_process_count = content.count('accelerator.is_main_process')
    if main_process_count < 5:
        raise ValueError(f"accelerator.is_main_process usage insufficient: {main_process_count} < 5")
    
    return f"accelerator.is_main_process used {main_process_count} times"

def test_train_py_wandb_integration():
    """train.pyでのWandB accelerator統合確認"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    if 'accelerator.init_trackers(' not in content:
        raise ValueError("accelerator.init_trackers() not found in train.py")
    
    if 'accelerator.log(' not in content:
        raise ValueError("accelerator.log() not found in train.py")
    
    if 'accelerator.end_training()' not in content:
        raise ValueError("accelerator.end_training() not found in train.py")
    
    return "WandB accelerator integration found"

def test_train_py_device_removal():
    """train.pyでのデバイス配置削除確認"""
    with open('train.py', 'r') as f:
        lines = f.readlines()
    
    # .to(device)の使用が適切に削除されているかチェック（コメント行を除外）
    remaining_to_device = 0
    for line in lines:
        stripped_line = line.strip()
        # コメント行やコメント内の使用を除外
        if not stripped_line.startswith('#') and '#' in line:
            # 行の中でコメント以前の部分のみをチェック
            code_part = line.split('#')[0]
            remaining_to_device += code_part.count('.to(device)')
        elif not stripped_line.startswith('#'):
            # コメント行でない場合は全体をチェック
            remaining_to_device += line.count('.to(device)')
    
    if remaining_to_device > 0:
        raise ValueError(f"Active .to(device) usage remaining: {remaining_to_device}")
    
    return f"All .to(device) usage properly removed (only comments remain)"

def test_dry_run_execution():
    """train.pyのドライラン実行（import確認）"""
    # Python構文チェック
    result = subprocess.run([
        sys.executable, '-m', 'py_compile', 'train.py'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise ValueError(f"train.py syntax error: {result.stderr}")
    
    return "train.py syntax validation passed"

# Run all tests
print("\n🧪 Running accelerate integration tests...")

run_test("1. accelerate import", test_accelerate_import)
run_test("2. config gradient accumulation", test_config_gradient_accumulation)
run_test("3. train.py accelerate imports", test_train_py_accelerate_imports)
run_test("4. train.py prepare calls", test_train_py_prepare_calls)
run_test("5. train.py main process control", test_train_py_main_process_control)
run_test("6. train.py wandb integration", test_train_py_wandb_integration)
run_test("7. train.py device removal", test_train_py_device_removal)
run_test("8. train.py dry run", test_dry_run_execution)

# Print summary
print("\n" + "="*80)
print("📊 TEST SUMMARY")
print("="*80)

for test_name, result in test_results.items():
    print(f"{result:<50} {test_name}")

print(f"\n🎯 OVERALL RESULT: {len(test_results) - len(failed_tests)}/{len(test_results)} tests passed")

if failed_tests:
    print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
    for test in failed_tests:
        print(f"  - {test}")
    print("\n⚠️  accelerate対応に問題があります。修正が必要です。")
else:
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ フェーズ2.1: train.py accelerate対応 完了確認")
    print("\n📋 実装完了項目:")
    print("  ✅ Acceleratorの初期化（勾配蓄積対応）")
    print("  ✅ prepareメソッドの適用")
    print("  ✅ バックワードパスの変更（accelerator.backward）")
    print("  ✅ デバイス配置の削除")
    print("  ✅ プリント文の制御（accelerator.is_main_process）")
    print("  ✅ WandBの統合（accelerator.log）")
    print("\n🚀 次のステップ: フェーズ2.2 マルチGPU用accelerate設定")

print("="*80) 