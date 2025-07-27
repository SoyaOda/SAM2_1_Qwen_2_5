#!/usr/bin/env python3
"""
DeepSpeed環境診断・修復スクリプト
ローカル環境のCUDA/DeepSpeed問題を特定し、解決策を提示
"""

import os
import sys
import subprocess
import importlib
import torch
from packaging import version

def run_command(cmd, capture_output=True):
    """コマンド実行のヘルパー"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, 
            text=True, timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def check_cuda_environment():
    """CUDA環境の診断"""
    print("=== CUDA環境診断 ===")
    
    # PyTorch CUDA情報
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"PyTorch CUDA対応: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA バージョン: {torch.version.cuda}")
        print(f"利用可能GPU数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # システムCUDA情報
    success, stdout, stderr = run_command("nvcc --version")
    if success:
        cuda_version = stdout.split("release ")[-1].split(",")[0] if "release" in stdout else "不明"
        print(f"システム CUDA バージョン: {cuda_version}")
    else:
        print("システム CUDA: インストールされていない")
    
    # CUDA Driver情報
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        lines = stdout.split('\n')
        driver_line = [line for line in lines if "Driver Version" in line]
        if driver_line:
            driver_version = driver_line[0].split("Driver Version: ")[1].split()[0]
            cuda_driver = driver_line[0].split("CUDA Version: ")[1].split()[0]
            print(f"NVIDIA Driver: {driver_version}")
            print(f"CUDA Driver: {cuda_driver}")
    else:
        print("nvidia-smi: 実行できません")

def check_deepspeed_environment():
    """DeepSpeed環境の診断"""
    print("\n=== DeepSpeed環境診断 ===")
    
    try:
        import deepspeed
        print(f"DeepSpeed バージョン: {deepspeed.__version__}")
        
        # DeepSpeed環境チェック
        success, stdout, stderr = run_command("ds_report")
        if success:
            print("DeepSpeedレポート:")
            print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
        else:
            print(f"ds_reportエラー: {stderr}")
            
    except ImportError:
        print("DeepSpeed: インストールされていません")
    
    # MPI チェック
    try:
        import mpi4py
        print(f"mpi4py バージョン: {mpi4py.__version__}")
    except ImportError:
        print("mpi4py: インストールされていません")

def analyze_compatibility():
    """互換性問題の分析"""
    print("\n=== 互換性分析 ===")
    
    issues = []
    solutions = []
    
    # PyTorch CUDA バージョン確認
    torch_cuda = torch.version.cuda if torch.cuda.is_available() else None
    
    success, stdout, stderr = run_command("nvcc --version")
    if success and torch_cuda:
        system_cuda = stdout.split("release ")[-1].split(",")[0] if "release" in stdout else None
        
        if system_cuda and torch_cuda != system_cuda:
            issues.append(f"CUDA バージョン不整合: PyTorch={torch_cuda}, システム={system_cuda}")
            solutions.append("PyTorchの再インストール（CUDA 12.9対応版）")
    
    # DeepSpeed CUDA拡張問題
    try:
        import deepspeed
        success, stdout, stderr = run_command("python -c 'import deepspeed; deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()'")
        if not success:
            issues.append("DeepSpeed CUDA拡張がビルドできません")
            solutions.append("DeepSpeedの再インストール or CPUオフロード無効化")
    except ImportError:
        pass
    
    # GLIBC問題
    success, stdout, stderr = run_command("ldd --version")
    if success:
        glibc_version = stdout.split('\n')[0].split()[-1]
        print(f"GLIBC バージョン: {glibc_version}")
        
        if version.parse(glibc_version) < version.parse("2.32"):
            issues.append(f"GLIBC バージョンが古い: {glibc_version}")
            solutions.append("システムのアップデート")
    
    if issues:
        print("🚨 発見された問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n💡 推奨解決策:")
        for i, solution in enumerate(solutions, 1):
            print(f"  {i}. {solution}")
    else:
        print("✅ 重大な互換性問題は見つかりませんでした")

def suggest_workarounds():
    """回避策の提案"""
    print("\n=== ローカル環境での回避策 ===")
    
    print("1. 🎯 PyTorch DDP使用（推奨）")
    print("   - train_ddp.py を使用")
    print("   - DeepSpeedと同等の機能をDDPで実現")
    print("   - クラウド移行時にDeepSpeedに簡単切り替え")
    print("   - 実行例: python train_ddp.py --batch_size 8 --world_size 1")
    
    print("\n2. 🔧 DeepSpeed CPU-only実行")
    print("   - CPUオフロードを無効化")
    print("   - CUDA拡張を回避")
    print("   - ds_config_local.jsonを使用")
    
    print("\n3. ⚡ CUDA環境修復")
    print("   - PyTorch CUDA 12.9版再インストール:")
    print("     pip uninstall torch torchvision torchaudio")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129")
    print("   - DeepSpeed再インストール:")
    print("     pip uninstall deepspeed")
    print("     DS_BUILD_OPS=1 pip install deepspeed")
    
    print("\n4. 🐳 Docker環境使用")
    print("   - 統一されたCUDA環境")
    print("   - 依存関係の問題を回避")

def recommend_next_steps():
    """次のステップの推奨"""
    print("\n=== 推奨実装手順 ===")
    
    print("📋 Phase 1: ローカル基盤構築 (今すぐ)")
    print("  1. train_ddp.py でDDP学習を確立")
    print("  2. 分散学習のロジックを完全実装")
    print("  3. デュアルストリーム処理の安定化")
    
    print("\n📋 Phase 2: DeepSpeed準備")
    print("  1. ds_config_*.json の環境別設定完成")
    print("  2. CUDA環境修復（オプション）")
    print("  3. DeepSpeed動作確認")
    
    print("\n📋 Phase 3: クラウド移行")
    print("  1. train_ddp.py → train_deepspeed.py 切り替え")
    print("  2. CPUオフロード有効化")
    print("  3. 大規模学習実行")
    
    print("\n🎯 今日の目標:")
    print("  - train_ddp.py の動作確認")
    print("  - 1エポック分の学習成功")
    print("  - DeepSpeedインターフェース習得")

def main():
    """メイン診断フロー"""
    print("🔍 LISA-Gemma3 DeepSpeed環境診断")
    print("=" * 50)
    
    check_cuda_environment()
    check_deepspeed_environment()
    analyze_compatibility()
    suggest_workarounds()
    recommend_next_steps()
    
    print("\n" + "=" * 50)
    print("診断完了。上記の分析と推奨事項をご確認ください。")

if __name__ == "__main__":
    main() 