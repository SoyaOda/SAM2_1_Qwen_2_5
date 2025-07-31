#!/usr/bin/env python3
"""
LISA-Gemma3 DeepSpeed学習起動スクリプト
仕様書第6章.1「起動コマンド」に従った実装
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_linux import *


def check_environment():
    """環境設定の確認"""
    print("🔍 環境設定の確認中...")
    
    # 必要なパッケージの確認
    required_packages = [
        'torch', 'transformers', 'deepspeed', 'peft', 'accelerate'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}: インストール済み")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}: 未インストール")
    
    if missing_packages:
        print(f"\n❌ 以下のパッケージをインストールしてください:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        return False
    
    # 設定ファイルの確認
    config_files = [
        'config_linux.py',
        'ds_config.json',
        'train_deepspeed.py'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  ✅ {config_file}: 存在")
        else:
            print(f"  ❌ {config_file}: 不存在")
            return False
    
    # データセットパスの確認
    if os.path.exists(DATASET_BASE_DIR):
        print(f"  ✅ データセットディレクトリ: {DATASET_BASE_DIR}")
    else:
        print(f"  ⚠️ データセットディレクトリが見つかりません: {DATASET_BASE_DIR}")
        print("  ダミーデータでの学習になります")
    
    # SAMチェックポイントの確認
    if os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"  ✅ SAMチェックポイント: {SAM_CHECKPOINT_PATH}")
    else:
        print(f"  ⚠️ SAMチェックポイントが見つかりません: {SAM_CHECKPOINT_PATH}")
        print("  SAMなしでの学習になります")
    
    print("✅ 環境確認完了")
    return True


def get_gpu_info():
    """GPU情報の取得"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory': int(parts[2])
                    })
            return gpu_info
        else:
            print("⚠️ nvidia-smiコマンドが使用できません")
            return []
    except FileNotFoundError:
        print("⚠️ nvidia-smiが見つかりません")
        return []


def generate_deepspeed_command(args):
    """DeepSpeedコマンドの生成"""
    gpu_info = get_gpu_info()
    
    if gpu_info:
        print(f"🖥️ 検出されたGPU: {len(gpu_info)}個")
        for gpu in gpu_info:
            print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory']}MB)")
        
        # GPU指定の生成
        if args.gpus:
            gpu_indices = args.gpus
        else:
            gpu_indices = ",".join([str(gpu['index']) for gpu in gpu_info])
        
        include_arg = f"localhost:{gpu_indices}"
    else:
        print("⚠️ GPUが検出されませんでした。CPUモードで実行します")
        include_arg = "localhost:0"  # 最低限の設定
    
    # DeepSpeedコマンドの構築
    cmd = [
        "deepspeed",
        f"--include={include_arg}",
        "train_deepspeed.py",
        f"--deepspeed_config={args.deepspeed_config}",
        f"--exp_name={args.exp_name}",
        f"--batch_size={args.batch_size}",
        f"--grad_accumulation_steps={args.grad_accumulation_steps}",
        f"--lr={args.lr}",
        f"--epochs={args.epochs}",
        f"--steps_per_epoch={args.steps_per_epoch}",
    ]
    
    # オプション引数の追加
    if args.use_sam:
        cmd.append("--use_sam")
    
    if args.precision:
        cmd.append(f"--precision={args.precision}")
    
    if args.log_dir:
        cmd.append(f"--log_dir={args.log_dir}")
    
    if args.save_dir:
        cmd.append(f"--save_dir={args.save_dir}")
    
    if args.resume_from:
        cmd.append(f"--resume_from={args.resume_from}")
    
    return cmd


def run_verification_first():
    """学習前の検証実行"""
    print("🔍 学習前の検証を実行中...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_verification_protocol.py"
        ], capture_output=True, text=True, timeout=300)  # 5分のタイムアウト
        
        print("📋 検証結果:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ 検証中の警告:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 検証完了")
            return True
        else:
            print("❌ 検証失敗")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 検証がタイムアウトしました")
        return False
    except Exception as e:
        print(f"❌ 検証実行エラー: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LISA-Gemma3 DeepSpeed学習起動スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必須引数
    parser.add_argument("--exp_name", type=str, required=True,
                       help="実験名（ログとモデル保存に使用）")
    
    # DeepSpeed設定
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json",
                       help="DeepSpeed設定ファイル")
    parser.add_argument("--gpus", type=str, default=None,
                       help="使用するGPUインデックス（例: '0,1,2,3'）")
    
    # 学習パラメータ
    parser.add_argument("--batch_size", type=int, default=16,
                       help="グローバルバッチサイズ")
    parser.add_argument("--grad_accumulation_steps", type=int, default=2,
                       help="勾配累積ステップ数")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                       help="学習率")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="エポック数")
    parser.add_argument("--steps_per_epoch", type=int, default=STEPS_PER_EPOCH,
                       help="エポックあたりのステップ数")
    
    # モデル設定
    parser.add_argument("--use_sam", action="store_true",
                       help="SAMを使用するかどうか")
    parser.add_argument("--precision", type=str, default="bf16",
                       choices=["fp16", "bf16", "fp32"],
                       help="学習精度")
    
    # ログとモデル保存
    parser.add_argument("--log_dir", type=str, default=LOG_BASE_DIR,
                       help="ログ保存ディレクトリ")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="モデル保存ディレクトリ")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="学習再開用のチェックポイントパス")
    
    # 実行オプション
    parser.add_argument("--skip_verification", action="store_true",
                       help="学習前の検証をスキップ")
    parser.add_argument("--dry_run", action="store_true",
                       help="コマンドを表示するだけで実行しない")
    
    args = parser.parse_args()
    
    print("🚀 LISA-Gemma3 DeepSpeed学習起動スクリプト")
    print("=" * 60)
    print(f"実験名: {args.exp_name}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"学習率: {args.lr}")
    print(f"エポック数: {args.epochs}")
    print(f"SAM使用: {'Yes' if args.use_sam else 'No'}")
    print("=" * 60)
    
    # 環境確認
    if not check_environment():
        print("❌ 環境確認に失敗しました")
        sys.exit(1)
    
    # 学習前検証
    if not args.skip_verification:
        print("\n🔍 学習前検証を実行します...")
        if not run_verification_first():
            response = input("検証に問題がありました。学習を続行しますか？ (y/N): ")
            if response.lower() != 'y':
                print("学習を中止しました")
                sys.exit(1)
    
    # DeepSpeedコマンド生成
    cmd = generate_deepspeed_command(args)
    
    print(f"\n🚀 実行コマンド:")
    print(" ".join(cmd))
    
    if args.dry_run:
        print("\n🔍 ドライランモードのため、実際の実行はスキップします")
        return
    
    # 実行確認
    response = input(f"\n学習を開始しますか？ (y/N): ")
    if response.lower() != 'y':
        print("学習を中止しました")
        return
    
    # 学習実行
    print(f"\n🎯 学習開始: {args.exp_name}")
    print("=" * 60)
    
    try:
        # 環境変数の設定
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # DeepSpeed実行
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # リアルタイムでログを表示
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ 学習が正常に完了しました")
        else:
            print(f"\n❌ 学習が失敗しました (終了コード: {process.returncode})")
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって学習が中断されました")
        process.terminate()
    except Exception as e:
        print(f"\n❌ 学習実行エラー: {e}")


if __name__ == "__main__":
    main() 