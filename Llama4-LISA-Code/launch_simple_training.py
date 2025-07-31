#!/usr/bin/env python3
"""
LISA-Llama4 Model Parallelism学習ランチャー（シングルプロセス版）
成功したoverfit_llama4_lisa_batch.pyの手法を使用（Web調査ベース）

注意: これは分散学習ではありません。Model Parallelismのみです。
大規模データでの学習にはスループットの限界があります。

実行方法:
# 最小時間テスト
python launch_simple_training.py --exp_name quick_test --steps_per_epoch 10 --epochs 1

# 通常学習（小規模データセットのみ）
python launch_simple_training.py --exp_name lisa_training --epochs 5
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import socket

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.hf_auth import ensure_hf_login

def find_free_port():
    """利用可能なポートを探す"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_environment():
    """環境変数の設定（成功したoverfitパターンベース）"""
    # GPU表示順序を固定（Web調査：Model Parallelism安定化）
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    
    # メモリ最適化（成功したoverfit設定）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # HuggingFaceキャッシュ設定
    setup_hf_cache_dirs()
    
    print("✅ Model Parallelism環境設定完了（成功パターンベース）")

def setup_hf_cache_dirs():
    """HuggingFaceキャッシュディレクトリを自動設定・作成（分散学習競合対策）"""
    print("=== HuggingFaceキャッシュディレクトリ設定 ===")
    
    # 候補ディレクトリ（優先順位順）
    cache_candidates = [
        "/lambda/nfs/lisa-gemma-project-fs/cache/huggingface",
        os.path.expanduser("~/.cache/huggingface"),
        "/tmp/huggingface_cache"
    ]
    
    transformers_candidates = [
        "/lambda/nfs/lisa-gemma-project-fs/cache/transformers",
        os.path.expanduser("~/.cache/transformers"),
        "/tmp/transformers_cache"
    ]
    
    # 書き込み可能なディレクトリを探す
    hf_cache_dir = None
    for candidate in cache_candidates:
        try:
            # ディレクトリ作成（競合対策でexist_ok=True）
            os.makedirs(candidate, exist_ok=True)
            
            # 追加：サブディレクトリも事前作成（分散学習競合対策）
            os.makedirs(os.path.join(candidate, "models"), exist_ok=True)
            os.makedirs(os.path.join(candidate, "datasets"), exist_ok=True)
            
            # 書き込みテスト
            test_file = os.path.join(candidate, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            hf_cache_dir = candidate
            break
        except (OSError, PermissionError):
            continue
    
    transformers_cache_dir = None
    for candidate in transformers_candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            # 書き込みテスト
            test_file = os.path.join(candidate, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            transformers_cache_dir = candidate
            break
        except (OSError, PermissionError):
            continue
    
    # 環境変数に設定
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        print(f"✅ HF_HOME設定: {hf_cache_dir}")
    else:
        print("⚠️ HF_HOME設定に失敗")
    
    if transformers_cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache_dir
        print(f"✅ TRANSFORMERS_CACHE設定: {transformers_cache_dir}")
    else:
        print("⚠️ TRANSFORMERS_CACHE設定に失敗")
    
    # 分散学習特有の設定
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir if hf_cache_dir else os.path.expanduser("~/.cache/huggingface/datasets")
    os.environ["TOKENIZERS_CACHE"] = hf_cache_dir if hf_cache_dir else os.path.expanduser("~/.cache/huggingface/tokenizers")
    
    print("=== キャッシュディレクトリ設定完了（分散学習最適化済み） ===")

def main():
    parser = argparse.ArgumentParser(description="LISA-Llama4 Model Parallelism学習ランチャー（成功パターン完全移植）")
    
    # 基本設定
    parser.add_argument("--exp_name", type=str, required=True, help="実験名")
    # Model Parallelismでは固定で8GPU使用（成功パターン）
    parser.add_argument("--num_gpus", type=int, default=8, help="GPU数（Model Parallelism用、固定）")
    
    # 学習設定
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ（GPU毎）")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学習率")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="エポックあたりのステップ数（短時間テスト用）")
    parser.add_argument("--workers", type=int, default=4, help="データローダーワーカー数")
    parser.add_argument("--print_freq", type=int, default=10, help="ログ出力頻度")
    
    args = parser.parse_args()
    
    # HuggingFace自動ログイン
    print("=== HuggingFace認証確認 ===")
    if not ensure_hf_login():
        print("❌ HuggingFace認証に失敗しました。処理を中断します。")
        sys.exit(1)
    
    # 環境設定
    setup_environment()
    
    # タイムスタンプとログディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # マスターポート
    master_port = find_free_port()
    
    print("=" * 80)
    print("LISA-Llama4 Model Parallelism学習（Web調査ベース）")
    print("=" * 80)
    print(f"実験名: {args.exp_name}")
    print(f"タイムスタンプ: {timestamp}")
    print(f"GPU数: {args.num_gpus} (Model Parallelism)")
    print(f"ログディレクトリ: {log_dir}")
    print(f"成功パターン: overfit_llama4_lisa_batch.pyベース")
    if args.steps_per_epoch:
        print(f"ステップ制限: {args.steps_per_epoch}/エポック (短時間テスト)")
    print("=" * 80)
    
    # 成功したModel Parallelismコマンド構築（シングルプロセス）
    cmd = [
        "python",
        "train_llama4_lisa_single_process.py",
        "--exp_name", args.exp_name,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--workers", str(args.workers),
        "--print_freq", str(args.print_freq)
    ]
    
    # ステップ制限指定
    if args.steps_per_epoch is not None:
        cmd.extend(["--steps_per_epoch", str(args.steps_per_epoch)])
    
    # ログファイル
    log_file = os.path.join(log_dir, "train.log")
    
    print(f"\n実行コマンド:")
    print(" ".join(cmd))
    print(f"\nログ出力: {log_file}")
    print("\n学習を開始します...\n")
    
    # 実行
    try:
        with open(log_file, "w") as f:
            # リアルタイムで出力を表示しながらファイルにも保存
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
        if process.returncode == 0:
            print("\n" + "=" * 80)
            print("✅ LISA-Llama4 Model Parallelism学習が正常に完了しました!")
            print(f"ログ: {log_file}")
            print(f"チェックポイント: ./checkpoints/{args.exp_name}")
            print(f"TensorBoard: tensorboard --logdir=./runs/{args.exp_name}_{timestamp}")
            print("🎆 完全移植成功: overfit_llama4_lisa_batch.py成功パターン100%適用済み")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"❌ エラーが発生しました (終了コード: {process.returncode})")
            print(f"詳細はログを確認してください: {log_file}")
            print("=" * 80)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 学習が中断されました")
        print(f"ログ: {log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()