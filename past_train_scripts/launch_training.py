#!/usr/bin/env python3
"""
LISA-Llama4 分散学習ランチャー
A100 80GB × 8GPU用の起動スクリプト

実行方法:
python launch_training.py --exp_name lisa_llama4_exp1 --num_gpus 8
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import socket
import random

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


def detect_network_interface():
    """利用可能なネットワークインターフェースを自動検出（シンプル版）"""
    try:
        # ip routeコマンドでデフォルトルートを確認
        result = subprocess.run(['ip', 'route', 'show', 'default'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # "default via ... dev eth0" の形式から interface を抽出
            for line in result.stdout.split('\n'):
                if 'default via' in line and 'dev' in line:
                    parts = line.split()
                    try:
                        dev_index = parts.index('dev')
                        if dev_index + 1 < len(parts):
                            interface = parts[dev_index + 1]
                            print(f"✅ デフォルトルートインターフェース検出: {interface}")
                            return interface
                    except ValueError:
                        continue
        
        # フォールバック：ifconfigで確認
        result = subprocess.run(['ifconfig'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('eth0:') or line.startswith('eth0 '):
                    print("✅ eth0インターフェース確認")
                    return "eth0"
                elif line.startswith('ib0:') or line.startswith('ib0 '):
                    print("✅ ib0インターフェース確認")
                    return "ib0"
        
        print("⚠️ インターフェース自動検出失敗 - eth0をデフォルト使用")
        return "eth0"
        
    except Exception as e:
        print(f"⚠️ インターフェース検出エラー: {e} - eth0をデフォルト使用")
        return "eth0"


def setup_environment(zero_stage=3):
    """環境変数の設定"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 第3段階：ZeRO-3用CUDAメモリ最適化
    if zero_stage == 3:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:32"
        # ZeRO-3特有の最適化
        os.environ["DEEPSPEED_ZERO3_INIT_FLAG"] = "true"
        os.environ["DEEPSPEED_ZERO_STAGE"] = "3"
        print("✅ ZeRO-3用メモリ最適化を設定")
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True,roundup_power2_divisions:16"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    # DeepSpeedのCPUオフロード最適化
    os.environ["DEEPSPEED_CUDA_MEMORY_LIMIT"] = "0.9"  # GPU使用量を90%に制限
    
    # 2025年NCCL/TCP通信エラー修正：A100 GPU対応（最新Web調査結果反映）
    os.environ["NCCL_DEBUG"] = "INFO"  # 明示的警告メッセージ表示（デバッグ推奨）
    os.environ["NCCL_IB_DISABLE"] = "1"  # InfiniBandを無効化（IP socketsにフォールバック）
    os.environ["NCCL_P2P_DISABLE"] = "1"  # P2P通信を無効化（Web調査での推奨解決策）
    
    # ネットワークインターフェース自動検出
    network_interface = detect_network_interface()
    os.environ["NCCL_SOCKET_IFNAME"] = network_interface  # 自動検出されたインターフェース使用
    
    os.environ["NCCL_IB_TIMEOUT"] = "22"  # InfiniBandタイムアウト値を増加
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"  # CPUヘルパースレッド数
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"  # スレッド毎のソケット数
    os.environ["NCCL_CUMEM_ENABLE"] = "0"  # CUDAメモリプール無効化（トラブルシューティングガイド推奨）
    
    # 2025年PyTorch分散学習の既知問題修正：TCPStore heartbeat monitor競合状態対策
    os.environ["TORCH_NCCL_ABORT_IN_DESTROY_PG"] = "1"  # プロセスグループ破棄時の強制終了
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # 情報レベルデバッグ（DETAIL→INFOで安定化）
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"  # C++スタックトレース表示
    
    # 2025年NCCL heartbeat monitor無効化とタイムアウト延長：根本的解決策（Web調査結果）
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"  # NCCLモニタリング完全無効化（推奨解決策）
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"  # 30分タイムアウト（Web調査推奨値）
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "3"  # 非同期エラーハンドリング強化
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"  # 非ブロッキング待機（最新推奨）
    os.environ["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] = "60000"  # 1分タイムアウトでダンプ
    os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "1"  # 非同期デバッグ情報有効化
    os.environ["TORCH_DISABLE_ADDR2LINE"] = "1"  # アドレス変換無効化（スタックトレース高速化）
    
    # 2025年DeepSpeed ZeRO-3特有の追加対策（最新Web調査結果）
    os.environ["NCCL_DEBUG_SUBSYS"] = "COLL"  # 集約操作の詳細デバッグ（ハング原因特定）
    os.environ["TORCH_DISTRIBUTED_DETAIL_DEBUG"] = "1"  # PyTorch分散処理詳細デバッグ
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU順序を物理バス順に固定
    os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Tree algorithmを無効化（Ring algorithm強制）
    os.environ["NCCL_ALGO"] = "Ring"  # Ring algorithmを明示的に指定
    os.environ["GLOO_SOCKET_IFNAME"] = network_interface  # GlooバックエンドのIF指定
    
    print("✅ NCCL環境変数設定完了: Broken Pipe エラー対策（Web調査最新解決策適用）")
    print("✅ PyTorch分散学習の競合状態対策完了（TCPStore heartbeat monitor対応）")
    print("✅ NCCL heartbeat monitor完全無効化とタイムアウト最適化完了")
    print("✅ DeepSpeed ZeRO-3追加最適化完了（Ring algorithm強制・GPU順序固定）")
    if zero_stage == 3:
        print("✅ 第3段階：完全ZeRO-3移行完了（103Bモデル対応）")
    else:
        print("✅ 第1段階：バッチサイズ削減 + ZeRO-2最適化完了")
    
    # HuggingFaceキャッシュディレクトリを自動設定
    setup_hf_cache_dirs()
    
    # HuggingFaceトークンを確実に設定
    from utils.hf_auth import get_hf_token
    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print(f"✅ HuggingFaceトークンを環境変数に設定: hf_***{token[-8:]}")
    else:
        print("⚠️ HuggingFaceトークンが見つかりません")


def setup_hf_cache_dirs():
    """HuggingFaceキャッシュディレクトリを自動設定・作成"""
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
            os.makedirs(candidate, exist_ok=True)
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
        print(f"✅ HF_HOME設定: {hf_cache_dir}")
    else:
        print("⚠️ HF_HOME設定に失敗")
    
    if transformers_cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache_dir
        print(f"✅ TRANSFORMERS_CACHE設定: {transformers_cache_dir}")
    else:
        print("⚠️ TRANSFORMERS_CACHE設定に失敗")
    
    # HuggingFace Hub設定も追加
    if hf_cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        print(f"✅ HUGGINGFACE_HUB_CACHE設定: {hf_cache_dir}")
    
    print("=== キャッシュディレクトリ設定完了 ===")


def main():
    parser = argparse.ArgumentParser(description="LISA-Llama4 分散学習ランチャー")
    
    # 基本設定
    parser.add_argument("--exp_name", type=str, required=True,
                       help="実験名")
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="使用するGPU数")
    
    # 学習設定のオーバーライド
    parser.add_argument("--batch_size", type=int, default=1,
                       help="バッチサイズ（GPU毎）")
    parser.add_argument("--grad_accumulation_steps", type=int, default=8,
                       help="勾配蓄積ステップ数")
    parser.add_argument("--epochs", type=int, default=10,
                       help="エポック数")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="学習率")
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRAランク")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRAアルファ")
    
    # その他の設定
    parser.add_argument("--resume", type=str, default="",
                       help="レジュームするチェックポイントパス")
    parser.add_argument("--debug", action="store_true",
                       help="デバッグモード（1GPU、少ないステップ）")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                       help="エポックあたりのステップ数（短時間テスト用）")
    parser.add_argument("--dataset", type=str, default=None,
                       help="使用するデータセット（例: 'reason_seg'）")
    
    # 第3段階：完全ZeRO-3移行設定
    parser.add_argument("--zero_stage", type=int, default=3, choices=[2, 3],
                       help="DeepSpeed ZeRO stage (2: ZeRO-2, 3: ZeRO-3)")
    parser.add_argument("--cpu_offload_params", action="store_true", default=True,
                       help="ZeRO-3でパラメータをCPUオフロード（103Bモデル推奨）")
    parser.add_argument("--conservative_mode", action="store_true",
                       help="保守的DeepSpeed設定を使用（NCCL通信エラー対策）")
    
    args = parser.parse_args()
    
    # HuggingFace自動ログイン
    print("=== HuggingFace認証確認 ===")
    if not ensure_hf_login():
        print("❌ HuggingFace認証に失敗しました。処理を中断します。")
        print("トークンを確認して再実行してください。")
        sys.exit(1)
    
    # デバッグモードの調整
    if args.debug:
        args.num_gpus = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("🔍 デバッグモード: 1GPU使用")
    
    # 環境設定
    setup_environment(args.zero_stage)
    
    # タイムスタンプとログディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # マスターポート
    master_port = find_free_port()
    
    print("=" * 80)
    print("LISA-Llama4 分散学習")
    print("=" * 80)
    print(f"実験名: {args.exp_name}")
    print(f"タイムスタンプ: {timestamp}")
    print(f"GPU数: {args.num_gpus}")
    print(f"マスターポート: {master_port}")
    print(f"ログディレクトリ: {log_dir}")
    print(f"ZeRO Stage: {args.zero_stage}")
    if args.cpu_offload_params:
        print("パラメータCPUオフロード: 有効")
    print("=" * 80)
    
    # DeepSpeedコマンド構築
    cmd = [
        "deepspeed",
        f"--num_gpus={args.num_gpus}",
        f"--master_port={master_port}",
        "train_llama4_lisa_deepspeed.py",
        "--exp_name", args.exp_name,
        "--batch_size", str(args.batch_size),
        "--grad_accumulation_steps", str(args.grad_accumulation_steps),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--gradient_checkpointing",
        "--precision", "bf16",
        "--workers", "4",
        "--print_freq", "10",
        "--save_freq", "1",
        "--auto_resume"
    ]
    
    # レジューム指定がある場合
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    # デバッグモードの場合
    if args.debug:
        cmd.extend(["--steps_per_epoch", "10"])
    
    # カスタムステップ数指定
    if args.steps_per_epoch is not None:
        cmd.extend(["--steps_per_epoch", str(args.steps_per_epoch)])
    
    # カスタムデータセット指定
    if args.dataset is not None:
        cmd.extend(["--dataset", args.dataset])
    
    # 第2段階：ZeRO Stage指定
    cmd.extend(["--zero_stage", str(args.zero_stage)])
    if args.cpu_offload_params:
        cmd.append("--cpu_offload_params")
    
    # 保守的モード設定
    if args.conservative_mode:
        cmd.extend(["--deepspeed_config", "ds_config_llama4_conservative.json"])
        print("⚠️ 保守的モード: NCCL通信エラー対策設定を適用")
    
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
            print("✅ 学習が正常に完了しました!")
            print(f"ログ: {log_file}")
            print(f"チェックポイント: ./checkpoints/{args.exp_name}")
            print(f"TensorBoard: tensorboard --logdir=./runs/{args.exp_name}")
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