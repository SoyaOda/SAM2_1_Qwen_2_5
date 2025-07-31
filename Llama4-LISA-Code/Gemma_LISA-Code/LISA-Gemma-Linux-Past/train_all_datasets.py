#!/usr/bin/env python3
"""
Gemma3-LISA 全データセット順次学習スクリプト

このスクリプトは、利用可能なすべてのデータセットに対して
順番にデバッグモードで学習を実行します。
各データセットタイプ（sem_seg, refer_seg, vqa, reason_seg）と
その中の具体的なデータセット（ade20k, cocostuff, refcoco等）を
それぞれデバッグモードで学習することができます。

オリジナルのLISAの実装を参考にしています。
"""

import argparse
import os
import sys
import time
import logging
import pickle
import subprocess
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# ロギング用のグローバル変数
LOG_DIR = None
MAIN_LOG_FILE = None
DATASET_LOGS = {}
EXECUTION_SUMMARY = {
    "start_time": None,
    "end_time": None,
    "datasets_processed": [],
    "success_count": 0,
    "failure_count": 0,
    "skipped_count": 0,
    "total_duration": 0
}

def setup_logging(args):
    """
    ロギング設定をセットアップします
    
    Args:
        args: コマンドライン引数
        
    Returns:
        logger: ロガーオブジェクト
    """
    global LOG_DIR, MAIN_LOG_FILE
    
    # タイムスタンプつきのログディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"training_logs_{timestamp}"
    LOG_DIR = os.path.join(args.log_base_dir, log_dir_name)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # メインログファイルパスを設定
    MAIN_LOG_FILE = os.path.join(LOG_DIR, "all_datasets_training.log")
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(MAIN_LOG_FILE)
        ]
    )
    
    # 実行設定情報をログディレクトリに保存
    with open(os.path.join(LOG_DIR, "execution_settings.json"), "w", encoding="utf-8") as f:
        settings = {
            "timestamp": timestamp,
            "debug_mode": args.debug,
            "debug_samples": args.debug_samples,
            "dataset_dir": args.dataset_dir,
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "run_sem_seg": args.run_sem_seg,
            "run_refer_seg": args.run_refer_seg,
            "run_vqa": args.run_vqa,
            "run_reason_seg": args.run_reason_seg,
            "dry_run": args.dry_run
        }
        json.dump(settings, f, indent=2, ensure_ascii=False)
    
    # 実行サマリーの初期化
    global EXECUTION_SUMMARY
    EXECUTION_SUMMARY["start_time"] = timestamp
    
    return logging.getLogger(__name__)

def get_dataset_logger(dataset_type, dataset):
    """
    データセット固有のロガーを取得します
    
    Args:
        dataset_type: データセットタイプ（sem_seg, refer_seg, vqa, reason_seg）
        dataset: データセット名
        
    Returns:
        logger: データセット固有のロガー
    """
    global LOG_DIR, DATASET_LOGS
    
    dataset_key = f"{dataset_type}_{dataset}"
    if dataset_key in DATASET_LOGS:
        return DATASET_LOGS[dataset_key]
    
    # データセット固有のログファイルを設定
    log_file = os.path.join(LOG_DIR, f"{dataset_type}_{dataset}.log")
    
    # データセット固有のロガーを作成
    dataset_logger = logging.getLogger(f"{dataset_type}.{dataset}")
    dataset_logger.setLevel(logging.INFO)
    
    # ハンドラーを設定
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    dataset_logger.addHandler(file_handler)
    
    # コンソール出力も維持
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    dataset_logger.addHandler(console_handler)
    
    # ロガーをキャッシュ
    DATASET_LOGS[dataset_key] = dataset_logger
    
    return dataset_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma3-LISA 全データセット順次学習")
    # 基本設定
    parser.add_argument("--batch_size", type=int, default=2, help="バッチサイズ")
    parser.add_argument("--grad_accumulation_steps", type=int, default=8, help="勾配累積ステップ数")
    parser.add_argument("--lr", type=float, default=0.0003, help="学習率")
    parser.add_argument("--epochs", type=int, default=1, help="各データセットのエポック数")
    parser.add_argument("--steps_per_epoch", type=int, default=5, help="1エポックあたりのステップ数")
    
    # デバッグ関連
    parser.add_argument("--debug", action="store_true", default=True, help="デバッグモードを有効にする")
    parser.add_argument("--debug_samples", type=int, default=5, help="各データセットから使用するサンプル数")
    parser.add_argument("--use_cached_data", action="store_true", help="キャッシュされたデータを使用する")
    parser.add_argument("--save_cached_data", action="store_true", help="データをキャッシュに保存する")
    parser.add_argument("--dry_run", action="store_true", help="コマンドを実行せず、パス検証のみ行う")
    
    # モデル設定
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="4bitモデル量子化を使用")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--vision_tower", type=str, default="sigLIP", help="ビジョンエンコーダ（CLIP or sigLIP）")
    
    # データセット設定
    parser.add_argument("--dataset_dir", type=str, default="/mnt/h/download/LISA-dataset/dataset", help="データセットディレクトリ")
    
    # データセットタイプの選択
    dataset_group = parser.add_argument_group('データセットタイプの選択')
    dataset_group.add_argument("--sem_seg", dest="run_sem_seg", action="store_true", help="セマンティックセグメンテーションデータセットを実行")
    dataset_group.add_argument("--no-sem_seg", dest="run_sem_seg", action="store_false", help="セマンティックセグメンテーションデータセットを実行しない")
    dataset_group.add_argument("--refer_seg", dest="run_refer_seg", action="store_true", help="参照セグメンテーションデータセットを実行")
    dataset_group.add_argument("--no-refer_seg", dest="run_refer_seg", action="store_false", help="参照セグメンテーションデータセットを実行しない")
    dataset_group.add_argument("--vqa", dest="run_vqa", action="store_true", help="VQAデータセットを実行")
    dataset_group.add_argument("--no-vqa", dest="run_vqa", action="store_false", help="VQAデータセットを実行しない")
    dataset_group.add_argument("--reason_seg", dest="run_reason_seg", action="store_true", help="理由付きセグメンテーションデータセットを実行")
    dataset_group.add_argument("--no-reason_seg", dest="run_reason_seg", action="store_false", help="理由付きセグメンテーションデータセットを実行しない")
    
    # デフォルト値を設定
    parser.set_defaults(run_sem_seg=True, run_refer_seg=True, run_vqa=True, run_reason_seg=True)
    
    # 再開位置
    parser.add_argument("--continue_from_dataset_type", type=str, default=None, 
                        choices=["sem_seg", "refer_seg", "vqa", "reason_seg"], 
                        help="特定のデータセットタイプから再開")
    parser.add_argument("--continue_from_dataset", type=str, default=None, 
                        help="特定のデータセット名から再開（例: cocostuff）")
    
    # 出力設定
    parser.add_argument("--log_base_dir", type=str, default="./runs", help="ログ出力ベースディレクトリ")
    
    return parser.parse_args()

def get_dataset_list(dataset_type):
    """
    データセットタイプに基づいて、利用可能なデータセットのリストを返します
    
    Args:
        dataset_type: データセットタイプ（sem_seg, refer_seg, vqa, reason_seg）
        
    Returns:
        list: 利用可能なデータセットのリスト
    """
    # 実際の利用可能なデータセットを基に更新
    if dataset_type == "sem_seg":
        # 確認されたディレクトリに基づいて利用可能なデータセットを設定
        return ["ade20k", "cocostuff", "mapillary"]
    elif dataset_type == "refer_seg":
        # refer_segディレクトリには画像はあるが、refclef, refcocoなどのサブディレクトリが確認できなかったので注意
        return ["refcoco", "refcoco+", "refcocog"]
    elif dataset_type == "vqa":
        # llava_datasetディレクトリが存在することを確認
        return ["llava_instruct_150k"]
    elif dataset_type == "reason_seg":
        # ReasonSegデータセットの存在は確認できなかったが、オリジナルのLISAに合わせて保持
        return ["ReasonSeg"]
    else:
        return []

def create_dataset_specific_args(dataset_type, dataset):
    """
    データセットタイプとデータセット名に基づいて、特定のコマンドライン引数を作成します
    
    Args:
        dataset_type: データセットタイプ（sem_seg, refer_seg, vqa, reason_seg）
        dataset: データセット名（ade20k, refcoco等）
        
    Returns:
        list: コマンドライン引数のリスト
    """
    args = []
    
    # データセットタイプに応じた引数を追加
    args.extend(["--dataset", dataset_type])
    
    # サンプルレートを設定（単一データセット用に1）
    args.extend(["--sample_rates", "1"])
    
    # データセット固有の引数
    if dataset_type == "sem_seg":
        args.extend(["--sem_seg_data", dataset])
    elif dataset_type == "refer_seg":
        args.extend(["--refer_seg_data", dataset])
    elif dataset_type == "vqa":
        args.extend(["--vqa_data", dataset])
        # llava_instruct_150kの場合、vqa_base_dirディレクトリを指定
        args.extend(["--vqa_base_dir", "llava_dataset"])
    elif dataset_type == "reason_seg":
        # ReasonSegの場合、正しいパスを指定
        # reason_seg/ReasonSeg/trainディレクトリにデータがあるので、そのパスを設定
        args.extend(["--reason_seg_data", f"{dataset}|train"])
        # ReasonSegのデータディレクトリを追加指定
        args.extend(["--reason_seg_base_dir", "reason_seg"])
    
    return args

def validate_dataset_path(dataset_dir, dataset_type, dataset):
    """
    データセットのパスが存在するか検証します
    
    Args:
        dataset_dir: ベースデータセットディレクトリ
        dataset_type: データセットタイプ
        dataset: データセット名
        
    Returns:
        bool: パスが存在するかどうか
    """
    # データセットタイプに応じたパスをチェック
    if dataset_type == "sem_seg":
        # sem_segデータセットは直接ベースディレクトリにあるので、そのパスをチェック
        path = os.path.join(dataset_dir, dataset)
    elif dataset_type == "refer_seg":
        # refer_segデータセットの構造はimages/mscoco/imagesとimages/saiapr_tc-12なので、
        # パス構造が正しいかチェック
        images_dir = os.path.join(dataset_dir, "refer_seg", "images")
        mscoco_dir = os.path.join(images_dir, "mscoco", "images")
        saiapr_dir = os.path.join(images_dir, "saiapr_tc-12")
        
        # 必要なディレクトリが存在するかチェック
        all_exist = os.path.exists(images_dir) and (os.path.exists(mscoco_dir) or os.path.exists(saiapr_dir))
        return all_exist
    elif dataset_type == "vqa":
        # VQAデータセットはllava_dataset内にあるのでそのパスをチェック
        path = os.path.join(dataset_dir, "llava_dataset")
    elif dataset_type == "reason_seg":
        # ReasonSegデータセットのパスを修正
        path = os.path.join(dataset_dir, "reason_seg", "ReasonSeg")
        # トレーニングフォルダが存在するかも確認
        train_path = os.path.join(path, "train")
        if os.path.exists(path) and os.path.exists(train_path):
            return True
        # デバッグ情報
        if not os.path.exists(path):
            logger.debug(f"ReasonSegディレクトリが見つかりません: {path}")
        elif not os.path.exists(train_path):
            logger.debug(f"ReasonSegのtrainディレクトリが見つかりません: {train_path}")
        return False
    else:
        return False
    
    # 指定されたパスが存在するかチェック
    exists = os.path.exists(path) if 'path' in locals() else False
    if not exists:
        logger.debug(f"パスが見つかりません: {path if 'path' in locals() else '不明'}")
    return exists

def run_training_for_dataset(args, dataset_type, dataset):
    """
    特定のデータセットに対してトレーニングを実行します
    
    Args:
        args: コマンドライン引数
        dataset_type: データセットタイプ（sem_seg, refer_seg, vqa, reason_seg）
        dataset: データセット名（ade20k, refcoco等）
        
    Returns:
        bool: 成功したかどうか
    """
    global EXECUTION_SUMMARY
    
    # データセット固有のロガーを取得
    dataset_logger = get_dataset_logger(dataset_type, dataset)
    dataset_logger.info(f"データセット '{dataset_type}/{dataset}' のトレーニングを開始します...")
    
    # 実行結果を保存する辞書
    result_summary = {
        "dataset_type": dataset_type,
        "dataset": dataset,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "skipped",  # 初期状態
        "duration": 0,
        "error": None
    }
    
    # データセットパスの検証
    if not validate_dataset_path(args.dataset_dir, dataset_type, dataset):
        dataset_logger.warning(f"データセットパスが見つかりません: {dataset_type}/{dataset}")
        dataset_logger.warning(f"指定されたデータセットディレクトリ: {args.dataset_dir}")
        
        # より詳細な情報を表示
        if dataset_type == "sem_seg":
            expected_path = os.path.join(args.dataset_dir, dataset)
            dataset_logger.warning(f"期待されるパス: {expected_path}")
            dataset_logger.warning(f"パスの存在: {os.path.exists(expected_path)}")
        elif dataset_type == "refer_seg":
            images_dir = os.path.join(args.dataset_dir, "refer_seg", "images")
            dataset_logger.warning(f"期待されるimagesパス: {images_dir}")
            dataset_logger.warning(f"パスの存在: {os.path.exists(images_dir)}")
        elif dataset_type == "vqa":
            expected_path = os.path.join(args.dataset_dir, "llava_dataset")
            dataset_logger.warning(f"期待されるパス: {expected_path}")
            dataset_logger.warning(f"パスの存在: {os.path.exists(expected_path)}")
        elif dataset_type == "reason_seg":
            base_path = os.path.join(args.dataset_dir, "reason_seg")
            dataset_path = os.path.join(base_path, "ReasonSeg")
            train_path = os.path.join(dataset_path, "train")
            dataset_logger.warning(f"ベースパス: {base_path}, 存在: {os.path.exists(base_path)}")
            dataset_logger.warning(f"データセットパス: {dataset_path}, 存在: {os.path.exists(dataset_path)}")
            dataset_logger.warning(f"トレーニングパス: {train_path}, 存在: {os.path.exists(train_path)}")
        
        dataset_logger.warning("トレーニングをスキップします。")
        
        # 実行サマリーを更新
        result_summary["status"] = "skipped"
        result_summary["error"] = "データセットパスが見つかりません"
        EXECUTION_SUMMARY["datasets_processed"].append(result_summary)
        EXECUTION_SUMMARY["skipped_count"] += 1
        
        return False
    
    # dry_runモードの場合はコマンド実行をスキップ
    if args.dry_run:
        dataset_logger.info(f"[Dry Run] データセットパスの検証が成功しました: {dataset_type}/{dataset}")
        
        # 実行サマリーを更新
        result_summary["status"] = "dry_run_success"
        EXECUTION_SUMMARY["datasets_processed"].append(result_summary)
        EXECUTION_SUMMARY["success_count"] += 1
        
        return True
    
    # 出力ディレクトリを設定
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"lisa-gemma3-{dataset_type}-{dataset}-{timestamp}"
    exp_dir = os.path.join(args.log_base_dir, exp_name)
    
    # コマンドの構築
    cmd = [
        "python", "train_full.py",
        "--load_in_4bit" if args.load_in_4bit else "",
        f"--lr", str(args.lr),
        f"--batch_size", str(args.batch_size),
        f"--grad_accumulation_steps", str(args.grad_accumulation_steps),
        f"--epochs", str(args.epochs),
        f"--steps_per_epoch", str(args.steps_per_epoch),
        f"--precision", args.precision,
        f"--vision_tower", args.vision_tower,
        f"--dataset_dir", args.dataset_dir,
        f"--log_base_dir", args.log_base_dir,
        f"--exp_name", exp_name,
    ]
    
    # デバッグモード関連の引数
    if args.debug:
        cmd.append("--debug")
        cmd.extend([f"--debug_dataset", dataset])
        cmd.extend([f"--debug_samples", str(args.debug_samples)])
    
    # キャッシュオプション
    if args.use_cached_data:
        cmd.append("--use_cached_data")
    if args.save_cached_data:
        cmd.append("--save_cached_data")
    
    # データセット固有の引数を追加
    dataset_args = create_dataset_specific_args(dataset_type, dataset)
    cmd.extend(dataset_args)
    
    # 空の引数を除去
    cmd = [arg for arg in cmd if arg]
    
    # コマンド表示
    cmd_str = " ".join(cmd)
    dataset_logger.info(f"実行コマンド: {cmd_str}")
    
    # コマンドをログファイルに保存
    with open(os.path.join(LOG_DIR, f"{dataset_type}_{dataset}_command.txt"), "w") as f:
        f.write(cmd_str)
    
    # コマンドの実行
    try:
        start_time = time.time()
        
        # トレーニングコマンドの出力をファイルにリダイレクト
        log_output_file = os.path.join(LOG_DIR, f"{dataset_type}_{dataset}_output.log")
        with open(log_output_file, "w") as log_file:
            result = subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        
        end_time = time.time()
        duration = end_time - start_time
        
        dataset_logger.info(f"データセット '{dataset_type}/{dataset}' の実行が完了しました (所要時間: {duration:.2f}秒)")
        
        # 実行サマリーを更新
        result_summary["status"] = "success"
        result_summary["duration"] = duration
        result_summary["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        EXECUTION_SUMMARY["datasets_processed"].append(result_summary)
        EXECUTION_SUMMARY["success_count"] += 1
        EXECUTION_SUMMARY["total_duration"] += duration
        
        return True
        
    except subprocess.CalledProcessError as e:
        dataset_logger.error(f"データセット '{dataset_type}/{dataset}' の実行中にエラーが発生しました: {e}")
        
        # 実行サマリーを更新
        result_summary["status"] = "failed"
        result_summary["error"] = str(e)
        result_summary["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        EXECUTION_SUMMARY["datasets_processed"].append(result_summary)
        EXECUTION_SUMMARY["failure_count"] += 1
        
        return False

def save_execution_summary():
    """実行サマリーをJSONファイルに保存します"""
    global LOG_DIR, EXECUTION_SUMMARY
    
    if LOG_DIR:
        EXECUTION_SUMMARY["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # サマリーファイルを保存
        summary_file = os.path.join(LOG_DIR, "execution_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(EXECUTION_SUMMARY, f, indent=2, ensure_ascii=False)
        
        # 人間が読みやすいレポートも生成
        report_file = os.path.join(LOG_DIR, "execution_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=== Gemma3-LISA 学習実行レポート ===\n\n")
            f.write(f"開始時刻: {EXECUTION_SUMMARY['start_time']}\n")
            f.write(f"終了時刻: {EXECUTION_SUMMARY['end_time']}\n")
            f.write(f"総実行時間: {EXECUTION_SUMMARY['total_duration']:.2f}秒\n")
            f.write(f"成功数: {EXECUTION_SUMMARY['success_count']}\n")
            f.write(f"失敗数: {EXECUTION_SUMMARY['failure_count']}\n")
            f.write(f"スキップ数: {EXECUTION_SUMMARY['skipped_count']}\n\n")
            
            f.write("=== データセット別結果 ===\n\n")
            for ds in EXECUTION_SUMMARY["datasets_processed"]:
                f.write(f"データセット: {ds['dataset_type']}/{ds['dataset']}\n")
                f.write(f"  ステータス: {ds['status']}\n")
                f.write(f"  開始時刻: {ds['start_time']}\n")
                
                if "end_time" in ds and ds["end_time"]:
                    f.write(f"  終了時刻: {ds['end_time']}\n")
                
                if ds["status"] != "skipped":
                    f.write(f"  所要時間: {ds.get('duration', 0):.2f}秒\n")
                
                if ds["error"]:
                    f.write(f"  エラー: {ds['error']}\n")
                
                f.write("\n")

def main():
    """メイン関数"""
    args = parse_args()
    
    # データセットディレクトリの絶対パスを確保
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    
    # ログディレクトリを作成
    os.makedirs(args.log_base_dir, exist_ok=True)
    
    # ロギングのセットアップ
    logger = setup_logging(args)
    
    # データセットディレクトリの存在を確認
    if not os.path.exists(args.dataset_dir):
        logger.error(f"指定されたデータセットディレクトリが存在しません: {args.dataset_dir}")
        save_execution_summary()
        return
    
    # データセットタイプとその中のデータセットを定義
    dataset_types = []
    if args.run_sem_seg:
        dataset_types.append("sem_seg")
    if args.run_refer_seg:
        dataset_types.append("refer_seg")
    if args.run_vqa:
        dataset_types.append("vqa")
    if args.run_reason_seg:
        dataset_types.append("reason_seg")
    
    # GPU情報の表示
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        logger.info(f"=== GPU環境で実行: 利用可能なGPU {device_count}台 ===")
        for i, name in enumerate(device_names):
            logger.info(f"GPU {i}: {name}")
    else:
        logger.info("=== CPU環境で実行: GPUが利用できません ===")
    
    # 実行設定の表示
    logger.info(f"=== 実行設定 ===")
    logger.info(f"デバッグモード: {args.debug}")
    logger.info(f"データセットサンプル数: {args.debug_samples}")
    logger.info(f"対象データセットタイプ: {', '.join(dataset_types)}")
    logger.info(f"データセットディレクトリ: {args.dataset_dir}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"勾配累積ステップ数: {args.grad_accumulation_steps}")
    logger.info(f"学習率: {args.lr}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"ステップ数/エポック: {args.steps_per_epoch}")
    logger.info(f"ログ出力ディレクトリ: {LOG_DIR}")
    
    # 再開位置の処理
    start_type_idx = 0
    if args.continue_from_dataset_type and args.continue_from_dataset_type in dataset_types:
        start_type_idx = dataset_types.index(args.continue_from_dataset_type)
        logger.info(f"データセットタイプ '{args.continue_from_dataset_type}' から実行を再開します")
    
    # 各データセットタイプに対する処理
    for type_idx, dataset_type in enumerate(dataset_types[start_type_idx:], start=start_type_idx):
        logger.info(f"[{type_idx+1}/{len(dataset_types)}] データセットタイプ '{dataset_type}' の処理を開始します...")
        
        # このタイプのデータセットリストを取得
        datasets = get_dataset_list(dataset_type)
        if not datasets:
            logger.warning(f"データセットタイプ '{dataset_type}' に対するデータセットが見つかりません。スキップします。")
            continue
        
        # 再開位置の処理（データセット内）
        start_dataset_idx = 0
        if type_idx == start_type_idx and args.continue_from_dataset:
            try:
                start_dataset_idx = datasets.index(args.continue_from_dataset)
                logger.info(f"データセット '{args.continue_from_dataset}' から実行を再開します")
            except ValueError:
                logger.warning(f"指定されたデータセット '{args.continue_from_dataset}' が見つかりません。")
        
        # 各データセットに対する処理
        for ds_idx, dataset in enumerate(datasets[start_dataset_idx:], start=start_dataset_idx):
            logger.info(f"[{type_idx+1}/{len(dataset_types)}] [{ds_idx+1}/{len(datasets)}] "
                       f"データセット '{dataset_type}/{dataset}' の処理...")
            
            # トレーニングの実行
            success = run_training_for_dataset(args, dataset_type, dataset)
            
            if not success:
                logger.warning(f"データセット '{dataset_type}/{dataset}' の処理に失敗しました。次に進みます。")
            
            # 中間サマリーを保存
            save_execution_summary()
            
            # 短い休止を入れる
            time.sleep(2)
    
    logger.info("すべてのデータセットの処理が完了しました！")
    
    # 最終的な実行サマリーを保存
    save_execution_summary()
    logger.info(f"実行結果サマリーを保存しました: {os.path.join(LOG_DIR, 'execution_summary.json')}")
    logger.info(f"実行レポートを保存しました: {os.path.join(LOG_DIR, 'execution_report.txt')}")

if __name__ == "__main__":
    main() 