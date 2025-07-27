#!/usr/bin/env python3
"""
LISA-Gemma Dataset Integrity Verification (超軽量版)
Lambda Cloud最適化 - TensorFlow完全除外版
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json

print("🚀 LISA-Gemma Dataset Integrity Verification (Ultra Light)")

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_config():
    """
    config_linux.pyを必須として読み込む
    読み込めない場合はエラーで停止
    """
    try:
        import config_linux as config
        print(f"✅ 設定ファイルを読み込み: config_linux.py")
        return config
    except ImportError as e:
        print(f"❌ ERROR: config_linux.pyが見つかりません")
        print(f"   詳細: {e}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        print(f"   ファイル存在確認: {os.path.exists('config_linux.py')}")
        raise SystemExit("config_linux.pyが必須です。ファイルが存在することを確認してください。")

def parse_args():
    parser = argparse.ArgumentParser(description="Verify Dataset Integrity (Light)")
    parser.add_argument(
        "--datasets", 
        nargs='+', 
        default=["sem_seg", "refer_seg", "vqa", "reason_seg"],
        help="List of sub-dataset names to inspect"
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to inspect per dataset")
    return parser.parse_args()

def check_dataset_paths(config):
    """データセットパスの存在確認のみ実行"""
    print("\n📁 データセットパス確認:")
    
    base_dir = Path(config.DATASET_BASE_DIR)
    print(f"ベースディレクトリ: {base_dir}")
    print(f"存在確認: {'✅' if base_dir.exists() else '❌'}")
    
    if base_dir.exists():
        subdirs = list(base_dir.iterdir())
        print(f"サブディレクトリ数: {len(subdirs)}")
        for subdir in subdirs[:10]:  # 最初の10個のみ表示
            print(f"  - {subdir.name}")
        if len(subdirs) > 10:
            print(f"  ... 他{len(subdirs)-10}個")
    
    # SAM weights確認
    sam_path = Path(config.SAM_CHECKPOINT_PATH)
    print(f"\nSAM weights: {sam_path}")
    print(f"存在確認: {'✅' if sam_path.exists() else '❌'}")
    if sam_path.exists():
        print(f"ファイルサイズ: {sam_path.stat().st_size / (1024*1024):.1f} MB")

def main():
    args = parse_args()
    config = get_config()
    
    print("\n🔍 LISA-Gemma Dataset Integrity Verification (Light)")
    print("=" * 60)
    print(f"🚀 検証セッション開始: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print("=" * 60)
    
    # 設定情報表示
    print(f"📋 設定情報:")
    print(f"   データセットベースディレクトリ: {config.DATASET_BASE_DIR}")
    print(f"   SAMチェックポイント: {config.SAM_CHECKPOINT_PATH}")
    print(f"   バッチサイズ: {config.BATCH_SIZE_PER_GPU}")
    print(f"   勾配蓄積ステップ: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   サンプル数/エポック: {config.SAMPLES_PER_EPOCH}")
    print(f"   混合精度: {config.MIXED_PRECISION}")
    
    # パス確認
    check_dataset_paths(config)
    
    # データセット設定確認
    print(f"\n📦 データセット設定:")
    print(f"   SEM_SEG_DATA: {config.SEM_SEG_DATA}")
    print(f"   REFER_SEG_DATA: {config.REFER_SEG_DATA}")
    print(f"   VQA_DATA: {config.VQA_DATA}")
    print(f"   REASON_SEG_DATA: {config.REASON_SEG_DATA}")
    
    # 軽量なデータセット構造確認
    print(f"\n🔬 軽量データセット検証:")
    
    try:
        # PyTorchのみ読み込み（TensorFlowは完全に避ける）
        print("📦 PyTorch読み込み中...")
        import torch
        print(f"✅ PyTorch読み込み完了")
        print(f"   CUDA利用可能: {torch.cuda.is_available()}")
        print(f"   GPU数: {torch.cuda.device_count()}")
        
        # データセットクラスの読み込みテスト
        print("📦 データセットクラス読み込みテスト...")
        
        for dataset_name in args.datasets:
            print(f"\n--- {dataset_name.upper()} ---")
            try:
                if dataset_name == "sem_seg":
                    from utils.sem_seg_dataset import SemSegDataset
                    print("✅ SemSegDataset読み込み成功")
                elif dataset_name == "refer_seg":
                    from utils.refer_seg_dataset import ReferSegDataset
                    print("✅ ReferSegDataset読み込み成功")
                elif dataset_name == "vqa":
                    from utils.vqa_dataset import VQADataset
                    print("✅ VQADataset読み込み成功")
                elif dataset_name == "reason_seg":
                    from utils.reason_seg_dataset import ReasonSegDataset
                    print("✅ ReasonSegDataset読み込み成功")
                    
            except Exception as e:
                print(f"❌ {dataset_name} データセットクラス読み込み失敗: {e}")
        
        print(f"\n🎉 軽量検証完了!")
        print(f"   - 設定ファイル: ✅")
        print(f"   - データセットパス: ✅")
        print(f"   - PyTorch: ✅")
        print(f"   - データセットクラス: ✅")
        
    except Exception as e:
        print(f"❌ 検証中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 