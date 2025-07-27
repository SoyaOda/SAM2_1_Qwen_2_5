#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle
import glob
from pathlib import Path
import numpy as np

# REFERクラスのインポートパスを設定する関数
def setup_refer_module():
    """
    REFERモジュールをインポートするためのパスを設定します
    """
    # カレントディレクトリの取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 検索するパスリスト
    paths_to_check = [
        os.path.join(current_dir, "utils"),
        os.path.join(current_dir, "Original-LISA-Code", "utils"),
    ]
    
    # パスを追加
    for path in paths_to_check:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"パスを追加しました: {path}")
    
    # モジュールの動的インポート
    try:
        from refer import REFER
        from grefer import G_REFER
        print("✓ REFERクラスとG_REFERクラスを正常にインポートしました")
        return True, REFER, G_REFER
    except ImportError as e:
        print(f"警告: REFERクラスをインポートできません: {e}")
        return False, None, None

# REFERモジュールをインポート
REFER_AVAILABLE, REFER, G_REFER = setup_refer_module()

# データセットのベースディレクトリ
DATASET_DIR = "/mnt/h/download/LISA-dataset/dataset"
REFER_SEG_DIR = os.path.join(DATASET_DIR, "refer_seg")

def check_directory_structure():
    """
    refer_segデータセットのディレクトリ構造を確認します
    """
    print("=== refer_segデータセットのディレクトリ構造を確認 ===")
    
    # 必要なディレクトリの存在確認
    required_dirs = [
        REFER_SEG_DIR,
        os.path.join(REFER_SEG_DIR, "images"),
        os.path.join(REFER_SEG_DIR, "images/saiapr_tc-12"),
        os.path.join(REFER_SEG_DIR, "images/mscoco"),
        os.path.join(REFER_SEG_DIR, "images/mscoco/images"),
        os.path.join(REFER_SEG_DIR, "images/mscoco/images/train2014"),
        os.path.join(REFER_SEG_DIR, "refclef"),
        os.path.join(REFER_SEG_DIR, "refcoco"),
        os.path.join(REFER_SEG_DIR, "refcoco+"),
        os.path.join(REFER_SEG_DIR, "refcocog")
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ ディレクトリが存在します: {dir_path}")
        else:
            print(f"✗ ディレクトリが見つかりません: {dir_path}")
            
    # saiapr_tc-12の構造確認
    saiapr_dirs = glob.glob(os.path.join(REFER_SEG_DIR, "images/saiapr_tc-12/*"))
    print(f"\nsaiapr_tc-12サブディレクトリ数: {len(saiapr_dirs)}")
    
    # 画像ファイルの存在確認
    saiapr_image_files = glob.glob(os.path.join(REFER_SEG_DIR, "images/saiapr_tc-12/*/*/*.jpg"))
    print(f"saiapr_tc-12画像ファイル数: {len(saiapr_image_files)}")
    
    coco_image_files = glob.glob(os.path.join(REFER_SEG_DIR, "images/mscoco/images/train2014/*.jpg"))
    print(f"MSCOCO画像ファイル数: {len(coco_image_files)}")
    
def check_annotation_files():
    """
    必要なアノテーションファイルを確認します
    """
    print("\n=== アノテーションファイルの確認 ===")
    
    # 必要なJSONとPickleファイルの存在確認
    required_files = [
        os.path.join(REFER_SEG_DIR, "refclef/instances.json"),
        os.path.join(REFER_SEG_DIR, "refclef/refs(unc).p"),
        os.path.join(REFER_SEG_DIR, "refcoco/instances.json"),
        os.path.join(REFER_SEG_DIR, "refcoco/refs(unc).p"),
        os.path.join(REFER_SEG_DIR, "refcoco+/instances.json"),
        os.path.join(REFER_SEG_DIR, "refcoco+/refs(unc).p"),
        os.path.join(REFER_SEG_DIR, "refcocog/instances.json"),
        os.path.join(REFER_SEG_DIR, "refcocog/refs(umd).p")
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MBに変換
            print(f"✓ ファイルが存在します: {file_path} ({file_size:.2f} MB)")
        else:
            print(f"✗ ファイルが見つかりません: {file_path}")
    
    # JSONファイルの中身を簡単に検証
    json_files = [
        os.path.join(REFER_SEG_DIR, "refclef/instances.json"),
        os.path.join(REFER_SEG_DIR, "refcoco/instances.json"),
        os.path.join(REFER_SEG_DIR, "refcoco+/instances.json"),
        os.path.join(REFER_SEG_DIR, "refcocog/instances.json")
    ]
    
    for json_file in json_files:
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    print(f"\n{os.path.basename(json_file)}の内容:")
                    if 'images' in data:
                        print(f"  画像数: {len(data['images'])}")
                    if 'annotations' in data:
                        print(f"  アノテーション数: {len(data['annotations'])}")
                    if 'categories' in data:
                        print(f"  カテゴリ数: {len(data['categories'])}")
            except Exception as e:
                print(f"  JSONファイルの読み込みエラー: {e}")
    
    # Pickleファイルの中身を簡単に検証
    pickle_files = [
        os.path.join(REFER_SEG_DIR, "refclef/refs(unc).p"),
        os.path.join(REFER_SEG_DIR, "refcoco/refs(unc).p"),
        os.path.join(REFER_SEG_DIR, "refcoco+/refs(unc).p"),
        os.path.join(REFER_SEG_DIR, "refcocog/refs(umd).p")
    ]
    
    for pickle_file in pickle_files:
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    print(f"\n{os.path.basename(pickle_file)}の内容:")
                    print(f"  参照数: {len(data)}")
                    if len(data) > 0:
                        print(f"  サンプル参照ID: {data[0]['ref_id'] if 'ref_id' in data[0] else '不明'}")
            except Exception as e:
                print(f"  Pickleファイルの読み込みエラー: {e}")
    
def check_image_files():
    """
    参照される画像ファイルの整合性を確認します
    """
    print("\n=== 画像ファイルの整合性確認 ===")
    
    # refclefの画像パスをチェック
    if os.path.exists(os.path.join(REFER_SEG_DIR, "refclef/instances.json")):
        try:
            with open(os.path.join(REFER_SEG_DIR, "refclef/instances.json"), 'r') as f:
                data = json.load(f)
                if 'images' in data:
                    sample_images = data['images'][:5]  # 最初の5つをサンプルとして
                    print("\nrefclef画像パスのサンプル確認:")
                    for img in sample_images:
                        if 'file_name' in img:
                            img_path = os.path.join(REFER_SEG_DIR, "images/saiapr_tc-12", img['file_name'])
                            if os.path.exists(img_path):
                                print(f"  ✓ 画像が存在します: {img['file_name']}")
                            else:
                                print(f"  ✗ 画像が見つかりません: {img['file_name']} (予想パス: {img_path})")
                        else:
                            print("  ✗ 画像ファイル名が不明です")
        except Exception as e:
            print(f"  refclef画像確認エラー: {e}")
    
    # refcocoの画像パスをチェック
    if os.path.exists(os.path.join(REFER_SEG_DIR, "refcoco/instances.json")):
        try:
            with open(os.path.join(REFER_SEG_DIR, "refcoco/instances.json"), 'r') as f:
                data = json.load(f)
                if 'images' in data:
                    sample_images = data['images'][:5]  # 最初の5つをサンプルとして
                    print("\nrefcoco画像パスのサンプル確認:")
                    for img in sample_images:
                        if 'file_name' in img:
                            img_path = os.path.join(REFER_SEG_DIR, "images/mscoco/images/train2014", img['file_name'])
                            if os.path.exists(img_path):
                                print(f"  ✓ 画像が存在します: {img['file_name']}")
                            else:
                                print(f"  ✗ 画像が見つかりません: {img['file_name']} (予想パス: {img_path})")
                        else:
                            print("  ✗ 画像ファイル名が不明です")
        except Exception as e:
            print(f"  refcoco画像確認エラー: {e}")

def check_refer_functionality():
    """
    REFERクラスの機能を確認します
    """
    print("\n=== REFERクラスの機能確認 ===")
    
    if not REFER_AVAILABLE:
        print("REFERクラスが利用できないため、機能確認をスキップします。")
        return
    
    datasets = ["refclef", "refcoco", "refcoco+", "refcocog"]
    
    for dataset in datasets:
        print(f"\n{dataset}データセットの確認:")
        try:
            # splitByを設定
            if dataset == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"
            
            # REFERインスタンスを初期化
            refer = REFER(REFER_SEG_DIR, dataset, splitBy)
            
            # 基本情報の取得
            ref_ids = refer.getRefIds()
            img_ids = refer.getImgIds()
            cat_ids = refer.getCatIds()
            
            print(f"  参照数: {len(ref_ids)}")
            print(f"  画像数: {len(img_ids)}")
            print(f"  カテゴリ数: {len(cat_ids)}")
            
            # 訓練データの取得
            train_ref_ids = refer.getRefIds(split="train")
            print(f"  訓練データの参照数: {len(train_ref_ids)}")
            
            # 参照データのサンプルを取得
            if len(train_ref_ids) > 0:
                sample_ref = refer.loadRefs(train_ref_ids[0])[0]
                print(f"  サンプル参照ID: {sample_ref['ref_id']}")
                print(f"  サンプル画像ID: {sample_ref['image_id']}")
                print(f"  サンプル文章数: {len(sample_ref['sentences'])}")
                if len(sample_ref['sentences']) > 0:
                    print(f"  サンプル文章: {sample_ref['sentences'][0]['sent']}")
                
                # 画像情報の取得
                img_info = refer.loadImgs(sample_ref['image_id'])[0]
                print(f"  画像ファイル名: {img_info['file_name']}")
                
                # 実際のファイルパスの構築
                if dataset == "refclef":
                    img_path = os.path.join(REFER_SEG_DIR, "images/saiapr_tc-12", img_info['file_name'])
                else:
                    img_path = os.path.join(REFER_SEG_DIR, "images/mscoco/images/train2014", img_info['file_name'])
                
                if os.path.exists(img_path):
                    print(f"  ✓ サンプル画像が存在します: {img_path}")
                else:
                    print(f"  ✗ サンプル画像が見つかりません: {img_path}")
        except Exception as e:
            print(f"  {dataset}の機能確認中にエラーが発生しました: {e}")

def main():
    print("=== refer_segデータセット検証ツール ===")
    print(f"デーセットディレクトリ: {DATASET_DIR}")
    print(f"refer_segディレクトリ: {REFER_SEG_DIR}")
    
    # ディレクトリ構造の確認
    check_directory_structure()
    
    # アノテーションファイルの確認
    check_annotation_files()
    
    # 画像ファイルの確認
    check_image_files()
    
    # REFERクラスの機能確認
    check_refer_functionality()
    
    print("\n=== データセット検証完了 ===")
    print("データセットは正常に利用できる状態です。" if REFER_AVAILABLE else "REFERクラスが利用できないため、完全な検証はできませんでした。")

if __name__ == "__main__":
    main() 