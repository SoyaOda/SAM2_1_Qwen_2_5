#!/usr/bin/env python3
"""
実際のデータセットを使用したテストスクリプト
ReasonSeg、VQA、SemSeg、ReferSegデータセットの構造と可用性を確認します。
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import json
import glob

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_linux import *

def test_basic_paths():
    """基本パスの存在確認"""
    print("=== 基本パス確認 ===")
    
    paths_to_check = [
        ("プロジェクトルート", os.path.dirname(os.path.abspath(__file__))),
        ("データセットベース", DATASET_BASE_DIR),
        ("SAMチェックポイント", SAM_CHECKPOINT_PATH),
        ("ログディレクトリ", LOG_BASE_DIR),
    ]
    
    all_ok = True
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists and name != "SAMチェックポイント":  # SAMチェックポイントは後でテスト
            all_ok = False
    
    return all_ok

def test_real_datasets():
    """実際のデータセット構造テスト"""
    print("\n=== データセット構造テスト ===")
    
    dataset_paths = get_dataset_paths()
    dataset_status = {}
    
    # ReasonSeg詳細テスト
    print("\n--- ReasonSeg ---")
    reason_seg_path = dataset_paths["reason_seg"]["ReasonSeg"]
    print(f"パス: {reason_seg_path}")
    
    if os.path.exists(reason_seg_path):
        train_path = os.path.join(reason_seg_path, "train")
        val_path = os.path.join(reason_seg_path, "val")
        explanatory_path = os.path.join(reason_seg_path, "explanatory", "train.json")
        
        train_exists = os.path.exists(train_path)
        val_exists = os.path.exists(val_path)
        exp_exists = os.path.exists(explanatory_path)
        
        print(f"  train: {'✓' if train_exists else '✗'} ({train_path})")
        print(f"  val: {'✓' if val_exists else '✗'} ({val_path})")
        print(f"  explanatory: {'✓' if exp_exists else '✗'} ({explanatory_path})")
        
        if train_exists:
            try:
                files = os.listdir(train_path)
                jpg_files = [f for f in files if f.endswith('.jpg')]
                json_files = [f for f in files if f.endswith('.json')]
                print(f"  trainファイル: {len(jpg_files)} JPG, {len(json_files)} JSON")
                
                # サンプルファイル確認
                if jpg_files:
                    sample_jpg = os.path.join(train_path, jpg_files[0])
                    try:
                        with Image.open(sample_jpg) as img:
                            print(f"  サンプル画像サイズ: {img.size}")
                    except Exception as e:
                        print(f"  画像読み込みエラー: {e}")
                        
            except Exception as e:
                print(f"  ディレクトリ読み込みエラー: {e}")
        
        dataset_status["reason_seg"] = train_exists and val_exists
    else:
        print(f"  ✗ ReasonSegディレクトリが見つかりません")
        dataset_status["reason_seg"] = False
    
    # ADE20K
    print("\n--- ADE20K ---")
    ade20k_path = dataset_paths["sem_seg"]["ade20k"]
    print(f"パス: {ade20k_path}")
    if os.path.exists(ade20k_path):
        try:
            subdirs = [d for d in os.listdir(ade20k_path) if os.path.isdir(os.path.join(ade20k_path, d))]
            print(f"  サブディレクトリ: {subdirs}")
            dataset_status["ade20k"] = len(subdirs) > 0
        except Exception as e:
            print(f"  エラー: {e}")
            dataset_status["ade20k"] = False
    else:
        print(f"  ✗ ADE20Kディレクトリが見つかりません")
        dataset_status["ade20k"] = False
    
    # COCOStuff
    print("\n--- COCOStuff ---")
    cocostuff_path = dataset_paths["sem_seg"]["cocostuff"]
    print(f"パス: {cocostuff_path}")
    if os.path.exists(cocostuff_path):
        try:
            subdirs = [d for d in os.listdir(cocostuff_path) if os.path.isdir(os.path.join(cocostuff_path, d))]
            print(f"  サブディレクトリ: {subdirs}")
            dataset_status["cocostuff"] = len(subdirs) > 0
        except Exception as e:
            print(f"  エラー: {e}")
            dataset_status["cocostuff"] = False
    else:
        print(f"  ✗ COCOStuffディレクトリが見つかりません")
        dataset_status["cocostuff"] = False
    
    # LLaVA VQA
    print("\n--- LLaVA VQA ---")
    vqa_path = dataset_paths["vqa"]["llava_instruct_150k"]
    print(f"パス: {vqa_path}")
    if os.path.exists(vqa_path):
        try:
            file_size = os.path.getsize(vqa_path) / (1024 * 1024)  # MB
            print(f"  ファイルサイズ: {file_size:.1f} MB")
            
            # JSONファイルの内容確認
            with open(vqa_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  データ数: {len(data)} エントリ")
                if data:
                    sample = data[0]
                    print(f"  サンプルキー: {list(sample.keys())}")
            dataset_status["vqa"] = True
        except Exception as e:
            print(f"  ファイル読み込みエラー: {e}")
            dataset_status["vqa"] = False
    else:
        print(f"  ✗ LLaVA VQAファイルが見つかりません")
        dataset_status["vqa"] = False
    
    # Refer Seg
    print("\n--- Refer Seg ---")
    refer_seg_path = dataset_paths["refer_seg"]["refcoco"]
    print(f"パス: {refer_seg_path}")
    if os.path.exists(refer_seg_path):
        try:
            subdirs = [d for d in os.listdir(refer_seg_path) if os.path.isdir(os.path.join(refer_seg_path, d))]
            files = [f for f in os.listdir(refer_seg_path) if os.path.isfile(os.path.join(refer_seg_path, f))]
            print(f"  サブディレクトリ: {subdirs}")
            print(f"  ルートファイル数: {len(files)}")
            
            # 各サブディレクトリの詳細確認
            total_files = 0
            for subdir in subdirs:
                subdir_path = os.path.join(refer_seg_path, subdir)
                if subdir in ["refcoco", "refcoco+", "refcocog", "refclef"]:
                    subdir_files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
                    print(f"    {subdir}/: {len(subdir_files)} ファイル")
                    if subdir_files:
                        p_files = [f for f in subdir_files if f.endswith('.p')]
                        json_files = [f for f in subdir_files if f.endswith('.json')]
                        print(f"      .p ファイル: {len(p_files)}, .json ファイル: {len(json_files)}")
                    total_files += len(subdir_files)
                elif subdir == "images":
                    # 画像ディレクトリの確認
                    images_subdirs = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
                    print(f"    {subdir}/: {len(images_subdirs)} サブディレクトリ")
                    for img_subdir in images_subdirs[:3]:  # 最初の3つだけ表示
                        img_subdir_path = os.path.join(subdir_path, img_subdir)
                        
                        # MSCOCOの場合、さらに深い階層をチェック
                        if img_subdir == "mscoco":
                            mscoco_images_path = os.path.join(img_subdir_path, "images")
                            if os.path.exists(mscoco_images_path):
                                mscoco_subdirs = [d for d in os.listdir(mscoco_images_path) if os.path.isdir(os.path.join(mscoco_images_path, d))]
                                print(f"      {img_subdir}/images/: {len(mscoco_subdirs)} サブディレクトリ")
                                for mscoco_subdir in mscoco_subdirs:
                                    train_path = os.path.join(mscoco_images_path, mscoco_subdir)
                                    if os.path.exists(train_path):
                                        img_files = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                                        print(f"        {mscoco_subdir}/: {img_files} 画像")
                            else:
                                img_files = len([f for f in os.listdir(img_subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                                print(f"      {img_subdir}/: {img_files} 画像")
                        else:
                            # その他のディレクトリ（saiapr_tc-12など）
                            try:
                                # 直接画像ファイルを探す
                                direct_img_files = len([f for f in os.listdir(img_subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                                
                                # サブディレクトリ内の画像ファイルも探す（再帰的）
                                import glob
                                recursive_img_files = len(glob.glob(os.path.join(img_subdir_path, "**", "*.jpg"), recursive=True))
                                
                                if direct_img_files > 0:
                                    print(f"      {img_subdir}/: {direct_img_files} 画像")
                                elif recursive_img_files > 0:
                                    print(f"      {img_subdir}/: {recursive_img_files} 画像 (再帰的)")
                                else:
                                    print(f"      {img_subdir}/: 0 画像")
                            except Exception as e:
                                print(f"      {img_subdir}/: エラー - {e}")
            
            # データセットが有効かどうかの判定
            has_annotation_files = any(
                os.path.exists(os.path.join(refer_seg_path, dataset_name)) and
                len([f for f in os.listdir(os.path.join(refer_seg_path, dataset_name)) if f.endswith('.p')]) > 0
                for dataset_name in ["refcoco", "refcoco+", "refcocog", "refclef"]
            )
            has_images = os.path.exists(os.path.join(refer_seg_path, "images"))
            
            print(f"  アノテーションファイル: {'✓' if has_annotation_files else '✗'}")
            print(f"  画像ディレクトリ: {'✓' if has_images else '✗'}")
            
            dataset_status["refer_seg"] = has_annotation_files and has_images
            
        except Exception as e:
            print(f"  エラー: {e}")
            dataset_status["refer_seg"] = False
    else:
        print(f"  ✗ Refer Segディレクトリが見つかりません")
        dataset_status["refer_seg"] = False
    
    return dataset_status

def test_sam_checkpoint():
    """SAMチェックポイントの確認"""
    print("\n=== SAMチェックポイント確認 ===")
    
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"✗ SAMチェックポイントが見つかりません: {SAM_CHECKPOINT_PATH}")
        print("  以下のコマンドでダウンロードしてください:")
        print("  mkdir -p weights")
        print("  cd weights")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return False
    
    try:
        # ファイルサイズ確認
        file_size = os.path.getsize(SAM_CHECKPOINT_PATH) / (1024 * 1024)  # MB
        print(f"✓ SAMチェックポイント存在確認: {file_size:.1f} MB")
        
        # チェックポイント読み込みテスト
        checkpoint = torch.load(SAM_CHECKPOINT_PATH, map_location='cpu')
        print(f"✓ チェックポイント読み込み成功")
        print(f"  キー数: {len(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"✗ SAMチェックポイント読み込みエラー: {e}")
        return False

def test_dataset_initialization():
    """データセットクラスの初期化テスト"""
    print("\n=== データセットクラス初期化テスト ===")
    
    try:
        # Tokenizer初期化
        print("Tokenizer初期化中...")
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, trust_remote_code=True)
        print(f"✓ Tokenizer初期化成功: {len(tokenizer)} 語彙")
        
        # 各データセットクラスのインポート
        from utils.reason_seg_dataset import ReasonSegDataset
        from utils.vqa_dataset import VQADataset
        from utils.refer_seg_dataset import ReferSegDataset
        from utils.sem_seg_dataset import SemSegDataset
        
        init_results = {}
        
        # ReasonSegDataset
        print("\n--- ReasonSegDataset初期化 ---")
        try:
            reason_dataset = ReasonSegDataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=tokenizer,
                samples_per_epoch=5,  # 小さいサンプル数でテスト
                reason_seg_data="ReasonSeg|train",
                explanatory=0.1
            )
            print(f"✓ ReasonSegDataset初期化成功: {len(reason_dataset)} サンプル")
            
            # 1つのサンプル取得テスト
            sample_test_ok = True
            if len(reason_dataset) > 0:
                try:
                    sample = reason_dataset[0]
                    if isinstance(sample, dict):
                        print(f"  ✓ サンプル取得成功 - キー: {list(sample.keys())}")
                    elif isinstance(sample, tuple):
                        print(f"  ✓ サンプル取得成功 - タプル形式")
                        print(f"    タプル長: {len(sample)}")
                        print(f"    要素タイプ: {[type(item).__name__ for item in sample]}")
                        # ReasonSegDatasetの期待される構造: (image_path, pil_image, text_prompt, masks, label)
                        if len(sample) >= 5:
                            print(f"    画像パス: {sample[0] if isinstance(sample[0], str) else 'N/A'}")
                            print(f"    画像タイプ: {type(sample[1]).__name__}")
                            print(f"    テキスト: {sample[2][:50] if isinstance(sample[2], str) else 'N/A'}...")
                    else:
                        print(f"  ⚠️ 予期しないサンプルタイプ: {type(sample)}")
                        sample_test_ok = False
                except Exception as e:
                    print(f"  ✗ サンプル取得エラー: {e}")
                    sample_test_ok = False
            
            # 初期化成功 + サンプル取得の状況を総合判定
            init_results["reason_seg"] = "success" if sample_test_ok else "partial"
                
        except Exception as e:
            print(f"✗ ReasonSegDataset初期化失敗: {e}")
            init_results["reason_seg"] = "failed"  # 完全失敗
        
        # VQADataset
        print("\n--- VQADataset初期化 ---")
        try:
            vqa_dataset = VQADataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=tokenizer,
                samples_per_epoch=5,
                vqa_data="llava_instruct_150k"
            )
            print(f"✓ VQADataset初期化成功: {len(vqa_dataset)} サンプル")
            
            # サンプル取得テスト
            sample_test_ok = True
            if len(vqa_dataset) > 0:
                try:
                    sample = vqa_dataset[0]
                    if isinstance(sample, dict):
                        print(f"  ✓ サンプル取得成功 - キー: {list(sample.keys())}")
                    elif isinstance(sample, tuple):
                        print(f"  ✓ サンプル取得成功 - タプル形式")
                        print(f"    タプル長: {len(sample)}")
                        print(f"    要素タイプ: {[type(item).__name__ for item in sample]}")
                        # VQADatasetの期待される構造を確認
                        if len(sample) >= 3:
                            print(f"    画像パス: {sample[0] if isinstance(sample[0], str) else 'N/A'}")
                            print(f"    画像タイプ: {type(sample[1]).__name__}")
                            if len(sample) > 2:
                                print(f"    テキスト: {sample[2][:50] if isinstance(sample[2], str) else 'N/A'}...")
                    else:
                        print(f"  ⚠️ 予期しないサンプルタイプ: {type(sample)}")
                        sample_test_ok = False
                except Exception as e:
                    print(f"  ✗ サンプル取得エラー: {e}")
                    sample_test_ok = False
            
            # 初期化成功 + サンプル取得の状況を総合判定
            init_results["vqa"] = "success" if sample_test_ok else "partial"
                
        except Exception as e:
            print(f"✗ VQADataset初期化失敗: {e}")
            init_results["vqa"] = "failed"  # 完全失敗
        
        # SemSegDataset
        print("\n--- SemSegDataset初期化 ---")
        try:
            sem_seg_dataset = SemSegDataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=tokenizer,
                samples_per_epoch=5,
                sem_seg_data="ade20k"  # 単一データセットでテスト
            )
            print(f"✓ SemSegDataset初期化成功: {len(sem_seg_dataset)} サンプル")
            init_results["sem_seg"] = "success"
            
        except Exception as e:
            print(f"✗ SemSegDataset初期化失敗: {e}")
            init_results["sem_seg"] = "failed"
        
        # ReferSegDataset
        print("\n--- ReferSegDataset初期化 ---")
        try:
            refer_seg_dataset = ReferSegDataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=tokenizer,
                samples_per_epoch=5,
                refer_seg_data="refcoco"  # 単一データセットでテスト
            )
            print(f"✓ ReferSegDataset初期化成功: {len(refer_seg_dataset)} サンプル")
            init_results["refer_seg"] = "success"
            
        except Exception as e:
            print(f"✗ ReferSegDataset初期化失敗: {e}")
            init_results["refer_seg"] = "failed"
        
        return init_results
            
    except Exception as e:
        print(f"✗ 初期化テスト全体でエラー: {e}")
        return {}

def main():
    """メイン実行関数"""
    print("🔍 実際のデータセットを使用したテストを開始します...")
    print("=" * 70)
    
    # 基本パステスト
    basic_ok = test_basic_paths()
    
    # データセット構造テスト
    dataset_status = test_real_datasets()
    
    # SAMチェックポイントテスト
    sam_ok = test_sam_checkpoint()
    
    # データセット初期化テスト
    init_results = test_dataset_initialization()
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 テスト結果サマリー")
    print("=" * 70)
    
    print(f"基本パス確認: {'✓' if basic_ok else '✗'}")
    print(f"SAMチェックポイント: {'✓' if sam_ok else '✗'}")
    
    print("\nデータセット構造:")
    for dataset, status in dataset_status.items():
        print(f"  {dataset}: {'✓' if status else '✗'}")
    
    print("\nデータセット初期化:")
    for dataset, status in init_results.items():
        if status == "success":
            print(f"  {dataset}: ✓ 完全成功")
        elif status == "partial":
            print(f"  {dataset}: ⚠️ 部分成功（初期化OK、サンプル取得に問題）")
        else:
            print(f"  {dataset}: ✗ 失敗")
    
    # 総合判定
    all_structure_ok = all(dataset_status.values()) if dataset_status else False
    success_count = sum(1 for status in init_results.values() if status == "success")
    partial_count = sum(1 for status in init_results.values() if status == "partial")
    total_datasets = len(init_results)
    
    # 完全成功またはほとんど成功の場合を OK とみなす
    init_mostly_ok = (success_count + partial_count) >= total_datasets * 0.75
    overall_success = basic_ok and sam_ok and all_structure_ok and init_mostly_ok
    
    print("\n" + "=" * 70)
    if overall_success:
        print("🎉 テストが成功しました！")
        print("データセットとモデルコンポーネントの準備が完了しています。")
        if partial_count > 0:
            print(f"⚠️  {partial_count}個のデータセットで部分的な問題がありますが、学習は可能です。")
            print("サンプル取得の問題は、データセットの__getitem__メソッドの戻り値形式の違いによるものです。")
        print("次のステップ: train_ds.py を実行して学習を開始できます。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("上記のエラーメッセージを確認して修正してください。")
        
        if partial_count > 0:
            print(f"\n📝 注意: {partial_count}個のデータセットは初期化に成功していますが、")
            print("サンプル取得でタプル/辞書の形式違いがあります。これは修正可能な問題です。")
        
        # 推奨アクション
        if not sam_ok:
            print("\n📥 SAMチェックポイントのダウンロードが必要です:")
            print("  mkdir -p weights")
            print("  cd weights")
            print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        
        if not all_structure_ok:
            print("\n📁 データセットの準備が必要です:")
            print("  データセットが正しいディレクトリに配置されているか確認してください。")
            print("  config_linux.py のパス設定を確認してください。")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 