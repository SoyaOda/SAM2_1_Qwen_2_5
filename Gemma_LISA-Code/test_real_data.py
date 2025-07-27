#!/usr/bin/env python3
"""
実データでのLISA-Gemma3統合テスト
実際のデータセットファイルを使用してエンドツーエンドのテストを実行
+ データセット構造の詳細解析機能
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import glob

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 動的設定管理機能を追加
def get_config():
    """動的設定読み込み（環境変数対応）"""
    config_path = os.environ.get('LISA_CONFIG_PATH', None)
    
    if config_path:
        # 環境変数で指定された設定ファイル
        try:
            config_module = __import__(config_path)
            print(f"✓ カスタム設定ファイルを使用: {config_path}")
            return config_module
        except ImportError:
            print(f"⚠️ カスタム設定ファイル {config_path} が見つかりません")
    
    # デフォルトの設定ファイル検索順序
    config_candidates = ['config_linux', 'config_small_test']  # config_linuxを優先に変更
    
    for config_name in config_candidates:
        try:
            config_module = __import__(config_name)
            print(f"✓ 設定ファイルを使用: {config_name}")
            return config_module
        except ImportError:
            continue
    
    raise ImportError("利用可能な設定ファイルが見つかりません")

# 動的設定読み込み
config = get_config()

from utils.dataset import HybridDataset, collate_fn
from utils.reason_seg_dataset import ReasonSegDataset
from utils.vqa_dataset import VQADataset
from utils.refer_seg_dataset import ReferSegDataset
from utils.sem_seg_dataset import SemSegDataset
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from torch.utils.data import DataLoader

class RealDataTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")
        
        # 設定の確認
        self.check_config()
        
    def check_config(self):
        """設定とデータセットパスの確認"""
        print("=== 設定確認 ===")
        print(f"Dataset base dir: {config.DATASET_BASE_DIR}")
        print(f"SAM checkpoint: {config.SAM_CHECKPOINT_PATH}")
        print(f"Gemma model ID: {config.GEMMA_MODEL_ID}")
        
        # 設定モジュールのcheck_paths関数を呼び出して必須パスを検証
        try:
            if hasattr(config, 'check_all_paths'):
                print("\n必須パスの検証中...")
                validation_result = config.check_all_paths()
                if validation_result:
                    print("✓ すべての必須パスが確認されました")
                else:
                    print("⚠️ 一部のパスが見つかりませんが、テストを続行します")
            elif hasattr(config, 'check_paths'):
                print("\n必須パスの検証中...")
                config.check_paths()
                print("✓ すべての必須パスが確認されました")
            else:
                print("⚠️ パス検証機能が見つかりません。手動でパスを確認してください")
        except Exception as e:
            print(f"\n⚠️ パス検証でエラーが発生: {e}")
            print("テストを続行しますが、データセットパスを確認してください")
    
    def analyze_dataset_structure(self):
        """データセット構造の詳細解析"""
        print("\n" + "=" * 80)
        print("📊 データセット構造詳細解析")
        print("=" * 80)
        
        # 各データセットの解析
        self.analyze_semantic_segmentation_datasets()
        self.analyze_referring_segmentation_datasets()
        self.analyze_vqa_datasets()
        self.analyze_reasoning_segmentation_datasets()
    
    def analyze_semantic_segmentation_datasets(self):
        """セマンティックセグメンテーションデータセットの詳細解析"""
        print("\n📋 セマンティックセグメンテーションデータセット解析")
        print("-" * 60)
        
        sem_seg_datasets = ["ade20k", "cocostuff", "mapillary", "pascal_part", "paco_lvis"]
        
        for ds_name in sem_seg_datasets:
            print(f"\n🔍 {ds_name.upper()} データセット:")
            try:
                if ds_name == "ade20k":
                    self._analyze_ade20k()
                elif ds_name == "cocostuff":
                    self._analyze_cocostuff()
                elif ds_name == "mapillary":
                    self._analyze_mapillary()
                elif ds_name == "pascal_part":
                    self._analyze_pascal_part()
                elif ds_name == "paco_lvis":
                    self._analyze_paco_lvis()
            except Exception as e:
                print(f"  ❌ 解析失敗: {e}")
                import traceback
                traceback.print_exc()
    
    def _analyze_ade20k(self):
        """ADE20Kデータセットの詳細解析"""
        ade_path = os.path.join(config.DATASET_BASE_DIR, "ade20k")
        
        if not os.path.exists(ade_path):
            print(f"  ❌ ディレクトリが存在しません: {ade_path}")
            return
            
        # ディレクトリ構造
        print(f"  📁 ベースパス: {ade_path}")
        
        # クラス情報
        classes_file = "utils/ade20k_classes.json"
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                classes = json.load(f)
            print(f"  📊 クラス数: {len(classes)}")
            print(f"  📝 サンプルクラス: {classes[:5]}")
        
        # 画像数確認
        train_images_dir = os.path.join(ade_path, "images", "training")
        if os.path.exists(train_images_dir):
            image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
            print(f"  🖼️  訓練画像数: {len(image_files)}")
            
            # サンプル画像の詳細
            if len(image_files) > 0:
                sample_img_path = os.path.join(train_images_dir, image_files[0])
                self._analyze_image_details(sample_img_path, "ADE20K")
        
        # アノテーション数確認
        train_ann_dir = os.path.join(ade_path, "annotations", "training")
        if os.path.exists(train_ann_dir):
            ann_files = [f for f in os.listdir(train_ann_dir) if f.endswith('.png')]
            print(f"  🎯 アノテーション数: {len(ann_files)}")
            
            if len(ann_files) > 0:
                sample_ann_path = os.path.join(train_ann_dir, ann_files[0])
                self._analyze_segmentation_mask(sample_ann_path, "ADE20K")
        
        print(f"  ✅ ADE20K解析完了")
    
    def _analyze_cocostuff(self):
        """COCO-Stuffデータセットの詳細解析"""
        cocostuff_path = os.path.join(config.DATASET_BASE_DIR, "cocostuff")
        
        if not os.path.exists(cocostuff_path):
            print(f"  ❌ ディレクトリが存在しません: {cocostuff_path}")
            return
            
        print(f"  📁 ベースパス: {cocostuff_path}")
        
        # クラス情報
        classes_file = "utils/cocostuff_classes.txt"
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                lines = f.readlines()
            classes = [line.strip().split(": ")[-1] for line in lines[1:]]
            print(f"  📊 クラス数: {len(classes)}")
            print(f"  📝 サンプルクラス: {classes[:5]}")
        
        # ラベル画像数確認
        train_labels_dir = os.path.join(cocostuff_path, "train2017")
        if os.path.exists(train_labels_dir):
            label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.png')]
            print(f"  🎯 ラベル画像数: {len(label_files)}")
            
            if len(label_files) > 0:
                sample_label_path = os.path.join(train_labels_dir, label_files[0])
                self._analyze_segmentation_mask(sample_label_path, "COCO-Stuff")
        
        # 対応するCOCO画像の確認
        coco_images_dir = os.path.join(config.DATASET_BASE_DIR, "coco", "train2017")
        if os.path.exists(coco_images_dir):
            coco_images = [f for f in os.listdir(coco_images_dir) if f.endswith('.jpg')]
            print(f"  🖼️  対応COCO画像数: {len(coco_images)}")
            
            if len(coco_images) > 0:
                sample_img_path = os.path.join(coco_images_dir, coco_images[0])
                self._analyze_image_details(sample_img_path, "COCO")
        
        print(f"  ✅ COCO-Stuff解析完了")
    
    def _analyze_mapillary(self):
        """Mapillaryデータセットの詳細解析"""
        config = get_config()  # config変数を取得
        mapillary_path = os.path.join(config.DATASET_BASE_DIR, "mapillary")
        
        if not os.path.exists(mapillary_path):
            print(f"  ❌ ディレクトリが存在しません: {mapillary_path}")
            return
            
        print(f"  📁 ベースパス: {mapillary_path}")
        
        # 設定ファイル
        config_path = os.path.join(mapillary_path, "config_v2.0.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"  📊 設定ラベル数: {len(config.get('labels', []))}")
            
            labels = config.get('labels', [])[:5]
            for i, label in enumerate(labels):
                print(f"    {i+1}. {label.get('readable', 'N/A')} (ID: {label.get('id', 'N/A')})")
        
        # 訓練画像
        train_images_dir = os.path.join(mapillary_path, "training", "images")
        if os.path.exists(train_images_dir):
            image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
            print(f"  🖼️  訓練画像数: {len(image_files)}")
            
            if len(image_files) > 0:
                sample_img_path = os.path.join(train_images_dir, image_files[0])
                self._analyze_image_details(sample_img_path, "Mapillary")
        
        # ラベル画像
        train_labels_dir = os.path.join(mapillary_path, "training", "v2.0", "labels")
        if os.path.exists(train_labels_dir):
            label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.png')]
            print(f"  🎯 ラベル画像数: {len(label_files)}")
            
            if len(label_files) > 0:
                sample_label_path = os.path.join(train_labels_dir, label_files[0])
                self._analyze_segmentation_mask(sample_label_path, "Mapillary")
        
        print(f"  ✅ Mapillary解析完了")
    
    def _analyze_pascal_part(self):
        """Pascal Partデータセットの詳細解析"""
        pascal_path = os.path.join(config.DATASET_BASE_DIR, "vlpart", "pascal_part")
        
        if not os.path.exists(pascal_path):
            print(f"  ❌ ディレクトリが存在しません: {pascal_path}")
            return
            
        print(f"  📁 ベースパス: {pascal_path}")
        
        # アノテーションファイル
        annotation_file = os.path.join(pascal_path, "train.json")
        if os.path.exists(annotation_file):
            coco_api = COCO(annotation_file)
            
            # カテゴリ情報
            categories = coco_api.loadCats(coco_api.getCatIds())
            print(f"  📊 カテゴリ数: {len(categories)}")
            print(f"  📝 サンプルカテゴリ:")
            for i, cat in enumerate(categories[:5]):
                print(f"    {i+1}. {cat['name']} (ID: {cat['id']}, supercategory: {cat.get('supercategory', 'N/A')})")
            
            # 画像情報
            img_ids = coco_api.getImgIds()
            print(f"  🖼️  画像数: {len(img_ids)}")
            
            # アノテーション情報の詳細分析
            if len(img_ids) > 0:
                sample_img_id = img_ids[0]
                ann_ids = coco_api.getAnnIds(imgIds=[sample_img_id])
                anns = coco_api.loadAnns(ann_ids)
                
                print(f"  🎯 サンプル画像のアノテーション数: {len(anns)}")
                
                valid_seg_count = 0
                valid_bbox_count = 0
                
                # 全アノテーションをチェック
                for ann in anns:
                    if ann.get('segmentation') and len(ann['segmentation']) > 0:
                        valid_seg_count += 1
                    if ann.get('bbox') and ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
                        valid_bbox_count += 1
                
                # 最初の3つを詳細表示
                for ann in anns[:3]:
                    print(f"    - アノテーション ID {ann['id']}:")
                    print(f"      カテゴリID: {ann['category_id']}")
                    print(f"      bbox: {ann['bbox']}")
                    print(f"      area: {ann['area']}")
                    print(f"      segmentation: {len(ann.get('segmentation', []))} 要素")
                
                print(f"  ✅ 有効なsegmentation: {valid_seg_count} / {len(anns)} ({valid_seg_count/len(anns)*100:.1f}%)")
                print(f"  ✅ 有効なbbox: {valid_bbox_count} / {len(anns)} ({valid_bbox_count/len(anns)*100:.1f}%)")
                
                # 画像ファイルの確認
                img_info = coco_api.loadImgs([sample_img_id])[0]
                img_path = os.path.join(pascal_path, "VOCdevkit", "VOC2010", "JPEGImages", img_info['file_name'])
                if os.path.exists(img_path):
                    self._analyze_image_details(img_path, "Pascal Part")
                else:
                    print(f"  ❌ 画像ファイルが見つかりません: {img_path}")
        
        print(f"  ✅ Pascal Part解析完了")
    
    def _analyze_paco_lvis(self):
        """PACO-LVISデータセットの詳細解析"""
        paco_path = os.path.join(config.DATASET_BASE_DIR, "vlpart", "paco")
        
        if not os.path.exists(paco_path):
            print(f"  ❌ ディレクトリが存在しません: {paco_path}")
            return
            
        print(f"  📁 ベースパス: {paco_path}")
        
        # アノテーションファイル
        annotation_file = os.path.join(paco_path, "annotations", "paco_lvis_v1", "paco_lvis_v1_train.json")
        if os.path.exists(annotation_file):
            coco_api = COCO(annotation_file)
            
            # カテゴリ情報
            categories = coco_api.loadCats(coco_api.getCatIds())
            print(f"  📊 カテゴリ数: {len(categories)}")
            print(f"  📝 サンプルカテゴリ:")
            for i, cat in enumerate(categories[:5]):
                name = cat['name']
                if ":" in name:
                    obj, part = name.split(":", 1)
                    print(f"    {i+1}. オブジェクト: {obj}, パート: {part} (ID: {cat['id']})")
                else:
                    print(f"    {i+1}. {name} (ID: {cat['id']})")
            
            # 画像情報
            img_ids = coco_api.getImgIds()
            print(f"  🖼️  画像数: {len(img_ids)}")
            
            # アノテーション情報
            if len(img_ids) > 0:
                sample_img_id = img_ids[0]
                ann_ids = coco_api.getAnnIds(imgIds=[sample_img_id])
                anns = coco_api.loadAnns(ann_ids)
                
                print(f"  🎯 サンプル画像のアノテーション数: {len(anns)}")
                
                if len(anns) > 0:
                    sample_ann = anns[0]
                    print(f"    - サンプルアノテーション:")
                    print(f"      カテゴリID: {sample_ann['category_id']}")
                    print(f"      bbox: {sample_ann['bbox']}")
                    print(f"      area: {sample_ann['area']}")
                    print(f"      segmentation: {len(sample_ann.get('segmentation', []))} 要素")
        
        print(f"  ✅ PACO-LVIS解析完了")
    
    def _analyze_image_details(self, image_path, dataset_name):
        """画像ファイルの詳細解析"""
        try:
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    height, width, channels = img.shape
                    file_size = os.path.getsize(image_path) / 1024  # KB
                    print(f"    📷 サンプル画像 ({dataset_name}): {width}x{height}x{channels}, {file_size:.1f}KB")
                else:
                    print(f"    ❌ 画像読み込み失敗: {image_path}")
            else:
                print(f"    ❌ 画像ファイル不存在: {image_path}")
        except Exception as e:
            print(f"    ❌ 画像解析エラー: {e}")
    
    def _analyze_segmentation_mask(self, mask_path, dataset_name):
        """セグメンテーションマスクの詳細解析"""
        try:
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    unique_labels = np.unique(mask)
                    height, width = mask.shape
                    file_size = os.path.getsize(mask_path) / 1024  # KB
                    print(f"    🎭 マスク ({dataset_name}): {width}x{height}, {len(unique_labels)}ラベル, {file_size:.1f}KB")
                    print(f"       ラベル値: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")
                else:
                    print(f"    ❌ マスク読み込み失敗: {mask_path}")
            else:
                print(f"    ❌ マスクファイル不存在: {mask_path}")
        except Exception as e:
            print(f"    ❌ マスク解析エラー: {e}")
    
    def analyze_referring_segmentation_datasets(self):
        """参照セグメンテーションデータセットの解析"""
        print("\n📋 参照セグメンテーションデータセット解析")
        print("-" * 60)
        
        refer_seg_path = os.path.join(config.DATASET_BASE_DIR, "refer_seg")
        if not os.path.exists(refer_seg_path):
            print(f"  ❌ ディレクトリが存在しません: {refer_seg_path}")
            return
        
        print(f"  📁 ベースパス: {refer_seg_path}")
        
        # サブデータセット
        refer_datasets = ["refcoco", "refcoco+", "refcocog", "refclef"]
        for ds in refer_datasets:
            ds_path = os.path.join(refer_seg_path, ds)
            if os.path.exists(ds_path):
                files = os.listdir(ds_path)
                print(f"  📊 {ds}: {len(files)} ファイル")
            else:
                print(f"  ❌ {ds}: ディレクトリ不存在")
        
        # 画像ディレクトリ
        images_path = os.path.join(refer_seg_path, "images")
        if os.path.exists(images_path):
            image_subdirs = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
            print(f"  🖼️  画像サブディレクトリ: {image_subdirs}")
        
        print(f"  ✅ 参照セグメンテーション解析完了")
    
    def analyze_vqa_datasets(self):
        """VQAデータセットの解析"""
        print("\n📋 VQAデータセット解析")
        print("-" * 60)
        
        vqa_path = os.path.join(config.DATASET_BASE_DIR, "llava_dataset", "llava_instruct_150k.json")
        if not os.path.exists(vqa_path):
            print(f"  ❌ ファイルが存在しません: {vqa_path}")
            return
        
        print(f"  📁 VQAファイル: {vqa_path}")
        
        try:
            with open(vqa_path, 'r') as f:
                vqa_data = json.load(f)
            
            print(f"  📊 VQAサンプル数: {len(vqa_data)}")
            
            # サンプルデータの解析
            if len(vqa_data) > 0:
                sample = vqa_data[0]
                print(f"  📝 サンプル構造:")
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    {key}: {type(value).__name__} (長さ: {len(value)})")
                    else:
                        print(f"    {key}: {value}")
            
            # 画像タイプの統計
            image_types = {}
            for i, item in enumerate(vqa_data[:100]):  # 最初の100サンプルをチェック
                if 'image' in item:
                    img_name = item['image']
                    if '/' in img_name:
                        img_type = img_name.split('/')[0]
                        image_types[img_type] = image_types.get(img_type, 0) + 1
            
            print(f"  🖼️  画像タイプ分布（最初の100サンプル）:")
            for img_type, count in image_types.items():
                print(f"    {img_type}: {count}")
        
        except Exception as e:
            print(f"  ❌ VQA解析エラー: {e}")
        
        print(f"  ✅ VQA解析完了")
    
    def analyze_reasoning_segmentation_datasets(self):
        """推論セグメンテーションデータセットの解析"""
        print("\n📋 推論セグメンテーションデータセット解析")
        print("-" * 60)
        
        reason_seg_path = os.path.join(config.DATASET_BASE_DIR, "reason_seg", "ReasonSeg")
        if not os.path.exists(reason_seg_path):
            print(f"  ❌ ディレクトリが存在しません: {reason_seg_path}")
            return
        
        print(f"  📁 ベースパス: {reason_seg_path}")
        
        # サブディレクトリ
        subdirs = ["train", "val", "explanatory"]
        for subdir in subdirs:
            subdir_path = os.path.join(reason_seg_path, subdir)
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                print(f"  📊 {subdir}: {len(files)} ファイル")
            else:
                print(f"  ❌ {subdir}: ディレクトリ不存在")
        
        # explanatory/train.jsonの解析
        train_json_path = os.path.join(reason_seg_path, "explanatory", "train.json")
        if os.path.exists(train_json_path):
            try:
                with open(train_json_path, 'r') as f:
                    train_data = json.load(f)
                
                print(f"  📊 訓練データサンプル数: {len(train_data)}")
                
                if len(train_data) > 0:
                    sample = train_data[0]
                    print(f"  📝 サンプル構造:")
                    for key, value in sample.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"    {key}: {type(value).__name__} (長さ: {len(value)})")
                        else:
                            print(f"    {key}: {value}")
            
            except Exception as e:
                print(f"  ❌ train.json解析エラー: {e}")
        
        print(f"  ✅ 推論セグメンテーション解析完了")
    
    def test_actual_dataset_samples(self):
        """実際のデータセットサンプルの取得・表示テスト"""
        print("\n" + "=" * 80)
        print("📋 実際のデータセットサンプルアクセステスト")
        print("=" * 80)
        
        from transformers import AutoTokenizer
        
        # トークナイザーの準備
        try:
            tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
            print("✅ トークナイザー初期化成功")
        except Exception as e:
            print(f"❌ トークナイザー初期化失敗: {e}")
            return
        
        # 各データセットのサンプルを実際に取得してテスト
        dataset_configs = [
            ("ReasonSeg (全データセット)", ReasonSegDataset, {"reason_seg_data": config.REASON_SEG_DATA}),
            ("VQA (全データセット)", VQADataset, {"vqa_data": config.VQA_DATA}),
            ("ReferSeg (全データセット)", ReferSegDataset, {"refer_seg_data": config.REFER_SEG_DATA}),
            ("SemSeg (全データセット)", SemSegDataset, {"sem_seg_data": config.SEM_SEG_DATA}),
        ]
        
        for ds_name, ds_class, ds_params in dataset_configs:
            print(f"\n🔍 {ds_name} データセットサンプルテスト:")
            try:
                dataset = ds_class(
                    base_image_dir=config.DATASET_BASE_DIR,
                    tokenizer=tokenizer,
                    samples_per_epoch=5,
                    **ds_params
                )
                
                print(f"  ✅ データセット初期化成功 (サンプル数: {len(dataset)})")
                
                # 最初のサンプルを取得
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"  📝 サンプル構造: {len(sample)} 要素のタプル")
                    
                    for i, element in enumerate(sample):
                        if isinstance(element, str):
                            # テキスト情報
                            if len(element) > 100:
                                print(f"    [{i}] テキスト: {element[:100]}...")
                            else:
                                print(f"    [{i}] テキスト: {element}")
                        elif isinstance(element, Image.Image):
                            # PIL画像
                            print(f"    [{i}] PIL画像: {element.size} ({element.mode})")
                        elif isinstance(element, np.ndarray):
                            # NumPy配列（マスクなど）
                            print(f"    [{i}] NumPy配列: {element.shape} ({element.dtype})")
                            if element.ndim == 2:  # 2Dマスクの場合
                                unique_vals = np.unique(element)
                                print(f"         一意値: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
                        elif isinstance(element, torch.Tensor):
                            # PyTorchテンソル
                            print(f"    [{i}] Tensor: {element.shape} ({element.dtype})")
                        elif isinstance(element, (list, tuple)):
                            # リストやタプル
                            print(f"    [{i}] {type(element).__name__}: 長さ {len(element)}")
                        else:
                            # その他
                            print(f"    [{i}] {type(element).__name__}: {element}")
                
                else:
                    print(f"  ⚠️  データセットにサンプルがありません")
                    
            except Exception as e:
                print(f"  ❌ {ds_name} サンプルテスト失敗: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ 実際のデータセットサンプルアクセステスト完了")
        
    def test_individual_datasets(self):
        """各データセットの個別テスト"""
        print("\n=== 個別データセットテスト ===")
        
        # ReasonSegデータセット
        print("\n1. ReasonSegデータセットテスト")
        try:
            dataset = ReasonSegDataset(
                base_image_dir=config.DATASET_BASE_DIR,
                tokenizer=None,  # 簡易テスト用
                samples_per_epoch=10,
                reason_seg_data=config.REASON_SEG_DATA  # 設定から取得
            )
            print(f"✓ ReasonSegデータセット作成成功 (サンプル数: {len(dataset)})")
            print(f"  使用データセット: {config.REASON_SEG_DATA}")
            
            # 最初のサンプルを取得
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  - サンプル構造: {len(sample)} 要素のタプル")
        except Exception as e:
            print(f"✗ ReasonSegデータセットテスト失敗: {e}")
        
        # VQAデータセット
        print("\n2. VQAデータセットテスト")
        try:
            dataset = VQADataset(
                base_image_dir=config.DATASET_BASE_DIR,
                tokenizer=None,
                samples_per_epoch=10,
                vqa_data=config.VQA_DATA  # 設定から取得
            )
            print(f"✓ VQAデータセット作成成功 (サンプル数: {len(dataset)})")
            print(f"  使用データセット: {config.VQA_DATA}")
        except Exception as e:
            print(f"✗ VQAデータセットテスト失敗: {e}")
        
        # ReferSegデータセット
        print("\n3. ReferSegデータセットテスト")
        try:
            dataset = ReferSegDataset(
                base_image_dir=config.DATASET_BASE_DIR,
                tokenizer=None,
                samples_per_epoch=10,
                refer_seg_data=config.REFER_SEG_DATA  # 設定から取得
            )
            print(f"✓ ReferSegデータセット作成成功 (サンプル数: {len(dataset)})")
            print(f"  使用データセット: {config.REFER_SEG_DATA}")
        except Exception as e:
            print(f"✗ ReferSegデータセットテスト失敗: {e}")
        
        # SemSegデータセット
        print("\n4. SemSegデータセットテスト")
        try:
            dataset = SemSegDataset(
                base_image_dir=config.DATASET_BASE_DIR,
                tokenizer=None,
                samples_per_epoch=10,
                sem_seg_data=config.SEM_SEG_DATA  # 全データセットを使用
            )
            print(f"✓ SemSegデータセット作成成功 (サンプル数: {len(dataset)})")
            print(f"  使用データセット: {config.SEM_SEG_DATA}")
        except Exception as e:
            print(f"✗ SemSegデータセットテスト失敗: {e}")
    
    def test_unified_dataset(self):
        """統合データセットの動作テスト"""
        print("\n=== 統合データセットテスト ===")
        
        try:
            # Gemma-3プロセッサーの初期化
            from transformers import AutoProcessor
            gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
            
            # [SEG]トークンの追加
            seg_token = "[SEG]"
            if seg_token not in gemma_processor.tokenizer.get_vocab():
                gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
            
            # config_linuxの完全な設定を明示的に使用
            import config_linux
            
            # 統合データセットの作成 - config_linuxの設定を強制使用
            dataset = HybridDataset(
                base_image_dir=config_linux.DATASET_BASE_DIR,
                gemma_processor=gemma_processor,
                samples_per_epoch=20,
                dataset="sem_seg||refer_seg||vqa||reason_seg",  # 全データセットを有効化
                sample_rate=[9, 3, 3, 1],  # 仕様書通りのサンプリングレート
                reason_seg_data=config_linux.REASON_SEG_DATA,    # "ReasonSeg|train"
                vqa_data=config_linux.VQA_DATA,                  # "llava_instruct_150k"
                refer_seg_data=config_linux.REFER_SEG_DATA,      # "refclef||refcoco||refcoco+||refcocog"
                sem_seg_data=config_linux.SEM_SEG_DATA,          # "ade20k||cocostuff||mapillary||pascal_part||paco_lvis"
                precision="bf16",
                gemma_image_size=config_linux.GEMMA_IMAGE_SIZE,
                sam_image_size=config_linux.SAM_IMAGE_SIZE,
            )
            print(f"✓ 統合データセット作成成功 (サンプル数: {len(dataset)})")
            
            # 初期化されたデータセットの詳細情報を表示
            print(f"📊 初期化されたデータセット数: {len(dataset.all_datasets)}")
            for i, ds in enumerate(dataset.all_datasets):
                ds_type = type(ds).__name__
                print(f"  {i+1}. {ds_type}: {len(ds)} サンプル")
            
            # データローダーの作成
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            print("✓ データローダー作成成功")
            
            # 1バッチだけテスト
            batch = next(iter(dataloader))
            print("✓ バッチ取得成功")
            
            print("📋 バッチ情報:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, list):
                    print(f"  {key}: List[{len(value)}]")
                else:
                    print(f"  {key}: {type(value)}")
            
        except Exception as e:
            print(f"✗ 統合データセットテスト失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def test_model_initialization(self):
        """モデル初期化テスト"""
        print("\n=== モデル初期化テスト ===")
        
        try:
            # 設定取得
            config = get_config()
            
            # SAMチェックポイントの存在確認
            sam_checkpoint_exists = os.path.exists(config.SAM_CHECKPOINT_PATH)
            print(f"SAMチェックポイント: {config.SAM_CHECKPOINT_PATH}")
            print(f"SAMチェックポイント存在: {sam_checkpoint_exists}")
            
            # SAMチェックポイントが存在する場合は完全初期化、ない場合は部分初期化
            if sam_checkpoint_exists:
                print("✓ SAMチェックポイントが見つかりました。完全なLISA-Gemmaモデルを初期化します。")
                model_config = LisaGemmaConfig(
                    gemma_model_id=config.GEMMA_MODEL_ID,
                    sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,  # SAMチェックポイントを使用
                    gemma_hidden_size=2560,
                    sam_prompt_embed_dim=256
                )
            else:
                print("⚠️ SAMチェックポイントが見つかりません。Gemmaのみで初期化します。")
                model_config = LisaGemmaConfig(
                    gemma_model_id=config.GEMMA_MODEL_ID,
                    sam_checkpoint_path="",  # SAMなしでテスト
                    gemma_hidden_size=2560,
                    sam_prompt_embed_dim=256
                )
            
            print("LISA-Gemmaモデル初期化中...")
            model = LisaGemmaForCausalLM(model_config)
            print("✓ モデル初期化成功")
            
            # SAM機能の確認
            has_sam = model.has_sam_capability()
            print(f"  SAMセグメンテーション機能: {'有効' if has_sam else '無効'}")
            
            # パラメータ情報
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  総パラメータ数: {total_params:,}")
            print(f"  訓練可能パラメータ数: {trainable_params:,}")
            print(f"  訓練可能割合: {100 * trainable_params / total_params:.2f}%")
            
            # 詳細パラメータ分析（仕様書準拠確認）
            print("\n📊 詳細パラメータ分析:")
            component_analysis = {}
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    component = "Unknown"
                    if "gemma_model" in name:
                        if "embed" in name.lower():
                            component = "Gemma Embeddings"
                        elif "lm_head" in name.lower():
                            component = "Gemma LM Head"
                        else:
                            component = "Gemma Other"
                    elif "sam_image_encoder" in name:
                        component = "SAM Image Encoder"
                    elif "sam_mask_decoder" in name:
                        component = "SAM Mask Decoder"
                    elif "mlp_projector" in name:
                        component = "MLP Projector"
                    
                    if component not in component_analysis:
                        component_analysis[component] = 0
                    component_analysis[component] += param.numel()
            
            total_trainable = sum(component_analysis.values())
            for component, params in component_analysis.items():
                percentage = 100 * params / total_params if total_params > 0 else 0
                print(f"  - {component}: {params:,} ({percentage:.3f}%)")
            
            # 仕様書準拠チェック
            expected_percentage = 1.0  # 期待値: 1%未満
            actual_percentage = 100 * trainable_params / total_params
            compliance_status = "✅ 準拠" if actual_percentage < expected_percentage else "⚠️ 要修正"
            print(f"\n🎯 仕様書準拠チェック:")
            print(f"  期待値: {expected_percentage}%未満")
            print(f"  実際値: {actual_percentage:.2f}%")
            print(f"  ステータス: {compliance_status}")
            
            return model
            
        except Exception as e:
            print(f"✗ モデル初期化失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_model_forward(self, model):
        """モデルのフォワードパステスト"""
        print("\n=== モデルフォワードパステスト ===")
        
        if model is None:
            print("✗ モデルが初期化されていません")
            return
        
        try:
            # ダミー画像とテキストでテスト
            dummy_image = Image.new('RGB', (512, 512), color='red')
            test_prompt = "この画像の赤い部分をセグメント化してください。 <SEG>"
            
            print("フォワードパス実行中...")
            with torch.no_grad():
                outputs = model(
                    image=dummy_image,
                    text_prompt=test_prompt,
                    generate_mask=False  # SAMなしなのでマスク生成は無効
                )
            
            print("✓ フォワードパス成功")
            print(f"  出力キー: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"    {key}: {type(value)}")
                    
        except Exception as e:
            print(f"✗ フォワードパス失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def test_loss_function(self):
        """損失関数のテスト"""
        print("\n=== 損失関数テスト ===")
        
        try:
            loss_fn = CompositeLoss(
                ce_loss_weight=1.0,
                dice_loss_weight=0.5,
                bce_loss_weight=2.0
            )
            print("✓ 損失関数初期化成功")
            
            # ダミーデータで損失計算テスト（実際のバッチ形式に合わせる）
            batch_size = 2
            seq_len = 100
            vocab_size = 262146  # Gemma-3の実際の語彙サイズ
            mask_size = 64
            
            # 実際のlogitsとlabels（修正されたlabels形式）
            logits = torch.randn(batch_size, seq_len, vocab_size)
            
            # 修正されたlabels（前半マスク、後半有効）
            labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
            labels[:, seq_len//2:] = torch.randint(0, 1000, (batch_size, seq_len//2))
            
            # 有効ラベルの確認
            valid_labels = (labels != -100).sum().item()
            total_labels = labels.numel()
            print(f"  有効ラベル: {valid_labels}/{total_labels} ({100*valid_labels/total_labels:.1f}%)")
            
            # モデル出力の辞書
            model_outputs = {
                "logits": logits,  # text_lossを計算するためのlogits
                "predicted_masks": torch.randn(batch_size, 1, mask_size, mask_size)
            }
            
            # バッチデータの辞書
            batch = {
                "labels": labels,  # 修正されたlabels
                "ground_truth_mask": torch.randint(0, 2, (batch_size, 1, mask_size, mask_size)).float()
            }
            
            print(f"  predicted_masks shape: {model_outputs['predicted_masks'].shape}")
            print(f"  ground_truth_mask shape: {batch['ground_truth_mask'].shape}")
            
            # 損失計算（修正されたCompositeLoss使用）
            losses = loss_fn(model_outputs, batch)
            print("✓ 損失計算成功")
            
            for key, value in losses.items():
                print(f"    {key}: {value.item():.4f}")
                
        except Exception as e:
            print(f"✗ 損失関数テスト失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def run_all_tests(self):
        """全テストの実行"""
        print("LISA-Gemma3 実データテスト開始")
        print("=" * 50)
        
        # 個別データセットテスト
        self.test_individual_datasets()
        
        # 統合データセットテスト
        self.test_unified_dataset()
        
        # モデル初期化テスト
        model = self.test_model_initialization()
        
        # モデルフォワードパステスト
        self.test_model_forward(model)
        
        # 損失関数テスト
        self.test_loss_function()
        
        print("\n" + "=" * 50)
        print("テスト完了")
    
    def run_dataset_analysis_only(self):
        """データセット構造解析のみ実行"""
        print("LISA-Gemma3 データセット構造解析")
        print("=" * 50)
        
        # データセット構造の詳細解析
        self.analyze_dataset_structure()
        
        # 実際のサンプルアクセステスト
        self.test_actual_dataset_samples()
        
        print("\n" + "=" * 50)
        print("データセット解析完了")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LISA-Gemma3 実データテスト・データセット解析")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="データセット構造解析のみ実行（モデルテストはスキップ）")
    parser.add_argument("--full-test", action="store_true", 
                       help="データセット解析と全モデルテストを実行")
    
    args = parser.parse_args()
    
    tester = RealDataTester()
    
    if args.analyze_only:
        # データセット解析のみ
        tester.run_dataset_analysis_only()
    elif args.full_test:
        # データセット解析 + 全テスト
        tester.run_dataset_analysis_only()
        print("\n" + "=" * 80)
        print("モデルテストに続行...")
        print("=" * 80)
        tester.run_all_tests()
    else:
        # デフォルト: 従来の全テスト
        print("使用方法:")
        print("  --analyze-only: データセット構造解析のみ")
        print("  --full-test: データセット解析 + 全モデルテスト")
        print("  引数なし: 従来の全テスト")
        print()
        tester.run_all_tests()


if __name__ == "__main__":
    main() 