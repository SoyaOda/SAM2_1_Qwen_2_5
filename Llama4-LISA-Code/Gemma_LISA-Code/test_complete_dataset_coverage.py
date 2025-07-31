#!/usr/bin/env python3
"""
完全なデータセットカバレッジテスト
READMEに記載されたすべてのデータセットが適切に処理できるかを検証する
"""

import os
import sys
import json
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_linux import *

def test_dataset_coverage():
    """READMEに記載されたデータセット構造と実装の一致を検証"""
    print("=" * 80)
    print("🔍 完全なデータセットカバレッジテスト")
    print("=" * 80)
    
    # READMEに記載された期待されるデータセット構造
    expected_structure = {
        "dataset/": {
            "ade20k/": ["annotations/", "images/"],
            "coco/": ["train2017/"],
            "cocostuff/": ["train2017/", "val2017/"],
            "llava_dataset/": ["llava_instruct_150k.json"],
            "mapillary/": ["config_v2.0.json", "testing/", "training/", "validation/"],
            "reason_seg/ReasonSeg/": ["train/", "val/", "explanatory/"],
            "refer_seg/": [
                "images/saiapr_tc-12/",
                "images/mscoco/images/train2014/",
                "refclef/", "refcoco/", "refcoco+/", "refcocog/"
            ],
            "vlpart/": [
                "paco/annotations/",
                "pascal_part/train.json",
                "pascal_part/VOCdevkit/"
            ]
        }
    }
    
    print("📁 データセット構造の検証...")
    
    # 基本パスの確認
    base_dir = DATASET_BASE_DIR
    if not os.path.exists(base_dir):
        print(f"❌ データセットベースディレクトリが見つかりません: {base_dir}")
        return False
    
    print(f"✅ データセットベースディレクトリ: {base_dir}")
    
    # 各データセットの詳細検証
    results = {}
    
    # 1. Semantic Segmentation データセット
    print("\n📊 Semantic Segmentation データセット:")
    
    # ADE20K
    ade20k_path = os.path.join(base_dir, "ade20k")
    ade20k_status = check_ade20k(ade20k_path)
    results["ade20k"] = ade20k_status
    print(f"  {'✅' if ade20k_status else '❌'} ADE20K")
    
    # COCO-Stuff
    cocostuff_path = os.path.join(base_dir, "cocostuff")
    cocostuff_status = check_cocostuff(cocostuff_path)
    results["cocostuff"] = cocostuff_status
    print(f"  {'✅' if cocostuff_status else '❌'} COCO-Stuff")
    
    # Mapillary
    mapillary_path = os.path.join(base_dir, "mapillary")
    mapillary_status = check_mapillary(mapillary_path)
    results["mapillary"] = mapillary_status
    print(f"  {'✅' if mapillary_status else '❌'} Mapillary")
    
    # PACO-LVIS
    paco_path = os.path.join(base_dir, "vlpart", "paco")
    paco_status = check_paco_lvis(paco_path)
    results["paco_lvis"] = paco_status
    print(f"  {'✅' if paco_status else '❌'} PACO-LVIS")
    
    # PASCAL-Part
    pascal_path = os.path.join(base_dir, "vlpart", "pascal_part")
    pascal_status = check_pascal_part(pascal_path)
    results["pascal_part"] = pascal_status
    print(f"  {'✅' if pascal_status else '❌'} PASCAL-Part")
    
    # COCO Images
    coco_path = os.path.join(base_dir, "coco")
    coco_status = check_coco_images(coco_path)
    results["coco"] = coco_status
    print(f"  {'✅' if coco_status else '❌'} COCO Images")
    
    # 2. Referring Segmentation データセット
    print("\n👉 Referring Segmentation データセット:")
    
    refer_seg_path = os.path.join(base_dir, "refer_seg")
    refer_status = check_referring_segmentation(refer_seg_path)
    results["refer_seg"] = refer_status
    print(f"  {'✅' if refer_status else '❌'} Referring Segmentation")
    
    # 3. VQA データセット
    print("\n💬 VQA データセット:")
    
    vqa_path = os.path.join(base_dir, "llava_dataset", "llava_instruct_150k.json")
    vqa_status = check_vqa_dataset(vqa_path)
    results["vqa"] = vqa_status
    print(f"  {'✅' if vqa_status else '❌'} LLaVA-Instruct-150k")
    
    # 4. Reasoning Segmentation データセット
    print("\n🧠 Reasoning Segmentation データセット:")
    
    reason_seg_path = os.path.join(base_dir, "reason_seg", "ReasonSeg")
    reason_status = check_reason_seg(reason_seg_path)
    results["reason_seg"] = reason_status
    print(f"  {'✅' if reason_status else '❌'} ReasonSeg")
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("📋 検証結果サマリー:")
    print("=" * 80)
    
    total_datasets = len(results)
    successful_datasets = sum(results.values())
    
    for dataset, status in results.items():
        print(f"  {'✅' if status else '❌'} {dataset}")
    
    print(f"\n📊 成功率: {successful_datasets}/{total_datasets} ({successful_datasets/total_datasets*100:.1f}%)")
    
    if successful_datasets == total_datasets:
        print("🎉 すべてのデータセットが正しく配置され、処理可能です！")
        return True
    else:
        print("⚠️  一部のデータセットに問題があります。上記の詳細を確認してください。")
        return False

def check_ade20k(path):
    """ADE20Kデータセットの検証"""
    required_dirs = ["annotations", "images"]
    required_files = ["annotations/training", "images/training"]
    
    if not os.path.exists(path):
        return False
    
    for dir_name in required_dirs:
        if not os.path.exists(os.path.join(path, dir_name)):
            return False
    
    # 実際のファイル数を確認
    train_images = os.path.join(path, "images", "training")
    train_annotations = os.path.join(path, "annotations", "training")
    
    if os.path.exists(train_images) and os.path.exists(train_annotations):
        img_count = len([f for f in os.listdir(train_images) if f.endswith('.jpg')])
        ann_count = len([f for f in os.listdir(train_annotations) if f.endswith('.png')])
        return img_count > 0 and ann_count > 0
    
    return False

def check_cocostuff(path):
    """COCO-Stuffデータセットの検証"""
    if not os.path.exists(path):
        return False
    
    train_dir = os.path.join(path, "train2017")
    if os.path.exists(train_dir):
        file_count = len([f for f in os.listdir(train_dir) if f.endswith('.png')])
        return file_count > 0
    return False

def check_mapillary(path):
    """Mapillaryデータセットの検証"""
    if not os.path.exists(path):
        return False
    
    required_items = ["config_v2.0.json", "training", "validation"]
    for item in required_items:
        if not os.path.exists(os.path.join(path, item)):
            return False
    
    # config.jsonの内容確認
    config_path = os.path.join(path, "config_v2.0.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return "labels" in config
    except:
        return False

def check_paco_lvis(path):
    """PACO-LVISデータセットの検証"""
    if not os.path.exists(path):
        return False
    
    annotation_file = os.path.join(path, "annotations", "paco_lvis_v1", "paco_lvis_v1_train.json")
    return os.path.exists(annotation_file)

def check_pascal_part(path):
    """PASCAL-Partデータセットの検証"""
    if not os.path.exists(path):
        return False
    
    required_items = ["train.json", "VOCdevkit"]
    for item in required_items:
        if not os.path.exists(os.path.join(path, item)):
            return False
    return True

def check_coco_images(path):
    """COCO画像データセットの検証"""
    if not os.path.exists(path):
        return False
    
    train_dir = os.path.join(path, "train2017")
    if os.path.exists(train_dir):
        file_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        return file_count > 0
    return False

def check_referring_segmentation(path):
    """Referring Segmentationデータセットの検証"""
    if not os.path.exists(path):
        return False
    
    # 必要なサブディレクトリ
    required_dirs = ["refcoco", "refcoco+", "refcocog", "refclef", "images"]
    for dir_name in required_dirs:
        if not os.path.exists(os.path.join(path, dir_name)):
            return False
    
    # 画像ディレクトリの確認
    mscoco_images = os.path.join(path, "images", "mscoco", "images", "train2014")
    saiapr_images = os.path.join(path, "images", "saiapr_tc-12")
    
    return os.path.exists(mscoco_images) and os.path.exists(saiapr_images)

def check_vqa_dataset(path):
    """VQAデータセットの検証"""
    if not os.path.exists(path):
        return False
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return len(data) > 0
    except:
        return False

def check_reason_seg(path):
    """ReasonSegデータセットの検証"""
    if not os.path.exists(path):
        return False
    
    required_dirs = ["train", "val", "explanatory"]
    for dir_name in required_dirs:
        if not os.path.exists(os.path.join(path, dir_name)):
            return False
    
    # explanatory/train.jsonの確認
    explanatory_file = os.path.join(path, "explanatory", "train.json")
    return os.path.exists(explanatory_file)

if __name__ == "__main__":
    success = test_dataset_coverage()
    sys.exit(0 if success else 1) 