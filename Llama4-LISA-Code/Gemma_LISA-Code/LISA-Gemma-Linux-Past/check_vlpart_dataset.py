import os
import sys
import json
import numpy as np
import cv2
import torch
import argparse
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image
import random

# 現在のパスを追加し、必要なモジュールをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# パスの設定
parser = argparse.ArgumentParser()
parser.add_argument("--base_image_dir", type=str, default="/mnt/h/download/LISA-dataset/dataset")
parser.add_argument("--samples", type=int, default=2)
args = parser.parse_args()

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> tuple:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (neww, newh)

def init_paco_lvis(base_image_dir):
    print("\n--- PACO LVISデータセットのチェック ---")
    # フォルダの構造を確認
    original_path = os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json")
    current_path = os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1", "paco_lvis_v1_train.json")
    
    if os.path.exists(original_path):
        print(f"✓ 旧パス構造でファイルが存在: {original_path}")
        coco_path = original_path
    elif os.path.exists(current_path):
        print(f"✓ 現在のパス構造でファイルが存在: {current_path}")
        coco_path = current_path
    else:
        print(f"✗ paco_lvis_v1_train.jsonが見つかりません")
        return None, None, None
    
    try:
        coco_api_paco_lvis = COCO(coco_path)
        all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
        class_map_paco_lvis = {}
        for cat in all_classes:
            cat_split = cat["name"].strip().split(":")
            if len(cat_split) == 1:
                name = cat_split[0].split("_(")[0]
            else:
                assert len(cat_split) == 2
                obj, part = cat_split
                obj = obj.split("_(")[0]
                part = part.split("_(")[0]
                name = (obj, part)
            class_map_paco_lvis[cat["id"]] = name
        img_ids = coco_api_paco_lvis.getImgIds()
        print(f"✓ paco_lvisデータ読み込み成功: {len(img_ids)}枚の画像, {len(class_map_paco_lvis)}個のクラス")
        
        # サンプル画像のパスを確認
        if len(img_ids) > 0:
            sample_img_id = img_ids[0]
            image_info = coco_api_paco_lvis.loadImgs([sample_img_id])[0]
            file_name = image_info["file_name"]
            image_path = os.path.join(base_image_dir, "coco", file_name)
            print(f"サンプル画像パス: {image_path}")
            print(f"画像ファイルの存在: {'はい' if os.path.exists(image_path) else 'いいえ'}")
        
        return class_map_paco_lvis, img_ids, coco_api_paco_lvis
    except Exception as e:
        print(f"✗ paco_lvis読み込み中にエラー: {e}")
        return None, None, None

def init_pascal_part(base_image_dir):
    print("\n--- Pascal Partデータセットのチェック ---")
    pascal_path = os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    
    if not os.path.exists(pascal_path):
        print(f"✗ train.jsonが見つかりません: {pascal_path}")
        return None, None, None
    
    print(f"✓ train.jsonが存在: {pascal_path}")
    
    try:
        coco_api_pascal_part = COCO(pascal_path)
        all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
        class_map_pascal_part = {}
        for cat in all_classes:
            # name属性に「:」が含まれているか確認
            if ":" in cat["name"]:
                cat_main, cat_part = cat["name"].strip().split(":")
                name = (cat_main, cat_part)
            else:
                # 「:」が含まれていない場合はそのまま使用
                name = cat["name"].strip()
            class_map_pascal_part[cat["id"]] = name
        img_ids = coco_api_pascal_part.getImgIds()
        print(f"✓ Pascal Partデータ読み込み成功: {len(img_ids)}枚の画像, {len(class_map_pascal_part)}個のクラス")
        
        # VOCdevkitのパスを確認
        voc_path = os.path.join(base_image_dir, "vlpart", "pascal_part", "VOCdevkit", "VOC2010", "JPEGImages")
        print(f"VOCdevkitパス: {voc_path}")
        print(f"VOCdevkitの存在: {'はい' if os.path.exists(voc_path) else 'いいえ'}")
        
        # サンプル画像のパスを確認
        if len(img_ids) > 0:
            sample_img_id = img_ids[0]
            image_info = coco_api_pascal_part.loadImgs([sample_img_id])[0]
            file_name = image_info["file_name"]
            file_name_full = os.path.join("VOCdevkit", "VOC2010", "JPEGImages", file_name)
            image_path = os.path.join(base_image_dir, "vlpart", "pascal_part", file_name_full)
            print(f"サンプル画像パス: {image_path}")
            print(f"画像ファイルの存在: {'はい' if os.path.exists(image_path) else 'いいえ'}")
        
        return class_map_pascal_part, img_ids, coco_api_pascal_part
    except Exception as e:
        print(f"✗ Pascal Part読み込み中にエラー: {e}")
        return None, None, None

def show_sample_image_with_mask(base_image_dir, ds, img_id, coco_api, class_map):
    image_info = coco_api.loadImgs([img_id])[0]
    file_name = image_info["file_name"]
    
    if ds == "pascal_part":
        file_name = os.path.join("VOCdevkit", "VOC2010", "JPEGImages", file_name)
        image_path = os.path.join(base_image_dir, "vlpart", ds, file_name)
    elif ds == "paco_lvis":
        image_path = os.path.join(base_image_dir, "coco", file_name)
    
    if not os.path.exists(image_path):
        print(f"画像が見つかりません: {image_path}")
        return None
    
    print(f"画像パス: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(512)
    image_resized = transform.apply_image(image)
    
    # アノテーションを取得
    annIds = coco_api.getAnnIds(imgIds=image_info["id"])
    anns = coco_api.loadAnns(annIds)
    
    if len(anns) == 0:
        print("アノテーションがありません")
        return None
    
    # ランダムにアノテーションを選択
    ann = random.choice(anns)
    sampled_cls = class_map[ann["category_id"]]
    
    if isinstance(sampled_cls, tuple):
        obj, part = sampled_cls
        class_name = f"{obj}の{part}"
    else:
        class_name = sampled_cls
    
    # マスクを取得
    try:
        # アノテーションの構造を確認
        if "segmentation" not in ann or not ann["segmentation"]:
            print(f"セグメンテーションが空または存在しません: {ann}")
            # 空のマスクを作成
            mask_img = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
        else:
            # segmentationの形式に応じて処理
            if isinstance(ann["segmentation"], list):
                if len(ann["segmentation"]) == 0:
                    # 空のセグメンテーション
                    mask_img = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                elif isinstance(ann["segmentation"][0], list):
                    # ポリゴン形式（リストのリスト）
                    from pycocotools import mask as mask_utils
                    rle = mask_utils.frPyObjects(
                        ann["segmentation"], 
                        image_info["height"], 
                        image_info["width"]
                    )
                    mask_img = mask_utils.decode(rle)
                    if len(mask_img.shape) > 2:
                        mask_img = np.sum(mask_img, axis=2)
                    mask_img = mask_img.astype(np.uint8)
                else:
                    # RLE形式
                    from pycocotools import mask as mask_utils
                    if isinstance(ann["segmentation"][0], dict):
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                        mask_img = mask_utils.decode(rle)
                        if len(mask_img.shape) > 2:
                            mask_img = np.sum(mask_img, axis=2)
                        mask_img = mask_img.astype(np.uint8)
                    else:
                        print(f"不明なセグメンテーション形式: {ann['segmentation']}")
                        # バウンディングボックスからマスクを作成
                        if "bbox" in ann:
                            bbox = ann["bbox"]
                            mask_img = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                            x, y, w, h = [int(v) for v in bbox]
                            mask_img[y:y+h, x:x+w] = 1
                        else:
                            mask_img = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            else:
                print(f"セグメンテーションが辞書型でもリスト型でもありません: {type(ann['segmentation'])}")
                mask_img = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                
        # マスクのリサイズ
        mask_resized = cv2.resize(mask_img, (image_resized.shape[1], image_resized.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"マスク生成エラー: {e}")
        # エラー内容をより詳細に表示
        import traceback
        traceback.print_exc()
        return None
    
    # 画像とマスクを表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    plt.title("元画像")
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_resized)
    plt.imshow(mask_resized, alpha=0.5, cmap='jet')
    plt.title(f"マスク: {class_name}")
    
    plt.tight_layout()
    plt.savefig(f"sample_{ds}_{img_id}.png")
    print(f"サンプル画像保存: sample_{ds}_{img_id}.png")
    return True

def main():
    print("\n=== vlpartデータセット検証スクリプト ===")
    base_image_dir = args.base_image_dir
    print(f"ベースディレクトリ: {base_image_dir}")
    
    # COCOデータセットが存在するか確認
    coco_path = os.path.join(base_image_dir, "coco")
    if os.path.exists(coco_path):
        print(f"✓ cocoディレクトリが存在: {coco_path}")
    else:
        print(f"✗ cocoディレクトリが見つかりません: {coco_path}")
    
    # 1. PACO LVISデータセットの初期化
    class_map_paco, img_ids_paco, coco_api_paco = init_paco_lvis(base_image_dir)
    
    # 2. Pascal Partデータセットの初期化
    class_map_pascal, img_ids_pascal, coco_api_pascal = init_pascal_part(base_image_dir)
    
    # サンプル画像を表示
    if args.samples > 0 and img_ids_paco is not None and coco_api_paco is not None:
        print("\n--- PACOデータセットのサンプル画像表示 ---")
        for i in range(min(args.samples, len(img_ids_paco))):
            sample_idx = random.randint(0, len(img_ids_paco) - 1)
            sample_id = img_ids_paco[sample_idx]
            show_sample_image_with_mask(base_image_dir, "paco_lvis", sample_id, coco_api_paco, class_map_paco)
    
    if args.samples > 0 and img_ids_pascal is not None and coco_api_pascal is not None:
        print("\n--- Pascal Partデータセットのサンプル画像表示 ---")
        for i in range(min(args.samples, len(img_ids_pascal))):
            sample_idx = random.randint(0, len(img_ids_pascal) - 1)
            sample_id = img_ids_pascal[sample_idx]
            show_sample_image_with_mask(base_image_dir, "pascal_part", sample_id, coco_api_pascal, class_map_pascal)
    
    # 検証結果のまとめ
    print("\n=== 検証結果まとめ ===")
    paco_status = "✓ 利用可能" if img_ids_paco is not None else "✗ 利用不可"
    pascal_status = "✓ 利用可能" if img_ids_pascal is not None else "✗ 利用不可"
    
    print(f"PACOデータセット: {paco_status}")
    print(f"Pascal Partデータセット: {pascal_status}")
    
    if img_ids_paco is not None or img_ids_pascal is not None:
        print("\n現在のデータセット構造で学習に利用できそうですが、以下の修正が必要かもしれません:")
        
        if img_ids_paco is not None:
            print("1. paco_lvis_v1_train.jsonのパスの修正")
            print("   - オリジナルパス: vlpart/paco/annotations/paco_lvis_v1_train.json")
            print("   - 現在のパス: vlpart/paco/annotations/paco_lvis_v1/paco_lvis_v1_train.json")
        
        print("\nコード修正例:")
        print("```python")
        print("def init_paco_lvis(base_image_dir):")
        print("    # オリジナルのパスをチェック")
        print("    original_path = os.path.join(base_image_dir, 'vlpart', 'paco', 'annotations', 'paco_lvis_v1_train.json')")
        print("    # 現在のパスをチェック")
        print("    current_path = os.path.join(base_image_dir, 'vlpart', 'paco', 'annotations', 'paco_lvis_v1', 'paco_lvis_v1_train.json')")
        print("    ")
        print("    # 存在するパスを使用")
        print("    if os.path.exists(original_path):")
        print("        coco_path = original_path")
        print("    elif os.path.exists(current_path):")
        print("        coco_path = current_path")
        print("    else:")
        print("        raise FileNotFoundError('paco_lvis_v1_train.jsonが見つかりません')")
        print("    ")
        print("    coco_api_paco_lvis = COCO(coco_path)")
        print("    # 以下は元のコードと同じ")
        print("```")
    else:
        print("\n✗ どちらのデータセットも利用できない状態です。パスやファイル構造を確認してください。")

if __name__ == "__main__":
    main() 