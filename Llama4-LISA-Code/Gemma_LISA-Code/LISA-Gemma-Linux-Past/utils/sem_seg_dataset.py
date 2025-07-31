import glob
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

# 会話テンプレートのインポート
try:
    from model.gemma3 import conversation as gemma_conversation_lib
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

try:
    from model.llava import conversation as conversation_lib
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

from model.segment_anything.utils.transforms import ResizeLongestSide
from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT, ANSWER_LIST, SHORT_QUESTION_LIST
from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json


def init_mapillary(base_image_dir):
    """Mapillaryデータセットの初期化
    
    Args:
        base_image_dir: ベースとなる画像ディレクトリ
        
    Returns:
        Mapillaryクラスのリスト、画像パスのリスト、ラベルパスのリスト
    """
    mapillary_path = os.path.join(base_image_dir, "mapillary")
    
    if not os.path.exists(mapillary_path):
        print(f"警告: Mapillaryディレクトリが見つかりません: {mapillary_path}")
        return [], [], []
    
    # 設定ファイルのロード
    config_path = os.path.join(mapillary_path, "config_v2.0.json")
    
    if not os.path.exists(config_path):
        print(f"警告: Mapillary設定ファイルが見つかりません: {config_path}")
        return [], [], []
    
    try:
        with open(config_path, "r") as f:
            mapillary_config = json.load(f)
        
        # クラスリストの作成 - オリジナルLISA実装と同様に
        mapillary_classes = [x["readable"].lower() for x in mapillary_config["labels"]]
        mapillary_classes = np.array(mapillary_classes)
        
        # ラベルファイルのパスを取得
        labels_dir = os.path.join(mapillary_path, "training", "v2.0", "labels")
        if not os.path.exists(labels_dir):
            print(f"警告: Mapillaryラベルディレクトリが見つかりません: {labels_dir}")
            return [], [], []
            
        mapillary_labels = sorted(
            glob.glob(os.path.join(labels_dir, "*.png"))
        )
        
        if len(mapillary_labels) == 0:
            print(f"警告: Mapillaryラベルファイルが見つかりません: {labels_dir}/*.png")
            return [], [], []
        
        # 画像ファイルのパスを生成 - ここを修正
        # 正しいパス: "H:\download\LISA-dataset\dataset\mapillary\training\images"
        images_dir = os.path.join(mapillary_path, "training", "images")
        if not os.path.exists(images_dir):
            print(f"警告: Mapillary画像ディレクトリが見つかりません: {images_dir}")
            return [], [], []
            
        mapillary_images = []
        valid_mapillary_labels = []
        
        for label_path in mapillary_labels:
            # ファイル名を抽出
            basename = os.path.basename(label_path)
            # 拡張子を.pngから.jpgに変更
            image_basename = basename.replace(".png", ".jpg")
            # 正しい画像パスを構築
            image_path = os.path.join(images_dir, image_basename)
            
            if os.path.exists(image_path):
                mapillary_images.append(image_path)
                valid_mapillary_labels.append(label_path)
            else:
                print(f"警告: 画像が見つかりません: {image_path}")
        
        print(f"Mapillary: 有効な画像数={len(mapillary_images)}, ラベル数={len(valid_mapillary_labels)}")
        
        if len(mapillary_images) == 0:
            print(f"警告: 有効なMapillary画像がありません")
            return [], [], []
            
        return mapillary_classes, mapillary_images, valid_mapillary_labels
        
    except Exception as e:
        print(f"Mapillaryデータセット初期化エラー: {e}")
        return [], [], []


def init_ade20k(base_image_dir):
    """ADE20Kデータセットの初期化
    
    Args:
        base_image_dir: ベースとなる画像ディレクトリ
        
    Returns:
        ADE20Kクラスのリスト、画像パスのリスト、ラベルパスのリスト
    """
    ade_path = os.path.join(base_image_dir, "ade20k")
    
    if not os.path.exists(ade_path):
        print(f"警告: ADE20Kディレクトリが見つかりません: {ade_path}")
        return [], [], []
    
    try:
        # クラスリストの作成 - objectInfo150.txtから直接読み込む
        objectinfo_path = os.path.join(ade_path, "objectInfo150.txt")
        if os.path.exists(objectinfo_path):
            with open(objectinfo_path, "r") as f:
                lines = f.readlines()
            
            # クラス名のリストを抽出
            ade_classes = []
            for line in lines[1:]:  # ヘッダー行をスキップ
                parts = line.strip().split('\t')
                if len(parts) >= 4:  # フォーマット: idx Idx name npic
                    name = parts[2].strip().lower()
                    ade_classes.append(name)
        else:
            print(f"警告: {objectinfo_path} が見つかりません。デフォルトのクラスリストを使用します。")
            # デフォルトのクラスリスト（オリジナルLISAのade20k_classes.jsonと同等）
            ade_classes = [
                "wall", "building", "sky", "floor", "tree", "ceiling", "road",
                "bed", "windowpane", "grass", "cabinet", "sidewalk",
                "person", "earth", "door", "table", "mountain", "plant",
                "curtain", "chair", "car", "water", "painting", "sofa",
                "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
                "seat", "fence", "desk", "rock", "wardrobe", "lamp",
                "bathtub", "railing", "cushion", "base", "box", "column",
                "signboard", "chest of drawers", "counter", "sand", "sink",
                "skyscraper", "fireplace", "refrigerator", "grandstand",
                "path", "stairs", "runway", "case", "pool table", "pillow",
                "screen door", "stairway", "river", "bridge", "bookcase",
                "blind", "coffee table", "toilet", "flower", "book", "hill",
                "bench", "countertop", "stove", "palm", "kitchen island",
                "computer", "swivel chair", "boat", "bar", "arcade machine",
                "hovel", "bus", "towel", "light", "truck", "tower",
                "chandelier", "awning", "streetlight", "booth",
                "television receiver", "airplane", "dirt track", "apparel",
                "pole", "land", "bannister", "escalator", "ottoman", "bottle",
                "buffet", "poster", "stage", "van", "ship", "fountain",
                "conveyer belt", "canopy", "washer", "plaything",
                "swimming pool", "stool", "barrel", "basket", "waterfall",
                "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
                "step", "tank", "trade name", "microwave", "pot", "animal",
                "bicycle", "lake", "dishwasher", "screen", "blanket",
                "sculpture", "hood", "sconce", "vase", "traffic light",
                "tray", "ashcan", "fan", "pier", "crt screen", "plate",
                "monitor", "bulletin board", "shower", "radiator", "glass",
                "clock", "flag"
            ]
        
        # numpyの配列に変換（オリジナルLISA実装と同様に）
        ade_classes = np.array(ade_classes)
        
        # 画像パスの検索
        # オリジナルでは training/ サブディレクトリを使用していたが、
        # 現在のデータセット構造に合わせて調整
        images_root = os.path.join(ade_path, "images")
        
        if not os.path.exists(images_root):
            print(f"警告: ADE20K画像ディレクトリが見つかりません: {images_root}")
            return [], [], []
            
        training_dir = os.path.join(images_root, "training")
        
        if os.path.exists(training_dir):
            # オリジナル構造がある場合
            image_ids = sorted([x for x in os.listdir(training_dir) if x.endswith(".jpg")])
            images_dir = training_dir
        else:
            # サブディレクトリがない場合は直接imagesディレクトリを使用
            image_ids = sorted([x for x in os.listdir(images_root) if x.endswith(".jpg")])
            images_dir = images_root
        
        if len(image_ids) == 0:
            print(f"警告: ADE20K画像が見つかりません: {images_dir}")
            return [], [], []
        
        # 画像パスのリストを作成
        ade_images = []
        for image_id in image_ids:
            image_path = os.path.join(images_dir, image_id)
            if os.path.exists(image_path):
                ade_images.append(image_path)
        
        if len(ade_images) == 0:
            print(f"警告: 有効なADE20K画像がありません")
            return [], [], []
        
        # アノテーションパスの生成
        annotations_root = os.path.join(ade_path, "annotations")
        
        if not os.path.exists(annotations_root):
            print(f"警告: ADE20Kアノテーションディレクトリが見つかりません: {annotations_root}")
            return [], [], []
            
        training_annotations_dir = os.path.join(annotations_root, "training")
        
        if os.path.exists(training_annotations_dir):
            annotations_dir = training_annotations_dir
        else:
            annotations_dir = annotations_root
        
        # 画像パスからアノテーションパスを生成（オリジナルLISA実装と同様に）
        ade_labels = []
        for img_path in ade_images:
            label_path = img_path.replace(".jpg", ".png").replace(images_dir, annotations_dir)
            if os.path.exists(label_path):
                ade_labels.append(label_path)
            else:
                # ラベルが見つからない場合は対応する画像も除外
                ade_images.remove(img_path)
        
        if len(ade_labels) == 0:
            print(f"警告: 有効なADE20Kアノテーションが見つかりません")
            return [], [], []
        
        print(f"ADE20Kデータセット: {len(ade_images)}画像、{len(ade_classes)}クラス")
        return ade_classes, ade_images, ade_labels
    except Exception as e:
        print(f"ADE20Kデータセットの初期化中にエラーが発生しました: {e}")
        return [], [], []


def init_cocostuff(base_image_dir):
    """COCO-Stuffデータセットの初期化
    
    Args:
        base_image_dir: ベースとなる画像ディレクトリ
        
    Returns:
        COCO-Stuffクラスのリスト、画像パスのリスト、ラベルパスのリスト
    """
    cocostuff_path = os.path.join(base_image_dir, "cocostuff")
    
    if not os.path.exists(cocostuff_path):
        print(f"警告: COCOStuffディレクトリが見つかりません: {cocostuff_path}")
        return [], [], []
    
    try:
        # クラスリストのロード - オリジナルLISA実装と同様に
        cocostuff_classes = []
        # utils/cocostuff_classes.txtファイルからクラスを読み込む
        try:
            with open("utils/cocostuff_classes.txt", "r") as f:
                for line in f.readlines()[1:]:  # ヘッダー行をスキップ
                    cocostuff_classes.append(line.strip().split(": ")[-1])
        except FileNotFoundError:
            print("警告: utils/cocostuff_classes.txt が見つかりません。デフォルトのクラスリストを使用します。")
            # デフォルトのクラスリスト
            cocostuff_classes = ["person", "bicycle", "car", "motorcycle", "airplane", 
                                "bus", "train", "truck", "boat", "traffic light"]
        
        # クラスのリストをnumpy配列に変換（オリジナルLISA実装と同様に）
        cocostuff_classes = np.array(cocostuff_classes)
        
        # ラベルディレクトリの確認
        train_dir = os.path.join(cocostuff_path, "train2017")
        if not os.path.exists(train_dir):
            print(f"警告: COCOStuffトレーニングディレクトリが見つかりません: {train_dir}")
            return [], [], []
        
        # ラベルファイルを検索
        cocostuff_labels = glob.glob(os.path.join(train_dir, "*.png"))
        
        if len(cocostuff_labels) == 0:
            print(f"警告: COCOStuffラベルファイルが見つかりません: {train_dir}/*.png")
            return [], [], []
        
        # COCOディレクトリの確認
        coco_path = os.path.join(base_image_dir, "coco")
        coco_train_dir = os.path.join(coco_path, "train2017")
        
        if not os.path.exists(coco_train_dir):
            print(f"警告: COCOトレーニング画像ディレクトリが見つかりません: {coco_train_dir}")
            # 代替パスを試す
            coco_train_dir = os.path.join(base_image_dir, "coco", "images", "train2017")
            if not os.path.exists(coco_train_dir):
                print(f"警告: 代替COCOトレーニング画像ディレクトリも見つかりません: {coco_train_dir}")
                return [], [], []
        
        # 画像パスを生成
        valid_images = []
        valid_labels = []
        
        for label_path in cocostuff_labels:
            # ファイル名を抽出
            file_name = os.path.basename(label_path)
            file_name = file_name.replace(".png", ".jpg")
            
            # 画像パスを生成
            img_path = os.path.join(coco_train_dir, file_name)
            
            # 画像が存在する場合のみリストに追加
            if os.path.exists(img_path):
                valid_images.append(img_path)
                valid_labels.append(label_path)
            else:
                if len(valid_images) == 0:  # 最初のエラーのみ表示
                    print(f"警告: 画像が見つかりません: {img_path}")
        
        if len(valid_images) == 0:
            print("警告: 有効なCOCOStuff画像が見つかりません。ディレクトリ構造を確認してください。")
            print(f"  - COCOStuffパス: {cocostuff_path}")
            print(f"  - COCOパス: {coco_path}")
            return [], [], []
        
        print(f"COCO-Stuffデータセット: {len(valid_images)}画像、{len(cocostuff_classes)}クラス")
        return cocostuff_classes, valid_images, valid_labels
    except Exception as e:
        print(f"COCO-Stuffデータセットの初期化中にエラーが発生しました: {e}")
        return [], [], []


def init_paco(base_image_dir):
    """PACO LVISデータセットの初期化
    
    Args:
        base_image_dir: ベースとなる画像ディレクトリ
        
    Returns:
        PACOクラスと画像のリスト
    """
    # 複数の可能性のあるパスをチェック
    possible_annotation_paths = [
        os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"),  # オリジナルLISAパス
        os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1", "paco_lvis_v1_train.json"),  # サブディレクトリ内
        os.path.join(base_image_dir, "paco", "annotations", "paco_lvis_v1_train.json"),  # ベースディレクトリ直下
        os.path.join(base_image_dir, "paco_lvis", "annotations", "paco_lvis_v1_train.json"),  # 別名のディレクトリ
    ]
    
    # 存在するパスを見つける
    annotation_path = None
    for path in possible_annotation_paths:
        if os.path.exists(path):
            annotation_path = path
            print(f"PACO: アノテーションファイルが見つかりました: {path}")
            break
    
    if annotation_path is None:
        print(f"警告: PACO LVISアノテーションファイルが見つかりません。")
        print(f"  - 検索したパス: {possible_annotation_paths}")
        return {}, []
    
    # アノテーションのロード
    try:
        with open(annotation_path, "r") as f:
            lvis_data = json.load(f)
        
        paco_classes = {}
        for category in lvis_data["categories"]:
            paco_classes[category["name"].lower()] = category["id"]
        
        # 画像のマッピング
        images_map = {}
        for image in lvis_data["images"]:
            images_map[image["id"]] = image["coco_url"].split("/")[-1]
        
        # 有効な画像IDを取得
        valid_image_ids = []
        for annotation in lvis_data["annotations"]:
            if annotation["image_id"] not in valid_image_ids:
                valid_image_ids.append(annotation["image_id"])
        
        # 可能性のあるCOCO画像ディレクトリをチェック
        possible_coco_dirs = [
            os.path.join(base_image_dir, "coco", "train2017"),
            os.path.join(base_image_dir, "coco", "images", "train2017"),
            os.path.join(base_image_dir, "vlpart", "coco", "train2017"),
            os.path.join(base_image_dir, "vlpart", "coco", "images", "train2017"),
        ]
        
        coco_train_dir = None
        for coco_dir in possible_coco_dirs:
            if os.path.exists(coco_dir):
                coco_train_dir = coco_dir
                print(f"PACO: COCO画像ディレクトリが見つかりました: {coco_dir}")
                break
        
        if coco_train_dir is None:
            print(f"警告: COCO画像ディレクトリが見つかりません")
            for path in possible_coco_dirs:
                print(f"  - 検索したパス: {path}")
            return paco_classes, []
        
        # 学習用画像
        train_images = []
        missing_images = 0
        
        for image_id in valid_image_ids:
            if image_id not in images_map:
                continue
                
            image_name = images_map[image_id]
            image_path = os.path.join(coco_train_dir, image_name)
            
            # 画像が存在する場合のみリストに追加
            if os.path.exists(image_path):
                train_images.append(image_path)
            else:
                missing_images += 1
                # 最初の5件のみ警告を表示
                if missing_images <= 5:
                    print(f"警告: 画像が見つかりません: {image_path}")
        
        if missing_images > 5:
            print(f"...他 {missing_images - 5} 件の画像が見つかりません")
        
        if len(train_images) == 0:
            print(f"警告: 有効なPACO LVIS画像が見つかりませんでした")
            return paco_classes, []
            
        print(f"PACO LVISデータセット: {len(train_images)}画像、{len(paco_classes)}クラス")
        return paco_classes, train_images
        
    except Exception as e:
        print(f"PACO LVISデータセットの初期化中にエラーが発生しました: {e}")
        import traceback
        print(traceback.format_exc())
        return {}, []


def init_pascal_part(base_image_dir):
    """Pascal Partデータセットの初期化
    
    Args:
        base_image_dir: ベースとなる画像ディレクトリ
        
    Returns:
        Pascal Partクラスと画像のリスト
    """
    pascal_path = os.path.join(base_image_dir, "vlpart", "pascal_part")
    train_json_path = os.path.join(pascal_path, "train.json")
    
    # アノテーションファイルの存在確認
    if not os.path.exists(train_json_path):
        print(f"警告: Pascal Part アノテーションファイルが見つかりません: {train_json_path}")
        return {}, []
    
    try:
        # アノテーションのロード
        with open(train_json_path, "r") as f:
            pascal_data = json.load(f)
        
        pascal_classes = {}
        for category in pascal_data["categories"]:
            pascal_classes[category["name"].lower()] = category["id"]
        
        # 画像のパス
        voc_path = os.path.join(pascal_path, "VOCdevkit", "VOC2010")
        
        # VOCdevkitディレクトリの存在確認
        if not os.path.exists(os.path.join(voc_path, "JPEGImages")):
            print(f"警告: Pascal Part VOCdevkitディレクトリが見つかりません: {os.path.join(voc_path, 'JPEGImages')}")
            return {}, []
        
        train_images = []
        for image in pascal_data["images"]:
            image_path = os.path.join(voc_path, "JPEGImages", image["file_name"])
            # 画像が実際に存在する場合のみ追加
            if os.path.exists(image_path):
                train_images.append(image_path)
            else:
                if len(train_images) == 0:  # 最初のエラーのみ表示
                    print(f"警告: 画像が見つかりません: {image_path}")
        
        if len(train_images) == 0:
            print(f"警告: Pascal Partデータセットの有効な画像がありません")
            return {}, []
        
        print(f"Pascal Partデータセット: {len(train_images)}画像、{len(pascal_classes)}クラス")
        return pascal_classes, train_images
    except Exception as e:
        print(f"Pascal Partデータセットの初期化中にエラーが発生しました: {e}")
        return {}, []


def init_partimagenet(base_image_dir):
    """PartImageNetデータセットの初期化
    
    Args:
        base_image_dir: ベースとなる画像ディレクトリ
        
    Returns:
        PartImageNetクラスと画像のリスト
    """
    partimagenet_path = os.path.join(base_image_dir, "partimagenet")
    
    if not os.path.exists(partimagenet_path):
        print(f"警告: PartImageNetディレクトリが見つかりません: {partimagenet_path}")
        return {}, []
    
    train_json_path = os.path.join(partimagenet_path, "train.json")
    if not os.path.exists(train_json_path):
        print(f"警告: PartImageNet アノテーションファイルが見つかりません: {train_json_path}")
        return {}, []
    
    try:
        # アノテーションのロード
        with open(train_json_path, "r") as f:
            partimagenet_data = json.load(f)
        
        partimagenet_classes = {}
        for category in partimagenet_data["categories"]:
            partimagenet_classes[category["name"].lower()] = category["id"]
        
        # 画像ディレクトリの確認
        images_dir = os.path.join(partimagenet_path, "images")
        if not os.path.exists(images_dir):
            print(f"警告: PartImageNet画像ディレクトリが見つかりません: {images_dir}")
            return {}, []
        
        # 画像のパス
        train_images = []
        for image in partimagenet_data["images"]:
            image_path = os.path.join(partimagenet_path, "images", image["file_name"])
            # 画像が実際に存在する場合のみ追加
            if os.path.exists(image_path):
                train_images.append(image_path)
            else:
                if len(train_images) == 0:  # 最初のエラーのみ表示
                    print(f"警告: 画像が見つかりません: {image_path}")
        
        if len(train_images) == 0:
            print(f"警告: PartImageNetデータセットの有効な画像がありません")
            return {}, []
        
        print(f"PartImageNetデータセット: {len(train_images)}画像、{len(partimagenet_classes)}クラス")
        return partimagenet_classes, train_images
    except Exception as e:
        print(f"PartImageNetデータセットの初期化中にエラーが発生しました: {e}")
        return {}, []


class SemSegDataset(torch.utils.data.Dataset):
    """セマンティックセグメンテーションデータセット"""
    
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        model_name,  # Gemma3モデル名
        samples_per_epoch=500 * 8 * 2 * 10,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        exclude_val=False,
        sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",  # オリジナルLISAと同様に全データセットを使用
        debug_mode=False,  # デバッグモードフラグ
        debug_samples=10,  # デバッグモードで使用するサンプル数
        processor=None,  # 親クラスから渡されるプロセッサ
        image_processor=None,  # 親クラスから渡される画像プロセッサ
    ):
        """初期化
        
        Args:
            base_image_dir: ベースとなる画像ディレクトリ
            tokenizer: トークナイザ
            model_name: Gemma3モデル名
            samples_per_epoch: エポックあたりのサンプル数
            precision: 精度
            image_size: 画像サイズ
            num_classes_per_sample: サンプルあたりのクラス数
            exclude_val: 検証データを除外するか
            sem_seg_data: セマンティックセグメンテーションデータ
            debug_mode: デバッグモードかどうか
            debug_samples: デバッグモードで使用するサンプル数
            processor: 親クラスから渡されるプロセッサ
            image_processor: 親クラスから渡される画像プロセッサ
            
        Raises:
            RuntimeError: 指定されたデータセットのいずれかが完全でない場合
        """
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.precision = precision
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.exclude_val = exclude_val
        self.debug_mode = debug_mode
        self.debug_samples = debug_samples
        
        # デバッグモードのログ
        if self.debug_mode:
            print(f"SemSegDataset: デバッグモード有効 - 各データセットの最初の{self.debug_samples}例のみを使用")
        
        # データセットの初期化（カンマ区切りとパイプ区切りの両方に対応）
        if isinstance(sem_seg_data, str):
            if "||" in sem_seg_data:
                self.sem_seg_datas = sem_seg_data.split("||")
            elif "," in sem_seg_data:
                self.sem_seg_datas = sem_seg_data.split(",")
            else:
                self.sem_seg_datas = [sem_seg_data] if sem_seg_data else []
        else:
            self.sem_seg_datas = sem_seg_data if isinstance(sem_seg_data, list) else []
        
        print(f"セマンティックセグメンテーションデータセット: {self.sem_seg_datas}")
        
        self.data2list = {}
        self.data2classes = {}
        
        # データセットの検証結果を記録
        dataset_status = {}
        valid_datasets = []
        invalid_datasets = []
        
        # 各データセットを初期化し、状態を検証
        for ds in self.sem_seg_datas:
            dataset_valid = False
            error_message = ""
            
            try:
                if ds == "ade20k":
                    ade_classes, ade_images, ade_labels = init_ade20k(base_image_dir)
                    if len(ade_images) > 0 and len(ade_labels) > 0:
                        self.data2classes[ds] = ade_classes
                        
                        # デバッグモードの場合は最初のdebug_samples個のサンプルのみを使用
                        if self.debug_mode:
                            ade_images = ade_images[:self.debug_samples]
                            ade_labels = ade_labels[:self.debug_samples]
                            print(f"ADE20K: {len(ade_images)}例のみを使用")
                        
                        self.data2list[ds] = (ade_images, ade_labels)
                        dataset_valid = True
                    else:
                        error_message = f"ADE20Kデータセットが不完全です。画像数: {len(ade_images)}, ラベル数: {len(ade_labels)}"
                
                elif ds == "cocostuff":
                    cocostuff_classes, cocostuff_images, cocostuff_labels = init_cocostuff(base_image_dir)
                    if len(cocostuff_images) > 0 and len(cocostuff_labels) > 0:
                        self.data2classes[ds] = cocostuff_classes
                        
                        # デバッグモードの場合は最初のdebug_samples個のサンプルのみを使用
                        if self.debug_mode:
                            cocostuff_images = cocostuff_images[:self.debug_samples]
                            cocostuff_labels = cocostuff_labels[:self.debug_samples]
                            print(f"COCO-Stuff: {len(cocostuff_images)}例のみを使用")
                        
                        self.data2list[ds] = (cocostuff_images, cocostuff_labels)
                        
                        # クラス名からインデックスへのマッピングを作成（オリジナル実装と同様に）
                        self.cocostuff_class2index = {
                            c: i for i, c in enumerate(self.data2classes["cocostuff"])
                        }
                        dataset_valid = True
                    else:
                        error_message = f"COCO-Stuffデータセットが不完全です。画像数: {len(cocostuff_images)}, ラベル数: {len(cocostuff_labels)}"
                
                elif ds == "mapillary":
                    mapillary_classes, mapillary_images, mapillary_labels = init_mapillary(base_image_dir)
                    if len(mapillary_images) > 0 and len(mapillary_labels) > 0:
                        self.data2classes[ds] = mapillary_classes
                        
                        # デバッグモードの場合は最初のdebug_samples個のサンプルのみを使用
                        if self.debug_mode:
                            mapillary_images = mapillary_images[:self.debug_samples]
                            mapillary_labels = mapillary_labels[:self.debug_samples]
                            print(f"Mapillary: {len(mapillary_images)}例のみを使用")
                        
                        self.data2list[ds] = (mapillary_images, mapillary_labels)
                        dataset_valid = True
                    else:
                        error_message = f"Mapillaryデータセットが不完全です。画像数: {len(mapillary_images)}, ラベル数: {len(mapillary_labels)}"
                
                elif ds == "pascal_part":
                    pascal_classes, pascal_images = init_pascal_part(base_image_dir)
                    if len(pascal_images) > 0 and len(pascal_classes) > 0:
                        self.data2classes[ds] = pascal_classes
                        
                        # デバッグモードの場合は最初のdebug_samples個のサンプルのみを使用
                        if self.debug_mode:
                            pascal_images = pascal_images[:self.debug_samples]
                            print(f"Pascal Part: {len(pascal_images)}例のみを使用")
                        
                        self.data2list[ds] = (pascal_images, None)
                        dataset_valid = True
                    else:
                        error_message = f"Pascal Partデータセットが不完全です。画像数: {len(pascal_images)}, クラス数: {len(pascal_classes)}"
                
                elif ds == "paco_lvis":
                    paco_classes, paco_images = init_paco(base_image_dir)
                    if len(paco_images) > 0 and len(paco_classes) > 0:
                        self.data2classes[ds] = paco_classes
                        
                        # デバッグモードの場合は最初のdebug_samples個のサンプルのみを使用
                        if self.debug_mode:
                            paco_images = paco_images[:self.debug_samples]
                            print(f"PACO LVIS: {len(paco_images)}例のみを使用")
                        
                        self.data2list[ds] = (paco_images, None)
                        dataset_valid = True
                    else:
                        error_message = f"PACO LVISデータセットが不完全です。画像数: {len(paco_images)}, クラス数: {len(paco_classes)}"
            
            except Exception as e:
                error_message = f"{ds}データセットの初期化中にエラーが発生しました: {e}"
            
            # データセットの状態を記録
            dataset_status[ds] = {
                "valid": dataset_valid,
                "error": error_message
            }
            
            if dataset_valid:
                valid_datasets.append(ds)
            else:
                invalid_datasets.append(ds)
        
        # すべてのデータセットが無効な場合はエラーをスロー
        if len(valid_datasets) == 0:
            error_msgs = [f"{ds}: {dataset_status[ds]['error']}" for ds in invalid_datasets]
            raise RuntimeError(f"すべてのデータセットが無効です。少なくとも1つの有効なデータセットが必要です。\n" + "\n".join(error_msgs))
        
        # 無効なデータセットがある場合は警告を表示
        if len(invalid_datasets) > 0:
            print("警告: 一部のデータセットが無効です。")
            for ds in invalid_datasets:
                print(f"  - {ds}: {dataset_status[ds]['error']}")
            
            # 有効なデータセットのみを使用するように設定を更新
            self.sem_seg_datas = valid_datasets
            print(f"有効なデータセットのみを使用します: {', '.join(valid_datasets)}")
        
        # 使用するデータセットの有効性を最終確認
        print("=== 使用するデータセット ===")
        for ds in self.sem_seg_datas:
            if ds in self.data2list:
                if ds in ["paco_lvis", "pascal_part"]:
                    print(f"  - {ds}: {len(self.data2list[ds][0])}画像, {len(self.data2classes[ds])}クラス")
                else:
                    print(f"  - {ds}: {len(self.data2list[ds][0])}画像, {len(self.data2list[ds][1])}ラベル, {len(self.data2classes[ds])}クラス")
        
        # Gemma3用の変換処理
        self.processor = processor
        self.image_processor = image_processor

        # 親クラスから渡されたプロセッサを使用
        if processor is not None and image_processor is not None:
            print(f"SemSegDataset: 親クラスから渡されたプロセッサを使用します")
        else:
            # プロセッサが渡されなかった場合は独自に初期化（後方互換性用）
            try:
                # transformersから直接AutoImageProcessorとAutoProcessorを使用
                from transformers import AutoProcessor, AutoImageProcessor
                
                # モデル名が明示的に指定されている場合は、それを使用
                if model_name == "google/gemma-3-4b-it" or "gemma-3" in model_name:
                    # Gemma3のVisionプロセッサを初期化
                    self.processor = AutoProcessor.from_pretrained(
                        "google/gemma-3-4b-it",  # 明示的に正確なモデル名を指定
                        trust_remote_code=True
                    )
                    
                    # GemmaImageProcessorを初期化
                    from model.gemma3.mm_utils import GemmaImageProcessor
                    self.image_processor = GemmaImageProcessor(self.processor)
                    
                    print(f"SemSegDataset: Gemma3プロセッサを初期化: google/gemma-3-4b-it")
                else:
                    # モデル名が指定されている場合は、それを使用
                    self.processor = AutoProcessor.from_pretrained(
                        model_name, 
                        trust_remote_code=True
                    )
                    print(f"SemSegDataset: 指定されたモデルプロセッサを初期化: {model_name}")
                
                # バックアップとしてAutoImageProcessorも取得（エラーチェックなし）
                self.image_processor_fallback = AutoImageProcessor.from_pretrained(
                    "openai/clip-vit-large-patch14",  # CLIPの画像処理器を使用
                    trust_remote_code=True
                )
            except Exception as e:
                # フォールバックを削除し、エラーを再発生させる
                error_msg = f"SemSegDataset: プロセッサの初期化エラー: {e}"
                print(error_msg)
                raise RuntimeError(error_msg) from e
        
        # 標準的な画像処理のための平均・標準偏差
        self.image_mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.image_std = np.array([0.26862954, 0.26130258, 0.27577711])
        
        # SAM用の変換処理
        self.sam_transform = ResizeLongestSide(self.img_size)
        
        # 質問と回答テンプレート
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        
        # 統一された会話長を確保するため
        self.max_question_tokens = 20  # 質問の最大トークン数の目安
        self.max_answer_tokens = 10    # 回答の最大トークン数の目安
    
    def __len__(self):
        return self.samples_per_epoch
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """SAM用に正規化して前処理する"""
        # 正規化
        x = (x - self.pixel_mean) / self.pixel_std
        
        # パディング
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def process_image_for_gemma(self, image):
        """Gemma3用に画像を処理する安全な方法
        
        Args:
            image: numpy配列の画像データ
            
        Returns:
            処理済み画像テンソル
        """
        # 画像が有効かチェック
        if image is None:
            print("警告: 画像がNoneです。ゼロテンソルを返します。")
            return torch.zeros(3, self.image_size, self.image_size)
            
        # 有効なGemmaImageProcessorを使用してシンプルに処理
        if hasattr(self, "image_processor") and self.image_processor is not None:
            try:
                # GemmaImageProcessorのコールは直接テンソルを返す
                tensor = self.image_processor(image)
                if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
                    return tensor
            except Exception as e:
                print(f"GemmaImageProcessor処理エラー: {e}")
                # 次の方法に進む
        
        # 手動での画像前処理（最終手段）- シンプルで明確な処理
        try:
            # RGBであることを確認
            if isinstance(image, np.ndarray) and image.shape[2] == 3:
                # リサイズ
                h, w = image.shape[:2]
                size = self.image_size
                resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
                
                # 浮動小数点に変換して正規化 (0-1)
                normalized = resized.astype(np.float32) / 255.0
                
                # CLIP標準の平均と標準偏差で正規化
                mean = self.image_mean.reshape(1, 1, 3)
                std = self.image_std.reshape(1, 1, 3)
                normalized = (normalized - mean) / std
                
                # PyTorchテンソルに変換 (C, H, W)形式
                tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
                return tensor
            else:
                print("警告: 画像が正しいRGB形式ではありません")
                return torch.zeros(3, self.image_size, self.image_size)
                
        except Exception as e:
            print(f"画像処理中のエラー: {e}")
            # 最終手段: ゼロテンソルを返す
            return torch.zeros(3, self.image_size, self.image_size)

    def create_conversation(self, question, answer, use_seg=True):
        """一貫した会話形式でプロンプトを生成
        
        Args:
            question: ユーザーの質問
            answer: アシスタントの回答
            use_seg: [SEG]トークンを含めるか
            
        Returns:
            会話プロンプト
        """
        # Gemma3用のテンプレート取得
        template = get_default_conv_template("gemma_v1")
        template.system = SYSTEM_PROMPT
        template.messages = []
        
        # 質問が空でないことを確認
        if not question or question.strip() == "":
            question = "What do you see in this image? Please output segmentation mask."
        
        # 回答が空でないことを確認
        if not answer or answer.strip() == "":
            answer = "I can see objects in the image. [SEG]."
        
        # 画像トークンがすでに含まれているか確認
        if DEFAULT_IMAGE_TOKEN not in question:
            question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        
        # ユーザーからの質問を追加
        template.append_message(template.roles[0], question)
        
        # アシスタントの回答を追加（[SEG]トークンを含める場合）
        if use_seg and "[SEG]" not in answer:
            answer = f"{answer} [SEG]."
        
        template.append_message(template.roles[1], answer)
        
        try:
            # プロンプトを取得
            prompt = template.get_prompt()
            return prompt
        except Exception as e:
            print(f"会話テンプレート生成エラー: {e}")
            # エラー時はフォールバックの会話を返す
            return f"System: {SYSTEM_PROMPT}\n\nUser: {DEFAULT_IMAGE_TOKEN}\nWhat is in this image? Please output segmentation mask.\n\nAssistant: I can see objects in this image. [SEG]."

    def __getitem__(self, idx):
        """データセットからアイテムを取得"""
        # ランダムにデータセットを選択
        ds_idx = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds_idx]
        
        if ds in ["paco_lvis", "pascal_part"]:
            # パーツセグメンテーション用処理（PACO LVISとPascal Part）
            # VLPart関連のデータはオリジナルとは異なる処理が必要
            images = self.data2list[ds][0]
            if len(images) == 0:
                return self.__getitem__(0)  # 空の場合、再帰的に呼び出し
                
            # ランダムに画像を選択
            image_path = random.choice(images)
            image = cv2.imread(image_path)
            if image is None:
                print(f"画像の読み込みに失敗: {image_path}")
                return self.__getitem__(0)
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # クラスをランダムに選択
            class_names = []
            if isinstance(self.data2classes[ds], dict):
                class_names = list(self.data2classes[ds].keys())
            
            if len(class_names) >= self.num_classes_per_sample:
                sampled_classes = random.sample(class_names, self.num_classes_per_sample)
            else:
                sampled_classes = class_names
            
        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            # セマンティックセグメンテーション用処理（ADE20K、COCO-Stuff、Mapillary）
            images, labels = self.data2list[ds]
            if len(images) == 0:
                return self.__getitem__(0)  # 空の場合、再帰的に呼び出し
                
            # ランダムに画像とラベルを選択
            idx = random.randint(0, len(images) - 1)
            image_path = images[idx]
            label_path = labels[idx]
            
            # 画像とラベルを読み込む
            image = cv2.imread(image_path)
            if image is None:
                print(f"画像の読み込みに失敗: {image_path}")
                return self.__getitem__(0)
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            try:
                label = Image.open(label_path)
                label = np.array(label)
            except Exception as e:
                print(f"ラベルの読み込みに失敗: {label_path} - {e}")
                return self.__getitem__(0)
            
            # データセット固有の処理
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                # COCOStuffデータセット固有の処理（ハイフンを含むクラスを無視）
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            
            # 使用可能なクラス（ラベル）を取得
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            
            # 有効なラベルがない場合は再試行
            if len(unique_label) == 0:
                return self.__getitem__(0)
            
            # 使用可能なクラスを取得
            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            
            # 指定された数だけランダムにクラスを選択
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes
        
        # 質問と回答を生成
        questions = []
        answers = []
        class_ids = []
        
        for sampled_cls in sampled_classes:
            # テキスト処理
            if isinstance(sampled_cls, tuple):
                obj, part = sampled_cls
                if random.random() < 0.5:
                    text = obj + " " + part
                else:
                    text = "the {} of the {}".format(part, obj)
            else:
                text = sampled_cls
            
            # 質問テンプレートからランダムに選択
            question_template = random.choice(self.short_question_list)
            question = question_template.format(class_name=text.lower())
            questions.append(question)
            
            # 回答テンプレートからランダムに選択
            answers.append(random.choice(self.answer_list))
            
            # クラスIDを保存（PACO LVISとPascal Partを除く）
            if ds not in ["paco_lvis", "pascal_part"] and isinstance(self.data2classes[ds], np.ndarray):
                class_id = self.data2classes[ds].tolist().index(sampled_cls)
                class_ids.append(class_id)
        
        # 会話を生成（統一された長さになるよう調整）
        conversations = []
        for question, answer in zip(questions, answers):
            # 会話テンプレートを使用して一貫したフォーマットを確保
            prompt = self.create_conversation(question, answer)
            conversations.append(prompt)
        
        # 画像を処理 - 安全な画像処理メソッドを使用
        images_gemma = self.process_image_for_gemma(image)
            
        # SAM用の高解像度画像処理
        image_sam = self.sam_transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        
        # マスクの処理
        if ds in ["paco_lvis", "pascal_part"]:
            # VLPart関連のデータセットでは、マスクをダミーとして作成
            masks = torch.zeros(len(sampled_classes), *resize)
            label = torch.ones(resize) * self.ignore_label
        else:
            # セマンティックセグメンテーションデータセットでは、クラスIDに基づいてマスクを作成
            label_tensor = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label_tensor == class_id)
            if masks:
                masks = torch.stack(masks, dim=0)
            else:
                masks = torch.zeros(len(sampled_classes), *resize)
            
            label = label_tensor
        
        # 推論フラグと質問のプレースホルダー
        inference = False
        
        return (
            image_path,
            image_sam,
            images_gemma,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            inference,
        )
