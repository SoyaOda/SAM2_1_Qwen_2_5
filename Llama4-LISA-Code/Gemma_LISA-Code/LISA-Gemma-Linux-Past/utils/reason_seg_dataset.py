import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from pycocotools import mask

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

from .data_processing import get_mask_from_json
from .constants import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
                       DEFAULT_IM_END_TOKEN, SHORT_QUESTION_LIST, SYSTEM_PROMPT)
from .conversation import get_default_conv_template

# 説明的な質問リスト
EXPLANATORY_QUESTION_LIST = [
    "Can you explain why this is {cls}?",
    "How do you recognize this as {cls}?",
    "What are the characteristics of {cls} in this image?",
    "What visual features identify this as {cls}?",
    "Why do you think this is {cls}?",
]

# 長い質問リスト
LONG_QUESTION_LIST = [
    "Can you provide a detailed explanation of why this region contains {cls}?",
    "Please explain in detail how you determined that this area shows {cls}.",
    "I'd like to understand the visual cues that led you to identify {cls} here. Can you elaborate?",
    "What specific features in this region indicate that this is {cls}? Please provide details.",
    "Could you give a comprehensive explanation of why you've identified this as {cls}?",
]

class ReasonSegDataset(torch.utils.data.Dataset):
    """理由付きセグメンテーションデータセット"""
    
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
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
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
            reason_seg_data: 理由付きセグメンテーションデータ
            explanatory: 説明付きデータの割合
            processor: 親クラスから渡されるプロセッサ
            image_processor: 親クラスから渡される画像プロセッサ
        """
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.precision = precision
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.explanatory = explanatory
        self.exclude_val = exclude_val
        
        # データセットの初期化 (複数データセット対応)
        self.reason_seg_data_list = []
        if isinstance(reason_seg_data, str):
            if "||" in reason_seg_data:
                datasets = reason_seg_data.split("||")
            elif "," in reason_seg_data:
                datasets = reason_seg_data.split(",")
            else:
                datasets = [reason_seg_data]
                
            for ds in datasets:
                self.reason_seg_data_list.append(ds)
        else:
            self.reason_seg_data_list = reason_seg_data if isinstance(reason_seg_data, list) else []
        
        print(f"理由付きセグメンテーションデータセット: {self.reason_seg_data_list}")
        
        # 画像とJSONのパスを収集
        self.images = []
        self.jsons = []
        
        for reason_seg_item in self.reason_seg_data_list:
            # データセット名とスプリットを分割
            parts = reason_seg_item.split("|")
            dataset_name = parts[0]
            
            if len(parts) > 1:
                # スプリットが指定されている場合
                splits_str = parts[1]
                
                # オリジナルLISAと同様に、スプリットを'_'で分割して複数のスプリットに対応
                splits = splits_str.split("_")
                
                for split in splits:
                    # 各スプリットの画像を読み込む
                    images_split = glob.glob(
                        os.path.join(
                            self.base_image_dir, dataset_name, split, "*.jpg"
                        )
                    )
                    self.images.extend(images_split)
                    
                    # 対応するJSONファイルのパスを生成
                    jsons_split = [path.replace(".jpg", ".json") for path in images_split]
                    self.jsons.extend(jsons_split)
                    
                    print(f"読み込み: {dataset_name}/{split} - {len(images_split)}画像")
            else:
                # スプリットが指定されていない場合、trainを使用
                images_split = glob.glob(
                    os.path.join(
                        self.base_image_dir, dataset_name, "train", "*.jpg"
                    )
                )
                self.images.extend(images_split)
                
                # 対応するJSONファイルのパスを生成
                jsons_split = [path.replace(".jpg", ".json") for path in images_split]
                self.jsons.extend(jsons_split)
                
                print(f"読み込み: {dataset_name}/train - {len(images_split)}画像")
        
        # 有効なJSONファイルをフィルタリング
        valid_images = []
        valid_jsons = []
        for img_path, json_path in zip(self.images, self.jsons):
            if os.path.exists(json_path):
                valid_images.append(img_path)
                valid_jsons.append(json_path)
        
        self.images = valid_images
        self.jsons = valid_jsons
        
        print(f"理由付きセグメンテーションデータセット: 合計画像数 = {len(self.images)}")
        
        # インデックスエラーを防ぐチェック
        if len(self.images) == 0:
            print("警告: 理由付きセグメンテーションデータセットが空です。データパスを確認してください。")
            print(f"検索パス: {os.path.join(self.base_image_dir, dataset_name)}")
            
            # フォールバック: explanatoryデータを使用
            if explanatory > 0:
                explanatory_dir = os.path.join(self.base_image_dir, "ReasonSeg", "explanatory")
                if os.path.exists(explanatory_dir):
                    fallback_images = glob.glob(os.path.join(explanatory_dir, "*.jpg"))
                    fallback_jsons = [path.replace(".jpg", ".json") for path in fallback_images]
                    
                    valid_fallbacks = []
                    valid_fallback_jsons = []
                    for img, js in zip(fallback_images, fallback_jsons):
                        if os.path.exists(js):
                            valid_fallbacks.append(img)
                            valid_fallback_jsons.append(js)
                    
                    if valid_fallbacks:
                        print(f"フォールバック: explanatoryディレクトリから{len(valid_fallbacks)}画像を読み込みました")
                        self.images = valid_fallbacks
                        self.jsons = valid_fallback_jsons
        
        # 説明付き応答用のサンプル
        self.explanatory_images = []
        self.img_to_explanation = {}
        
        explanatory_json_path = os.path.join(self.base_image_dir, "ReasonSeg", "explanatory", "train.json")
        if os.path.exists(explanatory_json_path):
            try:
                with open(explanatory_json_path) as f:
                    items = json.load(f)
                
                # 画像ファイルが実際に存在するディレクトリのリスト
                possible_image_dirs = [
                    os.path.join(self.base_image_dir, "ReasonSeg", "train"),
                    os.path.join(self.base_image_dir, "ReasonSeg", "val"),
                    os.path.join(self.base_image_dir, "ReasonSeg", "explanatory"),
                ]
                
                # 画像ファイルの実際の場所を検索するヘルパー関数
                def find_image_file(img_name):
                    for dir_path in possible_image_dirs:
                        img_path = os.path.join(dir_path, img_name)
                        if os.path.exists(img_path):
                            return img_path
                    return None
                
                valid_items = 0
                for item in items:
                    img_name = item["image"]
                    
                    # 画像ファイルの実際の場所を検索
                    img_path = find_image_file(img_name)
                    
                    if img_path:
                        self.img_to_explanation[img_name] = {
                            "query": item["query"],
                            "outputs": item["outputs"],
                            "path": img_path  # 実際の画像パスを保存
                        }
                        self.explanatory_images.append(img_path)
                        valid_items += 1
                
                print(f"説明付きデータセット: {valid_items}画像 (JSONから読み込み、有効な画像のみ)")
            except Exception as e:
                print(f"説明付きデータセットの読み込みエラー: {e}")
        
        # 画像ファイルからの読み込みはJSONから読み込めなかった場合のフォールバック
        if len(self.img_to_explanation) == 0 and os.path.exists(os.path.join(self.base_image_dir, "ReasonSeg", "explanatory")):
            self.explanatory_images = list(
                glob.glob(
                    os.path.join(self.base_image_dir, "ReasonSeg", "explanatory", "*.jpg")
                )
            )
            print(f"説明付きデータセット: {len(self.explanatory_images)}画像 (画像ファイルから読み込み)")
        
        # 親クラスから渡されたプロセッサを使用
        if processor is not None and image_processor is not None:
            self.processor = processor
            self.image_processor = image_processor
            print(f"ReasonSegDataset: 親クラスから渡されたプロセッサを使用します")
        else:
            # プロセッサが渡されなかった場合は独自に初期化（後方互換性用）
            from model.gemma3.mm_utils import get_gemma_processor, GemmaImageProcessor
            
            try:
                # モデル名からプロセッサを取得
                self.processor = get_gemma_processor(model_name)
                self.image_processor = GemmaImageProcessor(self.processor)
                print(f"ReasonSegDataset: Gemma3プロセッサを初期化: {model_name}")
            except Exception as e:
                error_msg = f"ReasonSegDataset: プロセッサの初期化に失敗しました: {e}"
                print(error_msg)
                # 例外を再発生させて親クラスにエラーを伝播
                raise RuntimeError(error_msg) from e
        
        # SAM用の変換
        self.sam_transform = ResizeLongestSide(self.img_size)
        
        # 質問タイプを準備
        self.questions = SHORT_QUESTION_LIST
        self.long_questions = LONG_QUESTION_LIST
        self.explanatory_questions = EXPLANATORY_QUESTION_LIST
    
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
    
    def __getitem__(self, idx):
        """データセットからアイテムを取得"""
        # 説明付きデータを使用するかを決定
        use_explanatory = random.random() < self.explanatory and len(self.img_to_explanation) > 0
        
        if len(self.images) == 0:
            raise IndexError("理由付きセグメンテーションデータセットが空です。データパスを確認してください。")
        
        # 画像パスの選択
        if use_explanatory and self.explanatory_images:
            # 説明付きデータセットから選択
            img_idx = random.randint(0, len(self.explanatory_images) - 1)
            image_path = self.explanatory_images[img_idx]
            json_path = image_path.replace(".jpg", ".json")
        else:
            # 通常のデータセットから選択
            img_idx = random.randint(0, len(self.images) - 1)
            image_path = self.images[img_idx]
            json_path = self.jsons[img_idx]  # 対応するJSONファイルを使用
            use_explanatory = False
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 画像の読み込みに失敗しました: {image_path}")
            # 代替の画像を使用
            if len(self.images) > 1:
                new_idx = (img_idx + 1) % len(self.images)
                image_path = self.images[new_idx]
                json_path = self.jsons[new_idx]
                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"画像の読み込みに失敗しました: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # JSONデータを読み込み
        try:
            mask_json, reason_text, is_sentence = get_mask_from_json(json_path, image)
        except Exception as e:
            print(f"JSONデータの読み込みに失敗しました: {json_path}, エラー: {e}")
            # 代替のJSONを使用
            if len(self.images) > 1:
                new_idx = (img_idx + 1) % len(self.images)
                image_path = self.images[new_idx]
                json_path = self.jsons[new_idx]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask_json, reason_text, is_sentence = get_mask_from_json(json_path, image)
        
        # 会話を生成
        conversations = []
        
        # Gemma3用のテンプレート取得
        template = get_default_conv_template("gemma_v1")
        template.system = SYSTEM_PROMPT
        
        # メッセージを追加
        template.messages = []
        
        # 画像ファイル名のみを取得（パスなし）
        img_name = os.path.basename(image_path)
        
        if is_sentence:
            # 文章ベースの理由付きセグメンテーション
            query = f"{DEFAULT_IMAGE_TOKEN}\n{reason_text} Please output segmentation mask."
            template.append_message(template.roles[0], query)
            
            if use_explanatory and img_name in self.img_to_explanation:
                # JSONから読み込んだ説明を使用
                response = f"[SEG]. {self.img_to_explanation[img_name]['outputs']}"
                template.append_message(template.roles[1], response)
            else:
                # 通常の回答
                template.append_message(template.roles[1], "[SEG].")
        else:
            # 物体ベースの理由付きセグメンテーション
            query = f"{DEFAULT_IMAGE_TOKEN}\nWhat is {reason_text} in this image? Please output segmentation mask."
            template.append_message(template.roles[0], query)
            
            if use_explanatory and img_name in self.img_to_explanation:
                # JSONから読み込んだ説明を使用
                response = f"[SEG]. {self.img_to_explanation[img_name]['outputs']}"
                template.append_message(template.roles[1], response)
            else:
                # 通常の回答
                template.append_message(template.roles[1], "[SEG].")
        
        conversations.append(template.get_prompt())
        
        # 画像を処理
        # Gemma3用の画像処理
        if self.image_processor is not None:
            images_gemma = self.image_processor(image)
        else:
            # フォールバック: 標準的なリサイズと正規化
            h, w = image.shape[:2]
            size = self.image_size
            image_gemma = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            image_gemma = torch.from_numpy(image_gemma).permute(2, 0, 1).float() / 255.0
            # 標準的な正規化値を適用
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
            images_gemma = (image_gemma - mean) / std
            
        # SAM用の高解像度画像処理
        image_sam = self.sam_transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        
        # マスクの処理
        mask = torch.from_numpy(mask_json).unsqueeze(0)  # (1, H, W)
        
        # ダミーラベルを作成
        label = torch.ones(image.shape[0], image.shape[1]) * self.ignore_label
        
        # 推論フラグと質問のプレースホルダー
        inference = False
        questions = reason_text
        sampled_classes = [reason_text]
        
        return (
            image_path,
            image_sam,
            images_gemma,
            conversations,
            mask,
            label,
            resize,
            questions,
            sampled_classes,
            inference,
        )
