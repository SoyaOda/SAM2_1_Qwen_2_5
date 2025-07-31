import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
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

from .grefer import G_REFER
from .refer import REFER
from .constants import ANSWER_LIST, SHORT_QUESTION_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT
from .conversation import get_default_conv_template


class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
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
            refer_seg_data: 参照セグメンテーションデータ
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
        self.exclude_val = exclude_val
        
        # データセットの初期化（カンマ区切りとパイプ区切りの両方に対応）
        if isinstance(refer_seg_data, str):
            if "||" in refer_seg_data:
                self.all_datasets = refer_seg_data.split("||")
            elif "," in refer_seg_data:
                self.all_datasets = refer_seg_data.split(",")
            else:
                self.all_datasets = [refer_seg_data] if refer_seg_data else []
        else:
            self.all_datasets = refer_seg_data if isinstance(refer_seg_data, list) else []
        
        print(f"参照セグメンテーションデータセット: {self.all_datasets}")
        
        # 各データセットの設定
        self.images = []
        self.refs_per_image = {}
        
        # REFERオブジェクトのキャッシュ（オリジナルLISAと同様の実装）
        self.refer_apis = {}
        self.dataset_images = {}
        self.dataset_annotations = {}
        self.dataset_img2refs = {}
        
        # 初期化関数
        for ds in self.all_datasets:
            # 利用可能なデータセット：refclef, refcoco, refcoco+, refcocog
            if ds == "refclef":
                self.init_refclef()
            elif ds == "refcoco":
                self.init_refcoco()
            elif ds == "refcoco+":
                self.init_refcocoplus()
            elif ds == "refcocog":
                self.init_refcocog()
        
        print(f"合計画像数: {len(self.images)}")
        
        # 親クラスから渡されたプロセッサを使用
        if processor is not None and image_processor is not None:
            self.processor = processor
            self.image_processor = image_processor
            print(f"ReferSegDataset: 親クラスから渡されたプロセッサを使用します")
        else:
            # プロセッサが渡されなかった場合は独自に初期化（後方互換性用）
            from model.gemma3.mm_utils import get_gemma_processor, GemmaImageProcessor
            
            try:
                # モデル名からプロセッサを取得
                self.processor = get_gemma_processor(model_name)
                self.image_processor = GemmaImageProcessor(self.processor)
                print(f"ReferSegDataset: Gemma3プロセッサを初期化: {model_name}")
            except Exception as e:
                error_msg = f"ReferSegDataset: プロセッサの初期化に失敗しました: {e}"
                print(error_msg)
                # 例外を再発生させて親クラスにエラーを伝播
                raise RuntimeError(error_msg) from e

        # SAM用の変換
        self.sam_transform = ResizeLongestSide(self.img_size)
        
        # 質問リスト
        self.questions = SHORT_QUESTION_LIST
    
    def init_refclef(self):
        """RefClefデータセットを初期化"""
        # 正しいパスを構成（refer_segディレクトリを含む）
        refer_seg_dir = os.path.join(self.base_image_dir, "refer_seg")
        refer_api = REFER(refer_seg_dir, "refclef", "unc")
        
        # REFERオブジェクトをキャッシュ
        self.refer_apis["refclef"] = refer_api
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # データセット情報を保存
        self.dataset_images["refclef"] = {img["id"]: img for img in images}
        self.dataset_annotations["refclef"] = refer_api.Anns
        self.dataset_img2refs["refclef"] = img2refs
        
        # 画像パスとアノテーションを保存
        for image in images:
            # フルパスの構成方法を修正（画像パスにあるディレクトリ構造を維持）
            file_name = image["file_name"]
            
            # file_nameが既にサブディレクトリ構造を含んでいるかチェック
            if "/" in file_name or "\\" in file_name:
                image_path = os.path.join(self.base_image_dir, "refer_seg/images/saiapr_tc-12", file_name)
            else:
                # 元の画像名からディレクトリを解析（例: "19/images/19135.jpg"）
                dir_prefix = os.path.dirname(file_name)
                if not dir_prefix:
                    # ディレクトリプレフィックスがない場合、画像IDからディレクトリを推測
                    img_id_str = str(image["id"])
                    dir_prefix = img_id_str[:2] if len(img_id_str) >= 2 else "00"
                    # 新しいパス形式を構築
                    file_name = f"{dir_prefix}/images/{file_name}"
                
                image_path = os.path.join(self.base_image_dir, "refer_seg/images/saiapr_tc-12", file_name)
            
            # 実際にファイルが存在するか確認
            if os.path.exists(image_path):
                self.images.append({
                    "path": image_path,
                    "dataset": "refclef",
                    "image_id": image["id"]
                })
                self.refs_per_image[image_path] = img2refs[image["id"]]
            else:
                # ファイルが見つからない場合、警告を出して異なるパス形式を試みる
                print(f"警告: 画像が見つかりません: {image_path}")
                
                # フォールバック: 最終的に画像をスキップ
                continue
        
        print(f"RefClef: {sum(1 for img in self.images if img['dataset'] == 'refclef')}画像 (有効なパスのみ)")
    
    def init_refcoco(self):
        """RefCOCOデータセットを初期化"""
        # 正しいパスを構成（refer_segディレクトリを含む）
        refer_seg_dir = os.path.join(self.base_image_dir, "refer_seg")
        refer_api = REFER(refer_seg_dir, "refcoco", "unc")
        
        # REFERオブジェクトをキャッシュ
        self.refer_apis["refcoco"] = refer_api
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # データセット情報を保存
        self.dataset_images["refcoco"] = {img["id"]: img for img in images}
        self.dataset_annotations["refcoco"] = refer_api.Anns
        self.dataset_img2refs["refcoco"] = img2refs
        
        # 画像パスとアノテーションを保存
        for image in images:
            image_path = os.path.join(
                self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image["file_name"]
            )
            if os.path.exists(image_path):
                self.images.append({
                    "path": image_path,
                    "dataset": "refcoco",
                    "image_id": image["id"]
                })
                self.refs_per_image[image_path] = img2refs[image["id"]]
            else:
                print(f"警告: 画像が見つかりません: {image_path}")
        
        print(f"RefCOCO: {sum(1 for img in self.images if img['dataset'] == 'refcoco')}画像")
    
    def init_refcocoplus(self):
        """RefCOCO+データセットを初期化"""
        # 正しいパスを構成（refer_segディレクトリを含む）
        refer_seg_dir = os.path.join(self.base_image_dir, "refer_seg")
        refer_api = REFER(refer_seg_dir, "refcoco+", "unc")
        
        # REFERオブジェクトをキャッシュ
        self.refer_apis["refcoco+"] = refer_api
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # データセット情報を保存
        self.dataset_images["refcoco+"] = {img["id"]: img for img in images}
        self.dataset_annotations["refcoco+"] = refer_api.Anns
        self.dataset_img2refs["refcoco+"] = img2refs
        
        # 画像パスとアノテーションを保存
        for image in images:
            image_path = os.path.join(
                self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image["file_name"]
            )
            if os.path.exists(image_path):
                self.images.append({
                    "path": image_path,
                    "dataset": "refcoco+",
                    "image_id": image["id"]
                })
                self.refs_per_image[image_path] = img2refs[image["id"]]
            else:
                print(f"警告: 画像が見つかりません: {image_path}")
        
        print(f"RefCOCO+: {sum(1 for img in self.images if img['dataset'] == 'refcoco+')}画像")
    
    def init_refcocog(self):
        """RefCOCOgデータセットを初期化"""
        # 正しいパスを構成（refer_segディレクトリを含む）
        refer_seg_dir = os.path.join(self.base_image_dir, "refer_seg")
        refer_api = REFER(refer_seg_dir, "refcocog", "umd")
        
        # REFERオブジェクトをキャッシュ
        self.refer_apis["refcocog"] = refer_api
        
        # 参照IDを取得
        ref_ids = refer_api.getRefIds(split="train")
        
        # valデータが含まれないようにする場合
        if self.exclude_val:
            val_ref_ids = refer_api.getRefIds(split="val")
            ref_ids = [ref_id for ref_id in ref_ids if ref_id not in val_ref_ids]
        
        # 画像IDを取得
        img_ids = refer_api.getImgIds(ref_ids=ref_ids)
        
        # 参照情報を取得
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        
        # 画像ごとの参照を整理
        img2refs = {}
        for ref in refs:
            img_id = ref["image_id"]
            if img_id in img2refs:
                img2refs[img_id].append(ref)
            else:
                img2refs[img_id] = [ref]
        
        # 画像情報を取得
        images = refer_api.loadImgs(img_ids)
        
        # データセット情報を保存
        self.dataset_images["refcocog"] = {img["id"]: img for img in images}
        self.dataset_annotations["refcocog"] = refer_api.Anns
        self.dataset_img2refs["refcocog"] = img2refs
        
        # 画像パスとアノテーションを保存
        for image in images:
            image_path = os.path.join(
                self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image["file_name"]
            )
            if os.path.exists(image_path):
                self.images.append({
                    "path": image_path,
                    "dataset": "refcocog",
                    "image_id": image["id"]
                })
                self.refs_per_image[image_path] = img2refs[image["id"]]
            else:
                print(f"警告: 画像が見つかりません: {image_path}")
        
        print(f"RefCOCOg: {sum(1 for img in self.images if img['dataset'] == 'refcocog')}画像")
    
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
    
    def _create_return_values(self, image_path, image_sam, images_gemma, conversations, masks, resize):
        """戻り値の作成（ヘルパーメソッド）"""
        # NumPy配列からTensorに変換
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        
        # ダミーラベルを作成
        label = torch.ones(resize[0], resize[1]) * self.ignore_label
        
        questions = [conv.split('\n')[1] for conv in conversations]
        sampled_classes = questions
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
    
    def __getitem__(self, idx):
        """データセットからアイテムを取得"""
        # ランダムに画像情報を選択
        image_info = random.choice(self.images)
        image_path = image_info["path"]
        dataset_name = image_info["dataset"]
        image_id = image_info["image_id"]
        
        # 事前にロードした画像情報とアノテーションを使用
        image_data = self.dataset_images[dataset_name][image_id]
        annotations = self.dataset_annotations[dataset_name]
        img2refs = self.dataset_img2refs[dataset_name]
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 参照を取得
        refs = img2refs[image_id]
        
        # ランダムに参照を選択
        if len(refs) >= self.num_classes_per_sample:
            sampled_refs = random.sample(refs, self.num_classes_per_sample)
        else:
            sampled_refs = refs
        
        # 文章と注釈IDを抽出
        sents = []
        ann_ids = []
        for ref in sampled_refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"].strip().lower())
                ann_ids.append(ref["ann_id"])
        
        # 会話を生成
        conversations = []
        
        # Gemma3用のテンプレート取得
        template = get_default_conv_template("gemma_v1")
        template.system = SYSTEM_PROMPT
        
        # 各文章について会話を生成
        for text in sents:
            template.messages = []
            
            # ユーザーからの質問を生成
            query = f"{DEFAULT_IMAGE_TOKEN}\n{text} Please output segmentation mask."
            template.append_message(template.roles[0], query)
            
            # アシスタントの回答を生成
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
        masks = []
        
        # 画像情報を取得
        height = image_data.get("height", image.shape[0])
        width = image_data.get("width", image.shape[1])
        
        # アノテーションを取得
        for ann_id in ann_ids:
            try:
                ann = annotations.get(ann_id)
                
                if ann is None:
                    print(f"警告: アノテーションID {ann_id} のアノテーションが見つかりません")
                    m = np.zeros((height, width), dtype=np.uint8)
                elif len(ann["segmentation"]) == 0:
                    print(f"警告: アノテーションID {ann_id} のセグメンテーションが空です")
                    m = np.zeros((height, width), dtype=np.uint8)
                else:
                    # RLEからマスクを復元
                    if type(ann["segmentation"][0]) == list:  # ポリゴン
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            height,
                            width,
                        )
                    else:
                        rle = ann["segmentation"]
                        # バイトエンコードを確認（配列の場合）
                        if isinstance(rle, list):
                            for i in range(len(rle)):
                                if not isinstance(rle[i]["counts"], bytes):
                                    rle[i]["counts"] = rle[i]["counts"].encode()
                        else:
                            # 単一のRLEの場合
                            if not isinstance(rle["counts"], bytes):
                                rle["counts"] = rle["counts"].encode()
                    
                    m = mask.decode(rle)
                    if len(m.shape) == 3:
                        m = np.sum(m, axis=2)  # 複数パートのセグメントを合成
                    
                    m = m.astype(np.uint8)
            except Exception as e:
                print(f"マスク生成中にエラー発生: {e}")
                # エラーが発生した場合は空のマスク
                m = np.zeros((height, width), dtype=np.uint8)
            
            masks.append(m)
        
        # マスクが空の場合はダミーマスクを生成
        if len(masks) == 0:
            print(f"警告: 画像 {image_path} のマスクがありません")
            masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(len(ann_ids))]
        
        # マスクをスタック
        masks = np.stack(masks, axis=0)
        
        return self._create_return_values(image_path, image_sam, images_gemma, conversations, masks, resize)
