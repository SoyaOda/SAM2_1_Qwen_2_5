import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
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

from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT
from .conversation import get_default_conv_template


def preprocess_multimodal(source, mm_use_im_start_end, conv_version="llava_v1"):
    """マルチモーダル入力の前処理"""
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            
            # Gemma3とLLaVAのテンプレート形式の違いを処理
            if "gemma" in conv_version and GEMMA_AVAILABLE:
                # Gemma3のフォーマット
                pass  # Gemma3では特別な処理は不要
            elif LLAVA_AVAILABLE and "mmtag" in conversation_lib.default_conversation.version:
                # LLaVAのタグ形式
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class VQADataset(torch.utils.data.Dataset):
    """ビジュアルQAデータセット"""
    
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
        vqa_data="llava_instruct_150k",
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
            vqa_data: VQAデータセット名
            processor: 親クラスから渡されるプロセッサ
            image_processor: 親クラスから渡される画像プロセッサ
        """
        self.base_image_dir = base_image_dir
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.model_name = model_name
        
        # データセットの初期化
        if isinstance(vqa_data, str):
            if "||" in vqa_data:
                self.all_datasets = vqa_data.split("||")
            elif "," in vqa_data:
                self.all_datasets = vqa_data.split(",")
            else:
                self.all_datasets = [vqa_data] if vqa_data else []
        else:
            self.all_datasets = vqa_data if isinstance(vqa_data, list) else []
        
        print(f"VQAデータセット: {self.all_datasets}")
        
        # 各データセットの設定
        self.images = []
        self.questions = []
        self.answers = []
        
        for ds in self.all_datasets:
            if ds == "llava_instruct_150k":
                self.init_llava_dataset()
        
        print(f"VQAデータセット: 合計質問数 = {len(self.questions)}")
        
        # 親クラスから渡されたプロセッサを使用
        if processor is not None and image_processor is not None:
            self.processor = processor
            self.image_processor = image_processor
            print(f"VQADataset: 親クラスから渡されたプロセッサを使用します")
        else:
            # プロセッサが渡されなかった場合は独自に初期化（後方互換性用）
            from model.gemma3.mm_utils import get_gemma_processor, GemmaImageProcessor
            
            try:
                # モデル名からプロセッサを取得
                self.processor = get_gemma_processor(model_name)
                self.image_processor = GemmaImageProcessor(self.processor)
                print(f"VQADataset: Gemma3プロセッサを初期化: {model_name}")
            except Exception as e:
                error_msg = f"VQADataset: プロセッサの初期化に失敗しました: {e}"
                print(error_msg)
                # 例外を再発生させて親クラスにエラーを伝播
                raise RuntimeError(error_msg) from e
        
        # SAM用の変換
        self.sam_transform = ResizeLongestSide(self.img_size)
    
    def init_llava_dataset(self):
        """LLaVAインストラクションデータセットを初期化"""
        # データセットパス
        DATA_DIR = os.path.join(self.base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(self.base_image_dir, "coco/train2017")
        
        dataset_path = os.path.join(DATA_DIR, "llava_instruct_150k.json")
        print(f"LLaVAデータセットのロード: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"警告: LLaVAデータセットファイルがありません: {dataset_path}")
            return
            
        try:
            with open(dataset_path, "r") as f:
                data = json.load(f)
            
            # オリジナルのLISAコードに合わせて、全データを保持
            self.vqa_data = data
            print(f"LLaVAデータ読み込み成功: {len(self.vqa_data)}件")
            
            # 質問と回答を抽出
            for item in data:
                if "image" in item and item["image"]:
                    image_path = os.path.join(self.vqa_image_root, item["image"])
                    if os.path.exists(image_path):
                        if "conversations" in item and len(item["conversations"]) >= 2:
                            # 人間の質問とGPTの回答を抽出
                            question = item["conversations"][0]["value"]
                            answer = item["conversations"][1]["value"]
                            
                            self.images.append(image_path)
                            self.questions.append(question)
                            self.answers.append(answer)
            
            # サンプルデータの表示（最初の5件のみ）                    
            for i in range(min(5, len(self.images))):
                print(f"サンプル[{i}] - 画像: {self.images[i]}")
                print(f"質問: {self.questions[i][:50]}...")
                print(f"回答: {self.answers[i][:50]}...")
                
            print(f"有効なVQAサンプル数: {len(self.images)}件")
                    
        except Exception as e:
            print(f"LLaVAデータセット読み込みエラー: {e}")
            self.vqa_data = []
            
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
        # ランダムに画像・質問・回答を選択 - オリジナルのLISAコードに合わせる
        if not hasattr(self, 'vqa_data') or len(self.vqa_data) == 0:
            raise ValueError("VQAデータセットが空です")
            
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        
        # オリジナルLISAコードと同じパス構築方法
        if "image" not in item or not item["image"]:
            # 不正なデータの場合は再帰的に別のアイテムを選択
            return self.__getitem__(random.randint(0, len(self.vqa_data) - 1))
            
        image_path = os.path.join(self.vqa_image_root, item["image"])
        
        # 画像ファイルの存在確認
        if not os.path.exists(image_path):
            print(f"警告: 画像が存在しません: {image_path}")
            # 再帰的に別の画像を選択
            return self.__getitem__(random.randint(0, len(self.vqa_data) - 1))
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 画像を読み込めませんでした: {image_path}")
            # 再帰的に別の画像を選択
            return self.__getitem__(random.randint(0, len(self.vqa_data) - 1))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        
        # Gemma3用の画像処理
        if self.image_processor is not None:
            images_gemma = self.image_processor(image)
        else:
            # フォールバック: 標準的なリサイズと正規化
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                    std=[0.26862954, 0.26130258, 0.27577711])
            ])
            images_gemma = transform(image).unsqueeze(0)
        
        # SAM用の画像処理
        image_sam = self.sam_transform.apply_image(image)
        resize = image_sam.shape[:2]
        
        # 会話を生成
        conv = get_default_conv_template("gemma_v1")
        conv.system = SYSTEM_PROMPT
        
        # オリジナルLISAコードと同様に会話データを処理
        if "conversations" not in item or len(item["conversations"]) < 2:
            # 不正なデータの場合は再帰的に別のアイテムを選択
            return self.__getitem__(random.randint(0, len(self.vqa_data) - 1))
            
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=True,
            conv_version="gemma_v1"
        )
        
        # 会話データの処理
        try:
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            conversations = []
            
            # 最初のメッセージがhumanからでない場合はスキップ
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]
                
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"会話の順序が不正: {j}"
                conv.append_message(role, sentence["value"])
                
            conversations.append(conv.get_prompt())
        except Exception as e:
            print(f"会話処理エラー: {e}, アイテム: {item['image']}")
            # 再帰的に別のアイテムを選択
            return self.__getitem__(random.randint(0, len(self.vqa_data) - 1))
        
        # 質問と選択されたクラスの設定
        questions = conversations
        sampled_classes = conversations
        
        # SAM用の画像前処理
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        
        # マスクと正解ラベルのダミーデータ
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        
        # 推論モードでない
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
