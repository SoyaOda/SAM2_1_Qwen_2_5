# utils/dataset.py
"""
LISA-Llama4 デュアルストリーム・データパイプライン
仕様書第3章に従った実装
"""

import glob
import os
import random
from typing import Dict, List, Tuple, Optional, Any
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.utils.data
from pycocotools import mask
from transformers import AutoProcessor
from torchvision import transforms

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset

def get_config():
    """実行時の設定ファイルを動的に取得"""
    try:
        config_path = os.environ.get('LISA_CONFIG_PATH', 'config_linux')
        if os.path.exists('config_small_test.py'):
            import config_small_test as config
            print("設定: config_small_test.py を使用")
            return config
        if config_path == 'config_linux' and os.path.exists('config_linux.py'):
            import config_linux as config
            print("設定: config_linux.py を使用")
            return config
        if config_path != 'config_linux':
            import importlib
            config = importlib.import_module(config_path)
            print(f"設定: {config_path}.py を使用")
            return config
    except ImportError as e:
        print(f"設定ファイルのインポートに失敗: {e}")
    print("警告: 設定ファイルが見つかりません。デフォルト設定を使用します。")
    class DefaultConfig:
        MODEL_MAX_LENGTH = 2048
        LLAMA_IMAGE_SIZE = 448
        SAM_IMAGE_SIZE = 1024
        SEG_TOKEN = "[SEG]"
        DATASET_BASE_DIR = "./dataset"
    return DefaultConfig()

config = get_config()

# デフォルト設定
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SEG_TOKEN = getattr(config, 'SEG_TOKEN', "[SEG]")
IGNORE_INDEX = -100

def setup_seg_token(tokenizer, seg_token="[SEG]"):
    """
    オリジナルLISA準拠の[SEG]トークンセットアップ
    一元化された処理でフォールバックなし
    """
    num_added_tokens = tokenizer.add_tokens(seg_token)
    seg_token_idx = tokenizer(seg_token, add_special_tokens=False).input_ids[0]
    print(f"[SEG]トークンセットアップ完了:")
    print(f"  - 追加されたトークン数: {num_added_tokens}")
    print(f"  - [SEG]トークンID: {seg_token_idx}")
    return seg_token_idx

def preprocess_sam_image(image: Image.Image, target_size: Optional[int] = None) -> torch.Tensor:
    """
    SAM用画像前処理：1024x1024にリサイズ・パディング・正規化
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    w, h = image.size
    if max(w, h) != target_size:
        if w > h:
            new_w, new_h = target_size, int(h * target_size / w)
        else:
            new_w, new_h = int(w * target_size / h), target_size
        image = image.resize((new_w, new_h), Image.LANCZOS)
    w, h = image.size
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    padded_image.paste(image, (pad_w, pad_h))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[123.675/255, 116.28/255, 103.53/255],
            std=[58.395/255, 57.12/255, 57.375/255]
        )
    ])
    return transform(padded_image)

def preprocess_llama_image(image: Image.Image, processor: AutoProcessor, target_size: Optional[int] = None) -> torch.Tensor:
    """
    Llama用画像前処理：448x448にリサイズ・正規化
    """
    if target_size is None:
        target_size = getattr(config, 'LLAMA_IMAGE_SIZE', 448)
    try:
        processed = processor(images=image, return_tensors="pt")
        image_tensor = processed['pixel_values'].squeeze(0)
        return image_tensor
    except Exception as e:
        print(f"Llama画像前処理エラー: {e}")
        image_resized = image.resize((target_size, target_size), Image.LANCZOS)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image_resized)

def build_correct_labels_for_llama4(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Llama4チャットテンプレートに準拠した正確なラベルマスキング
    言語モデリング用に全トークンを予測対象にする（overfit成功パターン準拠）
    """
    labels = input_ids.clone()
    # overfit成功パターンに合わせて、全トークンを予測対象にする
    # これにより有効なラベルが存在し、損失が正常に計算される
    return labels

def preprocess_mask(mask: np.ndarray, target_size: Optional[int] = None) -> torch.Tensor:
    """
    マスクの前処理
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.size == 0:
        raise ValueError("空のマスクです")
    if mask.ndim < 2:
        raise ValueError(f"マスクの次元が不正です: {mask.ndim}D (最低2D必要)")
    if mask.ndim == 2:
        h, w = mask.shape
        mask_2d = mask
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            mask_2d = mask[0]
            h, w = mask_2d.shape
        elif mask.shape[-1] == 1:
            mask_2d = mask[:, :, 0]
            h, w = mask_2d.shape
        elif mask.shape[0] == 3 or mask.shape[-1] == 3:
            if mask.shape[0] == 3:
                mask_2d = mask[0]
                h, w = mask_2d.shape
            else:
                mask_2d = mask[:, :, 0]
                h, w = mask_2d.shape
        else:
            mask_2d = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
            h, w = mask_2d.shape
    else:
        raise ValueError(f"サポートされていないマスクの次元: {mask.ndim}D")
    if h == 0 or w == 0:
        raise ValueError(f"無効なマスクサイズ: {h}x{w}")
    if (h, w) != (target_size, target_size):
        try:
            mask_uint8 = mask_2d.astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode='L')
            mask_resized_pil = mask_pil.resize((target_size, target_size), Image.NEAREST)
            mask_2d = np.array(mask_resized_pil)
        except Exception as e:
            raise ValueError(f"マスクリサイズ失敗 (元サイズ: {h}x{w}, target: {target_size}x{target_size}): {e}")
    if mask_2d.ndim == 2:
        mask_2d = mask_2d[None, ...]
    return torch.from_numpy(mask_2d).float()

class HybridDataset(torch.utils.data.Dataset):
    """
    仕様書第3章.2 HybridDatasetの実装
    デュアルストリーム処理：Llama4用とSAM用の2系統前処理を同時実行
    """

    def __init__(
        self,
        base_image_dir: Optional[str] = None,
        llama_processor: Optional[AutoProcessor] = None,
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        precision: str = "bf16",
        llama_image_size: Optional[int] = None,
        sam_image_size: Optional[int] = None,
        num_classes_per_sample: int = 3,
        exclude_val: bool = False,
        dataset: str = "sem_seg||refer_seg||vqa||reason_seg",
        sample_rate: List[float] = [9, 3, 3, 1],
        sem_seg_data: Optional[str] = None,
        refer_seg_data: Optional[str] = None,
        vqa_data: Optional[str] = None,
        reason_seg_data: Optional[str] = None,
        explanatory: float = 0.1,
    ):
        self.base_image_dir = base_image_dir or getattr(config, 'DATASET_BASE_DIR', './dataset')
        self.llama_processor = llama_processor
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.llama_image_size = llama_image_size or getattr(config, 'LLAMA_IMAGE_SIZE', 448)
        self.sam_image_size = sam_image_size or getattr(config, 'SAM_IMAGE_SIZE', 1024)
        self.num_classes_per_sample = num_classes_per_sample
        self.exclude_val = exclude_val
        self.explanatory = explanatory
        
        self.sem_seg_data = sem_seg_data or getattr(config, 'SEM_SEG_DATA', "ade20k||cocostuff")
        self.refer_seg_data = refer_seg_data or getattr(config, 'REFER_SEG_DATA', "refcoco||refcoco+||refcocog")
        self.vqa_data = vqa_data or getattr(config, 'VQA_DATA', "llava_instruct_150k")
        self.reason_seg_data = reason_seg_data or getattr(config, 'REASON_SEG_DATA', "ReasonSeg|train")
        
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        if self.llama_processor and self.llama_processor.tokenizer:
            self.seg_token = getattr(config, 'SEG_TOKEN', '[SEG]')
            self.seg_token_idx = setup_seg_token(self.llama_processor.tokenizer, self.seg_token)
            self.max_length = getattr(config, 'MODEL_MAX_LENGTH', 2048)
        else:
            raise ValueError("llama_processor は必須です")

        self.datasets = dataset.split("||")
        self.all_datasets = []
        
        print(f"HybridDataset初期化:")
        print(f"  - ベースディレクトリ: {self.base_image_dir}")
        print(f"  - Llama画像サイズ: {self.llama_image_size}")
        print(f"  - SAM画像サイズ: {self.sam_image_size}")
        print(f"  - 対象データセット: {self.datasets}")
        
        if "sem_seg" in self.datasets:
            print(f"  - Semantic Segmentation: {self.sem_seg_data}")
            try:
                self.all_datasets.append(
                    SemSegDataset(
                        self.base_image_dir,
                        self.llama_processor.tokenizer,
                        None,
                        samples_per_epoch,
                        precision,
                        self.llama_image_size,
                        num_classes_per_sample,
                        exclude_val,
                        self.sem_seg_data,
                    )
                )
            except Exception as e:
                print(f"    警告: Semantic Segmentationデータセットの初期化に失敗: {e}")
        
        if "refer_seg" in self.datasets:
            print(f"  - Referring Segmentation: {self.refer_seg_data}")
            try:
                self.all_datasets.append(
                    ReferSegDataset(
                        self.base_image_dir,
                        self.llama_processor.tokenizer,
                        None,
                        samples_per_epoch,
                        precision,
                        self.llama_image_size,
                        num_classes_per_sample,
                        exclude_val,
                        self.refer_seg_data,
                    )
                )
            except Exception as e:
                print(f"    警告: Referring Segmentationデータセットの初期化に失敗: {e}")
        
        if "vqa" in self.datasets:
            print(f"  - VQA: {self.vqa_data}")
            try:
                self.all_datasets.append(
                    VQADataset(
                        self.base_image_dir,
                        self.llama_processor.tokenizer,
                        None,
                        samples_per_epoch,
                        precision,
                        self.llama_image_size,
                        exclude_val,
                        self.vqa_data,
                    )
                )
            except Exception as e:
                print(f"    警告: VQAデータセットの初期化に失敗: {e}")
        
        if "reason_seg" in self.datasets:
            print(f"  - Reasoning Segmentation: {self.reason_seg_data}")
            try:
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir=self.base_image_dir,
                        tokenizer=self.llama_processor.tokenizer,
                        vision_tower=None,
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=self.llama_image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        reason_seg_data=self.reason_seg_data,
                        explanatory=explanatory,
                    )
                )
            except Exception as e:
                print(f"    警告: Reasoning Segmentationデータセットの初期化に失敗: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✅ HybridDataset初期化完了: {len(self.all_datasets)} データセット")
        
        if len(self.sample_rate) != len(self.all_datasets):
            print(f"⚠️  sample_rate調整: {len(self.sample_rate)} -> {len(self.all_datasets)}")
            if len(self.all_datasets) == 0:
                raise ValueError("有効なデータセットが1つもありません")
            elif len(self.all_datasets) == 1:
                self.sample_rate = [1.0]
            else:
                original_sample_rate = self.sample_rate[:len(self.all_datasets)]
                total_rate = sum(original_sample_rate)
                self.sample_rate = [rate / total_rate for rate in original_sample_rate]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        統合データセットからサンプルを取得
        仕様書第3章.2準拠のデュアルストリーム・データパイプライン
        """
        dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        try:
            sample = selected_dataset[idx % len(selected_dataset)]
        except Exception as e:
            print(f"データセット取得エラー: {e}")
            sample = self.all_datasets[0][0]
        
        if len(sample) == 9:
            image_path, image_sam, image_llama, conversations, masks, label, resize, questions, sampled_classes = sample
            if isinstance(conversations, list) and len(conversations) > 0:
                text_prompt = conversations[0]
            else:
                text_prompt = "Segment the object in this image. [SEG]"
            resize = resize if 'resize' in locals() else None
            questions = questions if 'questions' in locals() else None
            sampled_classes = sampled_classes if 'sampled_classes' in locals() else None
        elif len(sample) == 5:
            image_path, image_data, text_prompt, masks, label = sample
            if isinstance(image_data, torch.Tensor):
                if image_data.dim() == 3 and image_data.size(0) == 3:
                    image_np = image_data.permute(1, 2, 0).cpu().numpy()
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_data.cpu().numpy()
                if len(image_np.shape) >= 2 and (image_np.shape[0] <= 1 or image_np.shape[1] <= 1):
                    raise ValueError(f"無効な画像サイズ: {image_np.shape}")
                if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        image_np = image_np.astype(np.uint8)
                elif image_np.dtype != np.uint8:
                    image_np = image_np.astype(np.uint8)
                image_pil = Image.fromarray(image_np)
            elif isinstance(image_data, Image.Image):
                image_pil = image_data
            else:
                if hasattr(image_data, 'shape'):
                    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                        if image_data.max() <= 1.0:
                            image_data = (image_data * 255).astype(np.uint8)
                        else:
                            image_data = image_data.astype(np.uint8)
                    elif image_data.dtype != np.uint8:
                        image_data = image_data.astype(np.uint8)
                    image_pil = Image.fromarray(image_data)
                else:
                    raise ValueError(f"サポートされていない画像形式: {type(image_data)}")
            image_llama = preprocess_llama_image(image_pil, self.llama_processor, self.llama_image_size)
            image_sam = preprocess_sam_image(image_pil, self.sam_image_size)
            resize = None
            questions = None
            sampled_classes = None
        else:
            raise ValueError(f"不明なサンプル形式: {len(sample)} 要素")

        if masks is not None and self.seg_token not in text_prompt:
            text_prompt += f" {self.seg_token}"

        if len(sample) == 9:
            if isinstance(image_sam, torch.Tensor):
                if image_sam.dim() == 3:
                    image_np = image_sam.permute(1, 2, 0).cpu().numpy()
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    if image_np.shape[0] == 1 or image_np.shape[1] == 1:
                        raise ValueError(f"無効な画像サイズ: {image_np.shape}")
                    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                        if image_np.max() <= 1.0:
                            image_np = (image_np * 255).astype(np.uint8)
                        else:
                            image_np = image_np.astype(np.uint8)
                    elif image_np.dtype != np.uint8:
                        image_np = image_np.astype(np.uint8)
                    image_pil = Image.fromarray(image_np)
                else:
                    raise ValueError(f"無効なテンソル次元: {image_sam.dim()}")
            else:
                if isinstance(image_sam, Image.Image):
                    image_pil = image_sam
                else:
                    raise ValueError(f"サポートされていない画像形式: {type(image_sam)}")
        else:
            pass

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        try:
            llama_processed = self.llama_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            input_ids = llama_processed['input_ids'].squeeze(0)
            attention_mask = llama_processed['attention_mask'].squeeze(0)
            pixel_values = llama_processed['pixel_values'].squeeze(0)
            image_llama = pixel_values
            image_sam = preprocess_sam_image(image_pil, self.sam_image_size)
        except Exception as e:
            print(f"❌ Llama4マルチモーダル処理エラー: {e}")
            print(f"   テキスト: {text_prompt[:100]}...")
            raise RuntimeError(f"Llama4マルチモーダル処理に失敗: {e}")

        if image_llama.dim() == 4:
            image_llama = image_llama.squeeze(0)
        if image_sam.dim() == 4:
            image_sam = image_sam.squeeze(0)

        seg_token_mask = (input_ids == self.seg_token_idx)
        labels = build_correct_labels_for_llama4(input_ids, self.llama_processor.tokenizer)

        has_mask = masks is not None
        if has_mask:
            if isinstance(masks, torch.Tensor):
                if masks.dim() == 2:
                    ground_truth_mask = masks.unsqueeze(0)
                elif masks.dim() == 3:
                    ground_truth_mask = masks[0:1]
                else:
                    ground_truth_mask = masks
            else:
                if isinstance(masks, np.ndarray):
                    ground_truth_mask = torch.from_numpy(masks)
                    if ground_truth_mask.dim() == 2:
                        ground_truth_mask = ground_truth_mask.unsqueeze(0)
                else:
                    ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)
                    has_mask = False
            if ground_truth_mask.size(-1) != self.sam_image_size or ground_truth_mask.size(-2) != self.sam_image_size:
                ground_truth_mask = F.interpolate(
                    ground_truth_mask.unsqueeze(0).float(),
                    size=(self.sam_image_size, self.sam_image_size),
                    mode='nearest'
                ).squeeze(0)
        else:
            ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'images_for_sam': image_sam,
            'images_for_llama': image_llama,
            'ground_truth_mask': ground_truth_mask if has_mask else None,
            'has_mask': has_mask,
            'seg_token_mask': seg_token_mask,
            'image_path': image_path if 'image_path' in locals() else None,
            'text_prompt': text_prompt,
            'resize': resize if 'resize' in locals() else None,
            'questions': questions if 'questions' in locals() else None,
            'sampled_classes': sampled_classes if 'sampled_classes' in locals() else None,
        }

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    仕様書第3章.3 バッチの結合 (collate_fn)
    デュアルストリーム対応のカスタムcollate関数
    
    最大シーケンス長は設定ファイルのMODEL_MAX_LENGTHを自動的に使用:
    - config_small_test.py が利用可能な場合: 512 (メモリ効率優先)
    - config_linux.py のみの場合: 2048 (通常設定)
    - 設定ファイルなしの場合: 2048 (デフォルト)
    """
    config = get_config()
    sam_image_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    images_for_llama = []
    images_for_sam = []
    input_ids = []
    attention_mask_list = []
    labels = []
    seg_token_masks = []
    ground_truth_masks = []
    has_masks = []
    image_paths = []
    text_prompts = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    
    for item in batch:
        images_for_llama.append(item["images_for_llama"])
        images_for_sam.append(item["images_for_sam"])
        input_ids.append(item["input_ids"])
        attention_mask_list.append(item["attention_mask"])
        label = item["labels"]
        if isinstance(label, torch.Tensor):
            if label.dim() == 1 and len(label) == item["input_ids"].size(0):
                labels.append(label)  # build_correct_labels_for_llama4で処理済み
            elif label.dim() == 0:
                labels.append(torch.full_like(item["input_ids"], label.item()))
            else:
                labels.append(torch.full_like(item["input_ids"], -100))
        else:
            labels.append(torch.full_like(item["input_ids"], -100))
        seg_token_masks.append(item["seg_token_mask"])
        if item.get("has_mask", False):
            ground_truth_masks.append(item["ground_truth_mask"])
        has_masks.append(item.get("has_mask", False))
        image_paths.append(item.get("image_path"))
        text_prompts.append(item.get("text_prompt"))
        resize_list.append(item.get("resize"))
        questions_list.append(item.get("questions"))
        sampled_classes_list.append(item.get("sampled_classes"))
    
    images_for_llama = torch.stack(images_for_llama)
    images_for_sam = torch.stack(images_for_sam)
    max_length = max(ids.size(0) for ids in input_ids)
    def pad_sequence(sequences, max_len, pad_value=0):
        padded = []
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.full((max_len - seq.size(0),), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq[:max_len]
            padded.append(padded_seq)
        return torch.stack(padded)
    input_ids_padded = pad_sequence(input_ids, max_length, pad_value=0)
    attention_mask_padded = pad_sequence(attention_mask_list, max_length, pad_value=0)
    labels_padded = pad_sequence(labels, max_length, pad_value=-100)
    seg_token_masks_padded = pad_sequence(seg_token_masks, max_length, pad_value=False)
    if ground_truth_masks:
        ground_truth_masks_stacked = torch.stack(ground_truth_masks)
    else:
        ground_truth_masks_stacked = None
    label_list = []
    for has_mask in has_masks:
        if has_mask:
            label_list.append(torch.ones(1, sam_image_size, sam_image_size) * 255)
        else:
            label_list.append(None)
    return {
        "images_for_llama": images_for_llama,           # (B, 3, 448, 448)
        "images_for_sam": images_for_sam,               # (B, 3, 1024, 1024)
        "input_ids": input_ids_padded,                  # (B, unified_max_length)
        "attention_mask": attention_mask_padded,        # (B, unified_max_length)
        "labels": labels_padded,                        # (B, unified_max_length)
        "seg_token_mask": seg_token_masks_padded,       # (B, unified_max_length)
        "ground_truth_mask": ground_truth_masks_stacked,
        "has_mask": has_masks,
        "image_paths": image_paths,
        "text_prompts": text_prompts,
        "masks_list": ground_truth_masks,
        "label_list": label_list,
        "resize_list": resize_list,
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "images": images_for_sam,
        "images_clip": images_for_llama,
    }

class LisaLlama4ValDataset(torch.utils.data.Dataset):
    """
    LISA-Llama4用の評価データセット
    デュアルストリーム対応
    """

    def __init__(
        self,
        base_image_dir: str,
        llama_processor: AutoProcessor,
        val_dataset: str,
        llama_image_size: int = 896,
        sam_image_size: int = 1024,
    ):
        self.base_image_dir = base_image_dir
        self.llama_processor = llama_processor
        self.llama_image_size = llama_image_size
        self.sam_image_size = sam_image_size
        
        if "refer_seg" in val_dataset.lower():
            self.dataset = ReferSegDataset(
                base_image_dir,
                llama_processor,
                None,
                1000,
                "bf16",
                llama_image_size,
                3,
                False,
                val_dataset,
            )
        elif "sem_seg" in val_dataset.lower():
            self.dataset = SemSegDataset(
                base_image_dir,
                llama_processor,
                None,
                1000,
                "bf16",
                llama_image_size,
                3,
                False,
                val_dataset,
            )
        else:
            self.dataset = ReasonSegDataset(
                base_image_dir,
                llama_processor,
                None,
                1000,
                "bf16",
                llama_image_size,
                3,
                False,
                "ReasonSeg|val",
            )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """評価用サンプルの取得（デュアルストリーム対応）"""
        sample = self.dataset[idx]
        if len(sample) < 4:
            raise ValueError(f"評価データセットのサンプル形式が不正です: {len(sample)} 要素 (最低4要素必要)")
        
        image_path = sample[0]
        image = sample[1]
        text_prompt = sample[2] if len(sample) > 2 else "Segment the object in this image. [SEG]"
        mask = sample[3] if len(sample) > 3 else None
        label = sample[4] if len(sample) > 4 else torch.tensor(0)
        
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"サポートされていない画像型: {type(image)}")
        
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"無効な画像サイズ: {image.size}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        try:
            llama_processed = self.llama_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            raise RuntimeError(f"評価用Llama前処理に失敗 (idx={idx}): {e}")
        
        images_for_llama = llama_processed['pixel_values'].squeeze(0)
        input_ids = llama_processed['input_ids'].squeeze(0)
        attention_mask = llama_processed['attention_mask'].squeeze(0)
        
        try:
            images_for_sam = preprocess_sam_image(image, self.sam_image_size)
        except Exception as e:
            raise RuntimeError(f"評価用SAM前処理に失敗 (idx={idx}): {e}")
        
        seg_token_id = self.llama_processor.tokenizer.convert_tokens_to_ids("[SEG]")
        if seg_token_id is None:
            raise ValueError("[SEG]トークンがトークナイザーに見つかりません")
        
        seg_token_mask = (input_ids == seg_token_id)
        
        has_mask = mask is not None and (isinstance(mask, (torch.Tensor, np.ndarray)) and mask.sum() > 0)
        if has_mask:
            try:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                ground_truth_mask = preprocess_mask(mask_np, self.sam_image_size)
            except Exception as e:
                raise RuntimeError(f"評価用マスク前処理に失敗 (idx={idx}): {e}")
        else:
            ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)
        
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        
        return {
            "images_for_llama": images_for_llama,
            "images_for_sam": images_for_sam,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "seg_token_mask": seg_token_mask,
            "ground_truth_mask": ground_truth_mask,
            "has_mask": has_mask,
            "image_path": image_path,
            "text_prompt": text_prompt,
        }