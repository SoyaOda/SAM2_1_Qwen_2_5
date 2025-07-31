# utils/dataset.py
"""
SAM2.1 + Qwen2.5-VL統合モデル用データセットパイプライン
o3_spec2.md仕様に基づく<SEG>特殊トークン対応
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
from .transforms import ResizeLongestSide

# 🔧 統一設定ファイルの読み込み（config_unified.py対応）
_cached_config = None

def get_config():
    """実行時の設定ファイルを動的に取得（統一設定優先）"""
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    
    try:
        # 統一設定ファイルを最優先で使用
        if os.path.exists('config_unified.py'):
            import config_unified as config
            print("✅ 設定: config_unified.py を使用（SAM2.1+Qwen2.5-VL統合設定）")
            _cached_config = config
            return config
        
        # フォールバック: 既存設定ファイル
        config_path = os.environ.get('LISA_CONFIG_PATH', 'config_linux')
        if os.path.exists('config_small_test.py'):
            import config_small_test as config
            print("設定: config_small_test.py を使用")
            _cached_config = config
            return config
        if config_path == 'config_linux' and os.path.exists('config_linux.py'):
            import config_linux as config
            print("設定: config_linux.py を使用")
            _cached_config = config
            return config
        if config_path != 'config_linux':
            import importlib
            config = importlib.import_module(config_path)
            print(f"設定: {config_path}.py を使用")
            _cached_config = config
            return config
    except ImportError as e:
        print(f"設定ファイルのインポートに失敗: {e}")
    
    print("⚠️ 警告: 設定ファイルが見つかりません。デフォルト設定を使用します。")
    class DefaultConfig:
        MODEL_MAX_LENGTH = 8192  # Qwen2.5-VL用に拡張
        QWEN_IMAGE_SIZE = 448    # Qwen2.5-VL画像サイズ
        SAM_IMAGE_SIZE = 1024    # SAM2.1画像サイズ
        SEG_TOKEN = "<SEG>"      # o3_spec2.md準拠
        IMAGE_TOKEN = "<image>"
        DATASET_BASE_DIR = "./dataset"
        SAMPLES_PER_EPOCH = 500
        # データセット設定
        SEM_SEG_DATA = "ade20k||cocostuff"
        REFER_SEG_DATA = "refcoco||refcoco+||refcocog"
        VQA_DATA = "llava_instruct_150k"
        REASON_SEG_DATA = "ReasonSeg|train"
    _cached_config = DefaultConfig()
    return _cached_config

config = get_config()

# デフォルト設定（o3_spec2.md準拠）
DEFAULT_IMAGE_TOKEN = getattr(config, 'IMAGE_TOKEN', "<image>")
DEFAULT_SEG_TOKEN = getattr(config, 'SEG_TOKEN', "<SEG>")  # o3_spec2.md準拠で<SEG>
IGNORE_INDEX = -100

def setup_seg_token(tokenizer, seg_token="<SEG>"):
    """
    o3_spec2.md準拠の<SEG>特殊トークンセットアップ
    エンドツーエンドマスク生成に対応
    """
    # 既存の特殊トークンを確認
    if seg_token in tokenizer.get_vocab():
        seg_token_idx = tokenizer.convert_tokens_to_ids(seg_token)
        print(f"✅ <SEG>トークンは既に存在: ID={seg_token_idx}")
        return seg_token_idx
    
    # 新しい特殊トークンとして追加
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': [seg_token]})
    seg_token_idx = tokenizer.convert_tokens_to_ids(seg_token)
    
    print(f"✅ <SEG>特殊トークンセットアップ完了:")
    print(f"  - 追加されたトークン数: {num_added_tokens}")
    print(f"  - <SEG>トークンID: {seg_token_idx}")
    print(f"  - 総語彙サイズ: {len(tokenizer)}")
    
    return seg_token_idx

def setup_image_token(tokenizer, image_token="<image>"):
    """
    シングルエンコーダー構成用の<image>トークンセットアップ
    LLMに画像の存在を示すシンボルとして機能
    """
    # トークナイザの既存の特殊トークンを確認
    existing_special_tokens = tokenizer.special_tokens_map
    additional_special_tokens = tokenizer.additional_special_tokens if hasattr(tokenizer, 'additional_special_tokens') else []
    
    # <image>トークンが既に存在するか確認
    if image_token in tokenizer.get_vocab():
        image_token_idx = tokenizer(image_token, add_special_tokens=False).input_ids[0]
        print(f"<image>トークンは既に存在:")
        print(f"  - <image>トークンID: {image_token_idx}")
    else:
        # <image>トークンを追加
        num_added_tokens = tokenizer.add_tokens(image_token, special_tokens=True)
        image_token_idx = tokenizer(image_token, add_special_tokens=False).input_ids[0]
        print(f"<image>トークンセットアップ完了:")
        print(f"  - 追加されたトークン数: {num_added_tokens}")
        print(f"  - <image>トークンID: {image_token_idx}")
    
    return image_token_idx

def preprocess_sam_image(image: Image.Image, target_size: Optional[int] = None) -> torch.Tensor:
    """
    SAM用画像前処理：Original-LISA準拠の実装
    ResizeLongestSideを使用してリサイズ、その後正規化とパディング
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # numpy配列に変換
    image_np = np.array(image)
    
    # ResizeLongestSideを使用してリサイズ
    transform = ResizeLongestSide(target_size)
    resized_image = transform.apply_image(image_np)
    
    # CHW形式に変換
    image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float()
    
    # SAM準拠の正規化（pixel_mean/pixel_std）
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    image_tensor = (image_tensor - pixel_mean) / pixel_std
    
    # パディング（右下にゼロパディング）
    h, w = image_tensor.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    image_tensor = F.pad(image_tensor, (0, padw, 0, padh))
    
    return image_tensor


def preprocess_mask_sam2_compliant(mask: np.ndarray, target_size: int = 1024) -> torch.Tensor:
    """
    画像前処理と同等のマスク前処理（SAM2+Original-LISA準拠）
    
    Args:
        mask: 入力マスク（numpy配列、2D）
        target_size: 目標サイズ（通常1024）
        
    Returns:
        前処理済みマスクテンソル (H, W)
    """
    # 1. ResizeLongestSideで画像と同じ変換を適用
    transform = ResizeLongestSide(target_size)
    resized_mask = transform.apply_image(mask.astype(np.uint8))  # uint8で変換
    
    # 2. 二値化処理（補間による値のブレを修正）
    resized_mask = (resized_mask > 0.5).astype(np.float32)
    
    # 3. テンソル化（正規化は不要）
    mask_tensor = torch.from_numpy(resized_mask).float()
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)
    elif mask_tensor.dim() == 3 and mask_tensor.shape[2] == 1:
        # (H, W, 1) → (1, H, W)
        mask_tensor = mask_tensor.squeeze(-1).unsqueeze(0)
    
    # 4. パディング（画像と同じ処理）
    h, w = mask_tensor.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    mask_tensor = F.pad(mask_tensor, (0, padw, 0, padh))
    
    return mask_tensor.squeeze(0)  # (H, W)


# preprocess_llama_image関数は削除済み
# Meta公式推奨：AutoProcessorを直接使用してpixel_valuesの最初のパッチを選択

def build_correct_labels_for_llama4(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Llama4チャットテンプレートに準拠した正確なラベルマスキング
    言語モデリング用に全トークンを予測対象にする（overfit成功パターン準拠）
    """
    labels = input_ids.clone()
    # overfit成功パターンに合わせて、全トークンを予測対象にする
    # これにより有効なラベルが存在し、損失が正常に計算される
    return labels

def preprocess_mask(mask: np.ndarray, original_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    マスクの前処理（Original-LISA準拠）
    注意: データセット段階ではマスクは元のサイズのまま保持
    
    Args:
        mask: 入力マスク (numpy array)
        original_size: 元画像サイズ (H, W) - 現在は使用しない
    """
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # マスクを2Dに変換
    if mask.ndim == 2:
        mask_2d = mask
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            mask_2d = mask[0]
        elif mask.shape[-1] == 1:
            mask_2d = mask[:, :, 0]
        else:
            # 複数チャンネルの場合、最初のチャンネルを使用
            mask_2d = mask[0] if mask.shape[0] <= mask.shape[-1] else mask[:, :, 0]
    else:
        raise ValueError(f"サポートされていないマスクの次元: {mask.ndim}D")
    
    # Original-LISA準拠: マスクは元のサイズのまま保持
    # モデル内でSAMが適切にリサイズとパディングを行う
    # ここでは単にテンソルに変換するのみ
    
    # 3D tensorに変換 (1, H, W)
    return torch.from_numpy(mask_2d[None, ...]).float()

class HybridDataset(torch.utils.data.Dataset):
    """
    SAM2.1 + Qwen2.5-VL統合モデル用ハイブリッドデータセット
    
    o3_spec2.md仕様に基づく統合実装:
    - SAM2.1用画像前処理（1024x1024）
    - Qwen2.5-VL用マルチモーダル前処理
    - <SEG>特殊トークンによるエンドツーエンドマスク生成対応
    - 複数データセット統合（sem_seg, refer_seg, vqa, reason_seg）
    """

    def __init__(
        self,
        base_image_dir: Optional[str] = None,
        qwen_processor: Optional[AutoProcessor] = None,  # Qwen2.5-VL用プロセッサ
        samples_per_epoch: int = 500,  # デフォルト値を現実的に
        precision: str = "bf16",
        qwen_image_size: Optional[int] = None,      # Qwen2.5-VL画像サイズ
        sam_image_size: Optional[int] = None,       # SAM2.1画像サイズ
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
        self.qwen_processor = qwen_processor  # Qwen2.5-VL用プロセッサ
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.qwen_image_size = qwen_image_size or getattr(config, 'QWEN_IMAGE_SIZE', 448)
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

        if self.qwen_processor and self.qwen_processor.tokenizer:
            # o3_spec2.md準拠の<SEG>特殊トークンセットアップ
            self.seg_token = getattr(config, 'SEG_TOKEN', '<SEG>')
            self.seg_token_idx = setup_seg_token(self.qwen_processor.tokenizer, self.seg_token)
            # Qwen2.5-VL用<image>トークンセットアップ
            self.image_token = DEFAULT_IMAGE_TOKEN
            self.image_token_idx = setup_image_token(self.qwen_processor.tokenizer, self.image_token)
            self.max_length = getattr(config, 'MODEL_MAX_LENGTH', 8192)  # Qwen2.5-VL用に拡張
        else:
            raise ValueError("qwen_processor は必須です（Qwen2.5-VL用プロセッサが必要）")

        self.datasets = dataset.split("||")
        self.all_datasets = []
        
        print(f"✅ SAM2.1 + Qwen2.5-VL統合HybridDataset初期化:")
        print(f"  - ベースディレクトリ: {self.base_image_dir}")
        print(f"  - Qwen2.5-VL画像サイズ: {self.qwen_image_size}")
        print(f"  - SAM2.1画像サイズ: {self.sam_image_size}")
        print(f"  - 対象データセット: {self.datasets}")
        print(f"  - <SEG>トークン: {self.seg_token} (ID: {self.seg_token_idx})")
        
        if "sem_seg" in self.datasets:
            print(f"  - Semantic Segmentation: {self.sem_seg_data}")
            try:
                self.all_datasets.append(
                    SemSegDataset(
                        self.base_image_dir,
                        self.qwen_processor.tokenizer,  # Qwen2.5-VL用トークナイザー
                        None,
                        samples_per_epoch,
                        precision,
                        self.qwen_image_size,  # Qwen2.5-VL画像サイズ
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
                        self.qwen_processor.tokenizer,  # Qwen2.5-VL用トークナイザー
                        None,
                        samples_per_epoch,
                        precision,
                        self.qwen_image_size,  # Qwen2.5-VL画像サイズ
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
                        self.qwen_processor.tokenizer,  # Qwen2.5-VL用トークナイザー
                        None,
                        samples_per_epoch,
                        precision,
                        self.qwen_image_size,  # Qwen2.5-VL画像サイズ
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
                        tokenizer=self.qwen_processor.tokenizer,  # Qwen2.5-VL用トークナイザー
                        vision_tower=None,
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=self.qwen_image_size,  # Qwen2.5-VL画像サイズ
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
        シングルエンコーダー構成：SAM専用画像処理とテキストフォーマット
        """
        dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        
        # データセット名を取得
        dataset_name = "unknown"
        if dataset_idx < len(self.datasets):
            dataset_name = self.datasets[dataset_idx]
        else:
            # フォールバック：インデックスから推測
            dataset_types = ["sem_seg", "refer_seg", "vqa", "reason_seg"]
            if dataset_idx < len(dataset_types):
                dataset_name = dataset_types[dataset_idx]
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
            # デュアルエンコーダー構成：SAMとLlama両方の画像処理
            image_sam = preprocess_sam_image(image_pil, self.sam_image_size)
            # Llama-4画像処理は後でAutoProcessorで実行（テキストと一緒に処理）
            image_llama = None  # 後でAutoProcessorで処理
            # 元画像サイズを記録
            original_size = (image_pil.height, image_pil.width)
            resize = None
            questions = None
            sampled_classes = None
        else:
            raise ValueError(f"不明なサンプル形式: {len(sample)} 要素")

        if masks is not None and self.seg_token not in text_prompt:
            text_prompt += f" {self.seg_token}"

        if len(sample) == 9:
            # image_samはSAM用の画像データ（元画像またはテンソル）
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
            
            # デュアルエンコーダー構成：SAMとLlama両方の画像処理を実行
            image_sam = preprocess_sam_image(image_pil, self.sam_image_size)
            # Llama-4画像処理は後でAutoProcessorで実行（テキストと一緒に処理）
            image_llama = None  # 後でAutoProcessorで処理
            # 元画像サイズを記録（マスク処理用）
            original_size = (image_pil.height, image_pil.width)
        else:
            # len(sample) == 5の場合は既に処理済み
            original_size = None

        # デュアルエンコーダー構成：Llama-4とSAM2両方の処理
        # Llama-4のapply_chat_templateを使用（画像込み）
        clean_prompt = text_prompt.replace('<image>\n', '').replace('<image>', '').strip()
        
        # Llama-4のネイティブマルチモーダルフォーマットでメッセージを作成
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # 画像プレースホルダー
                    {"type": "text", "text": clean_prompt}
                ]
            }
        ]
        
        # apply_chat_templateでLlama-4形式のテキストを生成
        formatted_prompt = self.llama_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # テキストとして取得
        )
        
        # テキストと画像を一緒に処理（Llama-4のネイティブマルチモーダル）
        try:
            # Llama-4用画像がtensorの場合、PILに変換
            if isinstance(image_llama, torch.Tensor):
                # 逆正規化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                denorm_image = image_llama * std + mean
                denorm_image = torch.clamp(denorm_image, 0, 1)
                # HWC形式に変換してPILイメージに
                image_np = (denorm_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_for_processor = Image.fromarray(image_np)
            else:
                image_for_processor = image_pil
            
            # Llama-4プロセッサで画像とテキストを処理
            llama_inputs = self.llama_processor(
                text=formatted_prompt,
                images=image_for_processor,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = llama_inputs['input_ids'].squeeze(0)
            attention_mask = llama_inputs['attention_mask'].squeeze(0)
            
            # Meta公式推奨：AutoProcessorのpixel_valuesを標準処理
            if 'pixel_values' in llama_inputs:
                pixel_values_raw = llama_inputs['pixel_values']
                # Llama-4のEarly Fusionアーキテクチャ：[10, 3, 448, 448] → 最初のパッチを選択
                if pixel_values_raw.dim() == 4 and pixel_values_raw.size(0) == 10:
                    pixel_values = pixel_values_raw[0]  # [3, 448, 448]
                elif pixel_values_raw.dim() == 3:
                    pixel_values = pixel_values_raw  # [3, 448, 448]
                else:
                    pixel_values = pixel_values_raw.squeeze(0)
            else:
                # AutoProcessorでpixel_valuesが取得できない場合
                raise RuntimeError("AutoProcessorでpixel_valuesが取得できませんでした")
                
        except Exception as e:
            print(f"❌ Llama-4マルチモーダル処理エラー: {e}")
            raise RuntimeError(f"Llama-4処理に失敗: {e}")

        # SAM2画像の最終形状確認
        if image_sam.dim() == 4:
            image_sam = image_sam.squeeze(0)

        seg_token_mask = (input_ids == self.seg_token_idx)
        labels = build_correct_labels_for_llama4(input_ids, self.llama_processor.tokenizer)

        has_mask = masks is not None
        if has_mask:
            # preprocess_mask関数を使用（Original-LISA準拠）
            try:
                ground_truth_mask = preprocess_mask(masks, original_size)
            except Exception as e:
                print(f"⚠️ マスク前処理エラー: {e}")
                # フォールバック：単純なリサイズ
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
                
                # Original-LISA準拠：マスクは元のサイズのまま保持
        else:
            ground_truth_mask = None

        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,        # デュアルエンコーダー構成：Llama-4用（追加）
            'sam_pixel_values': image_sam,       # デュアルエンコーダー構成：SAM2用
            'ground_truth_mask': ground_truth_mask if has_mask else None,
            'has_mask': has_mask,
            'seg_token_mask': seg_token_mask,
            'image_path': image_path if 'image_path' in locals() else None,
            'text_prompt': text_prompt,
            'formatted_prompt': formatted_prompt,  # デバッグ用に追加
            'original_size': original_size,  # 元画像サイズ（デバッグ用）
            'resize': resize if 'resize' in locals() else None,
            'questions': questions if 'questions' in locals() else None,
            'sampled_classes': sampled_classes if 'sampled_classes' in locals() else None,
            'dataset_name': dataset_name,  # データセット名を追加
        }

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    デュアルエンコーダー構成対応のカスタムcollate関数
    Llama-4とSAM2両方のパイプラインに対応
    
    最大シーケンス長は設定ファイルのMODEL_MAX_LENGTHを自動的に使用:
    - config_small_test.py が利用可能な場合: 512 (メモリ効率優先)
    - config_linux.py のみの場合: 2048 (通常設定)
    - 設定ファイルなしの場合: 2048 (デフォルト)
    """
    config = get_config()
    sam_image_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    # デュアルエンコーダー構成: Llama-4とSAM2両方の画像
    pixel_values = []  # Llama-4用
    sam_pixel_values = []  # SAM2用
    input_ids = []
    attention_mask_list = []
    labels = []
    seg_token_masks = []
    ground_truth_masks = []
    has_masks = []
    image_paths = []
    text_prompts = []
    formatted_prompts = []
    original_sizes = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    
    for idx, item in enumerate(batch):
        # デュアルエンコーダー構成: 両方の画像を収集
        if "pixel_values" in item:
            pixel_values.append(item["pixel_values"])
        sam_pixel_values.append(item["sam_pixel_values"])
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
        formatted_prompts.append(item.get("formatted_prompt"))
        original_sizes.append(item.get("original_size"))
        resize_list.append(item.get("resize"))
        questions_list.append(item.get("questions"))
        sampled_classes_list.append(item.get("sampled_classes"))
    
    # デュアルエンコーダー構成: 両方の画像をスタック
    if pixel_values:
        pixel_values = torch.stack(pixel_values)
    else:
        # フォールバック：pixel_valuesがない場合はSAM画像から生成
        pixel_values = torch.stack([F.interpolate(sam.unsqueeze(0), size=(448, 448), mode='bilinear').squeeze(0) for sam in sam_pixel_values])
    sam_pixel_values = torch.stack(sam_pixel_values)
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
    # Original-LISA準拠: マスクは元のサイズのまま（スタックしない）
    # モデル内でバッチ処理される
    label_list = []
    for has_mask in has_masks:
        if has_mask:
            label_list.append(torch.ones(1, sam_image_size, sam_image_size) * 255)
        else:
            label_list.append(None)
    return {
        # デュアルエンコーダー構成: 両方の画像を返す
        "pixel_values": pixel_values,                    # (B, 3, 448, 448) - Llama-4用
        "sam_pixel_values": sam_pixel_values,           # (B, 3, 1024, 1024) - SAM2用
        "input_ids": input_ids_padded,                  # (B, unified_max_length)
        "attention_mask": attention_mask_padded,        # (B, unified_max_length)
        "labels": labels_padded,                        # (B, unified_max_length)
        "seg_token_mask": seg_token_masks_padded,       # (B, unified_max_length)
        "ground_truth_mask": ground_truth_masks,  # リストのまま返す（Original-LISA準拠）
        "has_mask": has_masks,
        "image_paths": image_paths,
        "text_prompts": text_prompts,
        "formatted_prompts": formatted_prompts,        # デバッグ用に追加
        "original_sizes": original_sizes,              # 元画像サイズ（デバッグ用）
        "masks_list": ground_truth_masks,
        "label_list": label_list,
        "resize_list": resize_list,
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        # 互換性のためのエイリアス
        "images": pixel_values,  # デュアルエンコーダー構成: Llama-4用画像を使用
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