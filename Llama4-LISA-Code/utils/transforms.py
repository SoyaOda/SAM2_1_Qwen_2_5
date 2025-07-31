# utils/transforms.py
"""
SAM互換の画像変換ユーティリティ
Original-LISAのResizeLongestSideを参考に実装
"""

from copy import deepcopy
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from PIL import Image


class ResizeLongestSide:
    """
    画像の最長辺を指定サイズにリサイズし、座標やボックスの変換メソッドも提供
    SAM公式の変換クラスと互換性を保つ
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        numpy配列（HxWxC、uint8形式）の画像をリサイズ
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        # PILを使用してリサイズ
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_size[1], target_size[0]), Image.LANCZOS)
        return np.array(resized_image)

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        座標の変換（元画像サイズからリサイズ後のサイズへ）
        coords: [..., 2] の形状を期待
        original_size: (H, W) 形式
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        バウンディングボックスの変換
        boxes: Bx4 の形状を期待
        original_size: (H, W) 形式
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        torch.Tensor（BxCxHxW、float形式）の画像をリサイズ
        """
        # Batchサイズが1の場合は次元を追加
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # CxHxWの場合のサイズ取得
        _, _, h, w = image.shape
        target_size = self.get_preprocess_shape(h, w, self.target_length)
        
        resized = F.interpolate(
            image, 
            size=target_size, 
            mode="bilinear", 
            align_corners=False
        )
        
        # 元の次元に戻す
        if image.dim() == 3:
            resized = resized.squeeze(0)
            
        return resized

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        torch.Tensorの座標変換
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = coords.clone().to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        torch.Tensorのバウンディングボックス変換
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        入力サイズと目標最長辺長から出力サイズを計算
        Returns: (new_h, new_w)
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def get_transform_info(self, original_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        変換情報を辞書形式で返す（マスク変換用）
        original_size: (H, W) 形式
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
        scale = self.target_length * 1.0 / max(old_h, old_w)
        
        return {
            'original_size': (old_h, old_w),
            'new_size': (new_h, new_w),
            'scale': scale,
            'target_length': self.target_length,
        }