# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F
import math

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # デバッグ情報: 入力テンソルの形状
        print(f"DEBUG: image_embeddings shape: {image_embeddings.shape}")
        print(f"DEBUG: sparse_prompt_embeddings shape: {sparse_prompt_embeddings.shape}")
        print(f"DEBUG: dense_prompt_embeddings shape: {dense_prompt_embeddings.shape}")
        
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        print(f"DEBUG: tokens shape: {tokens.shape}")

        # image_embeddings: [1, C, H, W], tokens: [B, N, C]
        # dense_prompt_embeddings: [B, C, H, W]
        # Expand per-image data in batch direction to be per-mask
        
        # バッチサイズの問題を修正 - できるだけ小さいバッチサイズに統一する
        # オリジナルのLISAと同様にトークン数でsrcを拡張せず、
        # すべてのテンソルが同じバッチサイズになるようにする
        sparse_batch_size = sparse_prompt_embeddings.shape[0]
        dense_batch_size = dense_prompt_embeddings.shape[0]
        
        # より小さい方のバッチサイズを使用
        target_batch_size = min(sparse_batch_size, dense_batch_size)
        print(f"DEBUG: 目標バッチサイズ: {target_batch_size}")
        
        # 各テンソルを目標バッチサイズに調整
        if tokens.shape[0] > target_batch_size:
            tokens = tokens[:target_batch_size]
            print(f"DEBUG: tokens調整後: {tokens.shape}")
            
        # image_embeddingsを目標バッチサイズに合わせて拡張
        if image_embeddings.shape[0] == 1:
            src = torch.repeat_interleave(image_embeddings, target_batch_size, dim=0)
        else:
            src = image_embeddings[:target_batch_size]
        print(f"DEBUG: src調整後: {src.shape}")
        
        # dense_prompt_embeddingsを目標バッチサイズに調整
        if dense_prompt_embeddings.shape[0] > target_batch_size:
            dense_prompt_embeddings = dense_prompt_embeddings[:target_batch_size]
        print(f"DEBUG: dense_prompt_embeddings調整後: {dense_prompt_embeddings.shape}")
        
        # テンソルを加算
        src = src + dense_prompt_embeddings
        
        # pos_srcを目標バッチサイズに合わせて拡張
        if image_pe.shape[0] == 1:
            pos_src = torch.repeat_interleave(image_pe, target_batch_size, dim=0)
        else:
            pos_src = image_pe[:target_batch_size]
        print(f"DEBUG: pos_src調整後: {pos_src.shape}")
        
        # 勾配計算のために明示的にrequires_gradを設定
        if not src.requires_grad:
            src.requires_grad = True
            
        # 形状情報を保存
        b, c, h, w = src.shape
        print(f"DEBUG: トランスフォーマー前 - 保存された形状: b={b}, c={c}, h={h}, w={w}")
        print(f"DEBUG: トランスフォーマー前 - src形状: {src.shape}, 要素数: {src.numel()}")

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        print(f"DEBUG: トランスフォーマー後 - src形状: {src.shape}, 要素数: {src.numel()}")
        print(f"DEBUG: hs形状: {hs.shape}")
        
        # トランスフォーマー後の実際のバッチサイズを取得
        actual_b = src.shape[0]
        
        # 保存した形状情報を更新
        if actual_b != b:
            print(f"DEBUG: バッチサイズが変更されました: {b} -> {actual_b}")
            b = actual_b
        
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        # トランスフォーマー出力の形状を変換
        # 元の形状と一致するか確認
        expected_elements = src.numel()
        actual_elements = src.numel()
        print(f"DEBUG: 期待される要素数: {expected_elements}, 実際の要素数: {actual_elements}")
        
        # トランスフォーマー出力の形状を適切に変換
        if src.dim() == 3:  # [B, HW, C]
            hw = src.shape[1]
            new_h = new_w = int(hw ** 0.5)
            if new_h * new_w == hw:
                src = src.permute(0, 2, 1).reshape(b, c, new_h, new_w)
                print(f"DEBUG: トランスフォーマー出力を形状変換: {src.shape}")
            else:
                print(f"ERROR: トランスフォーマー出力の空間次元が完全な二乗数ではありません: {hw}")
                raise ValueError(f"Cannot reshape transformer output with spatial dim {hw} to a square image")
        else:
            try:
                src = src.transpose(1, 2).view(b, c, h, w)
                print(f"DEBUG: 形状変換後: {src.shape}")
            except RuntimeError as e:
                print(f"ERROR: 形状変換に失敗: {e}")
                # フラット化してから再構成を試みる
                flattened = src.reshape(-1)
                print(f"DEBUG: フラット化: {flattened.shape}, 要素数: {flattened.numel()}")
                
                # 新しい形状を推測
                total_elements = flattened.numel()
                if total_elements % (b * c) == 0:
                    hw = total_elements // (b * c)
                    new_h = new_w = int(hw ** 0.5)
                    if new_h * new_w == hw:
                        try:
                            src = flattened.reshape(b, c, new_h, new_w)
                            h, w = new_h, new_w
                            print(f"DEBUG: 新しい形状で再構成: {src.shape}")
                        except RuntimeError as e2:
                            print(f"ERROR: 再構成に失敗: {e2}")
                            raise e2
                    else:
                        print(f"ERROR: 空間次元を正方形に再構成できません: {hw}")
                        raise ValueError(f"Cannot reshape spatial dim {hw} to a square")
                else:
                    print(f"ERROR: 要素数 {total_elements} をバッチサイズ {b} × チャネル数 {c} で割り切れません")
                    raise ValueError(f"Cannot reshape {total_elements} elements with batch {b} and channels {c}")

        upscaled_embedding = self.output_upscaling(src)
        print(f"DEBUG: upscaled_embedding形状: {upscaled_embedding.shape}")
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        print(f"DEBUG: hyper_in形状: {hyper_in.shape}")
        
        b, c, h, w = upscaled_embedding.shape
        print(f"DEBUG: 最終マスク生成 - upscaled_embedding形状: b={b}, c={c}, h={h}, w={w}")
        
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, self.num_mask_tokens, h, w
        )
        print(f"DEBUG: 生成されたマスク形状: {masks.shape}")

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        print(f"DEBUG: iou_pred形状: {iou_pred.shape}")

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
