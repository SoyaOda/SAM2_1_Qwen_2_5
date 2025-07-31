"""
LISA-Gemma3 モデルの損失関数モジュール
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Optional, Any


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """
    DICE損失を計算します（マスク用のIOU損失に類似）
    Args:
        inputs: 任意の形状の浮動小数点テンソル
                各サンプルの予測値
        targets: inputsと同じ形状の浮動小数点テンソル
                各要素のバイナリ分類ラベルを格納
                (0: ネガティブクラス、1: ポジティブクラス)
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    シグモイドクロスエントロピー損失を計算します
    Args:
        inputs: 任意の形状の浮動小数点テンソル
                各サンプルの予測値
        targets: inputsと同じ形状の浮動小数点テンソル
                各要素のバイナリ分類ラベルを格納
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss


class DiceLoss(nn.Module):
    """DICE損失の実装"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測マスク (B, 1, H, W) または (B, H, W)
            target: 正解マスク (B, 1, H, W) または (B, H, W)
        Returns:
            DICE損失値
        """
        # 次元を揃える
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)  # (B, H, W)
        if target.dim() == 4:
            if target.size(1) == 1:
                target = target.squeeze(1)  # (B, H, W)
            elif target.size(1) == 3:
                # RGBの場合はグレースケールに変換（平均を取る）
                target = target.mean(dim=1)  # (B, H, W)
            
        # サイズが異なる場合、targetをpredのサイズにリサイズ
        if pred.shape != target.shape:
            import torch.nn.functional as F
            if target.dim() == 3:  # (B, H, W)
                target = F.interpolate(
                    target.unsqueeze(1),  # (B, 1, H, W)
                    size=(pred.size(1), pred.size(2)),
                    mode='nearest'
                ).squeeze(1)  # (B, H, W)
            elif target.dim() == 2:  # (H, W)
                target = F.interpolate(
                    target.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                    size=(pred.size(1), pred.size(2)),
                    mode='nearest'
                ).squeeze(0).squeeze(0)  # (H, W)
        
        # シグモイドを適用（予測値が確率でない場合）
        pred = torch.sigmoid(pred)
        
        # バッチ次元以外をフラット化
        pred_flat = pred.view(pred.size(0), -1)  # (B, H*W)
        target_flat = target.view(target.size(0), -1)  # (B, H*W)
        
        # DICE係数の計算
        intersection = (pred_flat * target_flat).sum(dim=1)  # (B,)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # (B,)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # DICE損失 = 1 - DICE係数
        dice_loss = 1.0 - dice
        
        return dice_loss.mean()


class BCELoss(nn.Module):
    """Binary Cross Entropy損失の実装"""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測マスク (B, 1, H, W) または (B, H, W)
            target: 正解マスク (B, 1, H, W) または (B, H, W)
        Returns:
            BCE損失値
        """
        # 次元を揃える
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)  # (B, H, W)
        if target.dim() == 4:
            if target.size(1) == 1:
                target = target.squeeze(1)  # (B, H, W)
            elif target.size(1) == 3:
                # RGBの場合はグレースケールに変換（平均を取る）
                target = target.mean(dim=1)  # (B, H, W)
            
        # サイズが異なる場合、targetをpredのサイズにリサイズ
        if pred.shape != target.shape:
            import torch.nn.functional as F
            if target.dim() == 3:  # (B, H, W)
                target = F.interpolate(
                    target.unsqueeze(1),  # (B, 1, H, W)
                    size=(pred.size(1), pred.size(2)),
                    mode='nearest'
                ).squeeze(1)  # (B, H, W)
            elif target.dim() == 2:  # (H, W)
                target = F.interpolate(
                    target.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                    size=(pred.size(1), pred.size(2)),
                    mode='nearest'
                ).squeeze(0).squeeze(0)  # (H, W)
        
        # BCEWithLogitsLossを使用（内部でシグモイドを適用）
        return self.bce(pred, target.float())


class CompositeLoss(nn.Module):
    """
    複合損失関数
    テキスト生成損失 + セグメンテーション損失（DICE + BCE）
    """
    
    def __init__(
        self,
        ce_loss_weight: float = 1.0,
        dice_loss_weight: float = 0.5,
        bce_loss_weight: float = 2.0,
    ):
        super().__init__()
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        
        # 損失関数の初期化
        self.dice_loss = DiceLoss()
        self.bce_loss = BCELoss()
    
    def forward(
        self,
        model_outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        複合損失の計算
        
        Args:
            model_outputs: モデルの出力
                - "text_loss": テキスト生成損失 (Optional)
                - "predicted_masks": 予測マスク (Optional)
                - "logits": 言語モデルのロジット (Optional)
            batch: バッチデータ
                - "labels": テキストのラベル (Optional)
                - "ground_truth_mask": 正解マスク (Optional)
        
        Returns:
            損失の辞書
                - "total_loss": 総損失
                - "text_loss": テキスト損失
                - "dice_loss": DICE損失
                - "bce_loss": BCE損失
        """
        losses = {}
        
        # デバイスを安全に取得
        device = None
        for value in model_outputs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        
        if device is None:
            device = torch.device("cpu")
        
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. テキスト生成損失
        text_loss = model_outputs.get("text_loss")
        if text_loss is not None:
            losses["text_loss"] = text_loss
            total_loss += self.ce_loss_weight * text_loss
        else:
            # text_lossが直接渡されていない場合、logitsとlabelsから計算
            logits = model_outputs.get("logits")
            labels = batch.get("labels")
            
            if logits is not None and labels is not None:
                # CrossEntropyLossでtext_lossを計算
                ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                
                # logitsを(batch_size * seq_length, vocab_size)に変形
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                # 有効なラベルがあるかチェック
                valid_labels = (labels_flat != -100).sum().item()
                if valid_labels > 0:
                    text_loss = ce_loss_fn(logits_flat, labels_flat)
                    losses["text_loss"] = text_loss
                    total_loss += self.ce_loss_weight * text_loss
                else:
                    losses["text_loss"] = torch.tensor(0.0, device=device)
            else:
                losses["text_loss"] = torch.tensor(0.0, device=device)
        
        # 2. セグメンテーション損失
        predicted_masks = model_outputs.get("predicted_masks")
        ground_truth_mask = batch.get("ground_truth_mask")
        
        if predicted_masks is not None and ground_truth_mask is not None:
            # デバッグ: テンソルサイズを出力（最初の3回のみ）
            if hasattr(self, '_mask_debug_counter'):
                self._mask_debug_counter += 1
            else:
                self._mask_debug_counter = 1
                
            if self._mask_debug_counter <= 3:
                print(f"  predicted_masks shape: {predicted_masks.shape}")
                print(f"  ground_truth_mask shape: {ground_truth_mask.shape}")
            elif self._mask_debug_counter == 4:
                print(f"🔇 マスク形状 ログ表示を抑制（以降は省略）")
            
            # DICE損失
            dice_loss = self.dice_loss(predicted_masks, ground_truth_mask)
            losses["dice_loss"] = dice_loss
            total_loss += self.dice_loss_weight * dice_loss
            
            # BCE損失
            bce_loss = self.bce_loss(predicted_masks, ground_truth_mask)
            losses["bce_loss"] = bce_loss
            total_loss += self.bce_loss_weight * bce_loss
        else:
            # マスクデータが存在しない場合（VQAデータなど）
            losses["dice_loss"] = torch.tensor(0.0, device=total_loss.device)
            losses["bce_loss"] = torch.tensor(0.0, device=total_loss.device)
        
        losses["total_loss"] = total_loss
        
        return losses


class IoUMetric(nn.Module):
    """IoU（Intersection over Union）メトリクスの計算"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測マスク (B, 1, H, W) または (B, H, W)
            target: 正解マスク (B, 1, H, W) または (B, H, W)
        Returns:
            IoUスコア
        """
        # 次元を揃える
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)  # (B, H, W)
        if target.dim() == 4:
            if target.size(1) == 1:
                target = target.squeeze(1)  # (B, H, W)
            elif target.size(1) == 3:
                # RGBの場合はグレースケールに変換（平均を取る）
                target = target.mean(dim=1)  # (B, H, W)
            
        # サイズが異なる場合、targetをpredのサイズにリサイズ
        if pred.shape != target.shape:
            import torch.nn.functional as F
            if target.dim() == 3:  # (B, H, W)
                target = F.interpolate(
                    target.unsqueeze(1),  # (B, 1, H, W)
                    size=(pred.size(1), pred.size(2)),
                    mode='nearest'
                ).squeeze(1)  # (B, H, W)
            elif target.dim() == 2:  # (H, W)
                target = F.interpolate(
                    target.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                    size=(pred.size(1), pred.size(2)),
                    mode='nearest'
                ).squeeze(0).squeeze(0)  # (H, W)
        
        # 予測値を二値化
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()
        
        # バッチ次元以外をフラット化
        pred_flat = pred_binary.view(pred_binary.size(0), -1)  # (B, H*W)
        target_flat = target_binary.view(target_binary.size(0), -1)  # (B, H*W)
        
        # IoUの計算
        intersection = (pred_flat * target_flat).sum(dim=1)  # (B,)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection  # (B,)
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return iou.mean()


def test_losses():
    """損失関数のテスト"""
    print("=== 損失関数のテスト ===")
    
    # ダミーデータの作成
    batch_size = 2
    height, width = 64, 64
    
    # 予測マスクと正解マスク
    pred_masks = torch.randn(batch_size, 1, height, width)
    gt_masks = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # テキスト損失
    text_loss = torch.tensor(2.5)
    
    # モデル出力とバッチデータの作成
    model_outputs = {
        "text_loss": text_loss,
        "predicted_masks": pred_masks,
    }
    
    batch = {
        "ground_truth_mask": gt_masks,
    }
    
    # 複合損失の計算
    loss_fn = CompositeLoss()
    losses = loss_fn(model_outputs, batch)
    
    print(f"総損失: {losses['total_loss'].item():.4f}")
    print(f"テキスト損失: {losses['text_loss'].item():.4f}")
    print(f"DICE損失: {losses['dice_loss'].item():.4f}")
    print(f"BCE損失: {losses['bce_loss'].item():.4f}")
    
    # IoUメトリクスのテスト
    iou_metric = IoUMetric()
    iou_score = iou_metric(pred_masks, gt_masks)
    print(f"IoUスコア: {iou_score.item():.4f}")
    
    print("✓ 損失関数のテスト完了")


if __name__ == "__main__":
    test_losses() 