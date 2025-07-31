"""
Simple Loss Functions for SAM2.1 + Qwen2.5-VL Integration
シンプルな損失関数の実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    Commonly used with SAM models for better segmentation quality.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, H, W] or [B, 1, H, W]
            target: Ground truth labels [B, H, W] or [B, 1, H, W]
        
        Returns:
            Focal loss value
        """
        # Ensure same shape
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        
        # Convert to probabilities
        if pred.dtype != torch.float32:
            pred = pred.float()
        if target.dtype != torch.float32:
            target = target.float()
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute probabilities
        p_t = torch.exp(-bce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Apply focal term
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Directly optimizes for Dice coefficient (F1-score).
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, H, W] or [B, 1, H, W]
            target: Ground truth labels [B, H, W] or [B, 1, H, W]
        
        Returns:
            Dice loss value
        """
        # Ensure same shape
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        
        # Convert pred to probabilities if logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_coeff = (2 * intersection + self.smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)
        
        # Dice loss = 1 - Dice coefficient
        dice_loss = 1 - dice_coeff
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss function for segmentation tasks.
    Combines Focal Loss and Dice Loss for better performance.
    """
    
    def __init__(self, 
                 focal_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Args:
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Predicted logits/probabilities
            target: Ground truth labels
        
        Returns:
            Dictionary containing individual and total losses
        """
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total = self.focal_weight * focal + self.dice_weight * dice
        
        return {
            'focal_loss': focal,
            'dice_loss': dice,
            'total_loss': total,
            'segmentation_loss': total  # Alias for compatibility
        }


class MultiModalLoss(nn.Module):
    """
    Multi-modal loss function for SAM-Qwen integrated model.
    Combines segmentation loss and text generation loss.
    """
    
    def __init__(self,
                 seg_weight: float = 1.0,
                 text_weight: float = 1.0,
                 seg_loss_type: str = 'combined'):
        """
        Args:
            seg_weight: Weight for segmentation loss
            text_weight: Weight for text generation loss  
            seg_loss_type: Type of segmentation loss ('focal', 'dice', 'combined')
        """
        super().__init__()
        self.seg_weight = seg_weight
        self.text_weight = text_weight
        
        # Initialize segmentation loss
        if seg_loss_type == 'focal':
            self.seg_loss_fn = FocalLoss()
        elif seg_loss_type == 'dice':
            self.seg_loss_fn = DiceLoss()
        elif seg_loss_type == 'combined':
            self.seg_loss_fn = CombinedSegmentationLoss()
        else:
            raise ValueError(f"Unknown seg_loss_type: {seg_loss_type}")
        
        # Text generation loss (standard cross-entropy)
        self.text_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, 
                pred_masks: Optional[torch.Tensor] = None,
                target_masks: Optional[torch.Tensor] = None,
                pred_text_logits: Optional[torch.Tensor] = None,
                target_text_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_masks: Predicted segmentation masks
            target_masks: Ground truth segmentation masks
            pred_text_logits: Predicted text logits
            target_text_ids: Target text token IDs
        
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Compute segmentation loss
        if pred_masks is not None and target_masks is not None:
            if isinstance(self.seg_loss_fn, CombinedSegmentationLoss):
                seg_losses = self.seg_loss_fn(pred_masks, target_masks)
                losses.update(seg_losses)
                seg_loss = seg_losses['total_loss']
            else:
                seg_loss = self.seg_loss_fn(pred_masks, target_masks)
                losses['segmentation_loss'] = seg_loss
            
            total_loss += self.seg_weight * seg_loss
        
        # Compute text generation loss
        if pred_text_logits is not None and target_text_ids is not None:
            # Reshape for cross-entropy loss
            pred_flat = pred_text_logits.view(-1, pred_text_logits.size(-1))
            target_flat = target_text_ids.view(-1)
            
            text_loss = self.text_loss_fn(pred_flat, target_flat)
            losses['text_loss'] = text_loss
            
            total_loss += self.text_weight * text_loss
        
        losses['total_loss'] = total_loss
        return losses


def create_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function based on configuration.
    
    Args:
        config: Loss configuration dictionary
    
    Returns:
        Loss function instance
    """
    loss_type = config.get('type', 'multimodal')
    
    if loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
    elif loss_type == 'dice':
        return DiceLoss(smooth=config.get('smooth', 1e-6))
    elif loss_type == 'combined_seg':
        return CombinedSegmentationLoss(
            focal_weight=config.get('focal_weight', 1.0),
            dice_weight=config.get('dice_weight', 1.0)
        )
    elif loss_type == 'multimodal':
        return MultiModalLoss(
            seg_weight=config.get('segmentation_loss_weight', 1.0),
            text_weight=config.get('text_loss_weight', 1.0),
            seg_loss_type=config.get('seg_loss_type', 'combined')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Export functions and classes
__all__ = [
    'FocalLoss',
    'DiceLoss', 
    'CombinedSegmentationLoss',
    'MultiModalLoss',
    'create_loss_function'
]