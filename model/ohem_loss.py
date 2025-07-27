# model/ohem_loss.py
"""
Phase 3B: OHEM損失関数実装

論文準拠: SAM2+MLE \"Customize SAM for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts\"
目標: 学習収束2倍高速化、28.14%性能向上の重要構成要素

Llama-4 + SAM2統合学習最適化:
- Online Hard Example Mining (OHEM) による困難例選択
- Llama-4言語理解損失 + SAM2セグメンテーション損失統合
- Q-Former クロスモーダル融合損失
- 適応的重み調整による効率的学習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import sys
import os
import math

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_linux


class OnlineHardExampleMining(nn.Module):
    """
    Online Hard Example Mining (OHEM) メイン実装
    
    困難例選択により学習効率化:
    - 上位困難例選択（hard_ratio指定）
    - 最小サンプル数保証（min_kept）
    - 適応的閾値調整
    - バッチ内バランス維持
    """
    
    def __init__(
        self,
        hard_ratio: float = 0.25,           # 上位25%困難例選択
        min_kept: int = 512,                # 最小保持サンプル数
        ignore_index: int = 255,            # 無視インデックス
        adaptive_threshold: bool = True,     # 適応的閾値調整
        temperature: float = 1.0             # ソフトマックス温度
    ):
        super().__init__()
        
        self.hard_ratio = hard_ratio
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.adaptive_threshold = adaptive_threshold
        self.temperature = temperature
        
        print(f"🔧 OHEM初期化...")
        print(f"  - 困難例比率: {hard_ratio}")
        print(f"  - 最小サンプル数: {min_kept}")
        print(f"  - 適応的閾値: {adaptive_threshold}")
        
        # 統計情報保持（適応的閾値用）
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_std', torch.tensor(1.0))
        self.register_buffer('num_batches', torch.tensor(0))
        
        print(f"✅ OHEM初期化完了")
    
    def update_statistics(self, losses: torch.Tensor):
        """損失統計更新（適応的閾値用）"""
        if not self.adaptive_threshold:
            return
        
        batch_mean = losses.mean()
        batch_std = losses.std()
        
        if self.num_batches == 0:
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            # 指数移動平均
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_std = (1 - momentum) * self.running_std + momentum * batch_std
        
        self.num_batches += 1
    
    def select_hard_examples(
        self,
        losses: torch.Tensor,           # (N,) pixel-wise losses
        predictions: torch.Tensor,     # (N, C) or (N,) predictions
        targets: torch.Tensor          # (N,) targets
    ) -> torch.Tensor:
        """
        困難例選択実行
        
        Args:
            losses: ピクセル単位損失
            predictions: 予測値
            targets: 正解ラベル
            
        Returns:
            selected_mask: 選択されたピクセルのマスク
        """
        # 有効ピクセル（ignore_indexを除外）
        valid_mask = targets != self.ignore_index
        valid_losses = losses[valid_mask]
        
        if valid_losses.numel() == 0:
            return torch.zeros_like(losses, dtype=torch.bool)
        
        # 統計更新
        self.update_statistics(valid_losses)
        
        # 困難例数計算
        num_valid = valid_losses.numel()
        num_hard = max(
            int(num_valid * self.hard_ratio),
            min(self.min_kept, num_valid)
        )
        
        # 困難例選択
        if self.adaptive_threshold and self.num_batches > 10:
            # 適応的閾値（統計ベース）
            threshold = self.running_mean + 0.5 * self.running_std
            hard_mask = valid_losses > threshold
            
            # 閾値選択で十分でない場合はTop-k補完
            if hard_mask.sum() < num_hard:
                _, hard_indices = torch.topk(valid_losses, num_hard, largest=True)
                hard_mask = torch.zeros_like(valid_losses, dtype=torch.bool)
                hard_mask[hard_indices] = True
        else:
            # Top-k選択（初期段階）
            _, hard_indices = torch.topk(valid_losses, num_hard, largest=True)
            hard_mask = torch.zeros_like(valid_losses, dtype=torch.bool)
            hard_mask[hard_indices] = True
        
        # 全体マスクに適用
        selected_mask = torch.zeros_like(losses, dtype=torch.bool)
        selected_mask[valid_mask] = hard_mask
        
        return selected_mask


class DualModalityOHEMLoss(nn.Module):
    """
    デュアルモダリティOHEM損失関数
    
    Llama-4 + SAM2統合学習のためのOHEM実装:
    - Llama-4: 言語理解 + セマンティック損失
    - SAM2: セグメンテーション損失（BCE, Dice）
    - 統合: デュアルパスウェイ一貫性損失
    """
    
    def __init__(
        self,
        # OHEM設定
        hard_ratio: float = 0.25,
        min_kept: int = 512,
        
        # 損失重み（config_linux準拠）
        ce_loss_weight: float = 1.0,        # Llama-4 CE損失
        bce_loss_weight: float = 2.0,       # SAM2 BCE損失
        dice_loss_weight: float = 0.5,      # SAM2 Dice損失
        consistency_weight: float = 0.1,    # デュアルパス一貫性
        
        # 高度設定
        focal_alpha: float = 0.25,          # Focal Loss alpha
        focal_gamma: float = 2.0,           # Focal Loss gamma
        semantic_weight: float = 0.2,       # セマンティック分類重み
        use_focal_loss: bool = True,        # Focal Loss使用
        temperature: float = 1.0            # ソフトマックス温度
    ):
        super().__init__()
        
        self.hard_ratio = hard_ratio
        self.min_kept = min_kept
        self.ce_loss_weight = ce_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.consistency_weight = consistency_weight
        self.semantic_weight = semantic_weight
        self.use_focal_loss = use_focal_loss
        self.temperature = temperature
        
        print(f"🔧 デュアルモダリティOHEM損失初期化...")
        print(f"  - 困難例比率: {hard_ratio}")
        print(f"  - 損失重み: CE={ce_loss_weight}, BCE={bce_loss_weight}, Dice={dice_loss_weight}")
        print(f"  - Focal Loss: {use_focal_loss}")
        
        # OHEM機構
        self.ohem_selector = OnlineHardExampleMining(
            hard_ratio=hard_ratio,
            min_kept=min_kept,
            temperature=temperature
        )
        
        # 基本損失関数
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Focal Loss パラメータ
        if use_focal_loss:
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma
        
        print(f"✅ デュアルモダリティOHEM損失初期化完了")
    
    def compute_dice_loss(
        self,
        predictions: torch.Tensor,      # (B, 1, H, W)
        targets: torch.Tensor,          # (B, 1, H, W)
        smooth: float = 1e-5
    ) -> torch.Tensor:
        """Dice損失計算"""
        pred_flat = torch.sigmoid(predictions).view(-1)
        target_flat = targets.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
    def compute_focal_loss(
        self,
        predictions: torch.Tensor,      # (B, C, H, W) or (B, H, W)
        targets: torch.Tensor,          # (B, H, W)
        alpha: float = None,
        gamma: float = None
    ) -> torch.Tensor:
        """Focal Loss計算"""
        if alpha is None:
            alpha = self.focal_alpha
        if gamma is None:
            gamma = self.focal_gamma
        
        # Binary case
        if predictions.dim() == targets.dim():
            pred_sigmoid = torch.sigmoid(predictions)
            ce_loss = self.bce_loss(predictions, targets.float())
            
            pt = torch.where(targets == 1, pred_sigmoid, 1 - pred_sigmoid)
            focal_weight = alpha * (1 - pt) ** gamma
            
            return focal_weight * ce_loss
        
        # Multi-class case  
        else:
            ce_loss = self.ce_loss(predictions, targets.long())
            pt = torch.exp(-ce_loss)
            focal_weight = alpha * (1 - pt) ** gamma
            
            return focal_weight * ce_loss
    
    def forward(
        self,
        # Llama-4出力
        llama_logits: torch.Tensor,         # (B, seq_len, vocab_size)
        llama_targets: torch.Tensor,        # (B, seq_len)
        llama_attention_mask: torch.Tensor, # (B, seq_len)
        
        # SAM2セグメンテーション出力
        sam_predictions: torch.Tensor,      # (B, 1, H, W)
        sam_targets: torch.Tensor,          # (B, 1, H, W)
        
        # デュアルパスウェイ出力（オプション）
        aux_predictions: Optional[torch.Tensor] = None,  # (B, 1, H, W)
        semantic_logits: Optional[torch.Tensor] = None,  # (B, num_classes)
        semantic_targets: Optional[torch.Tensor] = None, # (B,)
        
        # 制御パラメータ
        apply_ohem: bool = True,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        統合OHEM損失計算
        
        Returns:
            Dict containing:
                - total_loss: 総損失
                - llama_loss: Llama-4言語損失
                - sam_loss: SAM2セグメンテーション損失
                - consistency_loss: 一貫性損失 [optional]
                - semantic_loss: セマンティック損失 [optional]
        """
        batch_size = sam_predictions.size(0)
        losses = {}
        
        print(f"  🔄 デュアルモダリティOHEM損失計算...")
        print(f"    - Llama logits: {llama_logits.shape}")
        print(f"    - SAM予測: {sam_predictions.shape}")
        print(f"    - SAMターゲット: {sam_targets.shape}")
        
        # 1. Llama-4言語理解損失
        llama_loss = self._compute_llama_loss(
            llama_logits, llama_targets, llama_attention_mask, apply_ohem
        )
        losses['llama_loss'] = llama_loss * self.ce_loss_weight
        
        # 2. SAM2セグメンテーション損失
        sam_loss = self._compute_sam_loss(
            sam_predictions, sam_targets, apply_ohem
        )
        losses['sam_loss'] = sam_loss
        
        # 3. デュアルパスウェイ一貫性損失（オプション）
        if aux_predictions is not None:
            consistency_loss = self._compute_consistency_loss(
                sam_predictions, aux_predictions
            )
            losses['consistency_loss'] = consistency_loss * self.consistency_weight
        
        # 4. セマンティック分類損失（オプション）
        if semantic_logits is not None and semantic_targets is not None:
            semantic_loss = self._compute_semantic_loss(
                semantic_logits, semantic_targets, apply_ohem
            )
            losses['semantic_loss'] = semantic_loss * self.semantic_weight
        
        # 5. 総損失計算
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        print(f"    - 総損失: {total_loss.item():.6f}")
        print(f"    - Llama損失: {losses['llama_loss'].item():.6f}")
        print(f"    - SAM損失: {losses['sam_loss'].item():.6f}")
        
        if return_individual:
            return losses
        else:
            return total_loss
    
    def _compute_llama_loss(
        self,
        logits: torch.Tensor,           # (B, seq_len, vocab_size)
        targets: torch.Tensor,          # (B, seq_len)
        attention_mask: torch.Tensor,   # (B, seq_len)
        apply_ohem: bool
    ) -> torch.Tensor:
        """Llama-4言語理解損失計算"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for loss calculation
        logits_flat = logits.view(-1, vocab_size)  # (B*seq_len, vocab_size)
        targets_flat = targets.view(-1)            # (B*seq_len,)
        mask_flat = attention_mask.view(-1)        # (B*seq_len,)
        
        # 基本CE損失
        if self.use_focal_loss:
            ce_losses = self.compute_focal_loss(logits_flat, targets_flat)
        else:
            ce_losses = self.ce_loss(logits_flat, targets_flat)
        
        # マスク適用
        valid_mask = (mask_flat == 1) & (targets_flat != -100)
        valid_losses = ce_losses[valid_mask]
        
        if valid_losses.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # OHEM適用
        if apply_ohem:
            dummy_preds = torch.zeros_like(valid_losses)
            dummy_targets = torch.zeros_like(valid_losses)
            hard_mask = self.ohem_selector.select_hard_examples(
                valid_losses, dummy_preds, dummy_targets
            )
            
            if hard_mask.sum() > 0:
                return valid_losses[hard_mask].mean()
        
        return valid_losses.mean()
    
    def _compute_sam_loss(
        self,
        predictions: torch.Tensor,      # (B, 1, H, W)
        targets: torch.Tensor,          # (B, 1, H, W)
        apply_ohem: bool
    ) -> torch.Tensor:
        """SAM2セグメンテーション損失計算"""
        batch_size, channels, height, width = predictions.shape
        
        # Flatten for loss calculation
        pred_flat = predictions.view(-1)      # (B*H*W,)
        target_flat = targets.view(-1)        # (B*H*W,)
        
        # BCE損失
        if self.use_focal_loss:
            bce_losses = self.compute_focal_loss(pred_flat, target_flat)
        else:
            bce_losses = self.bce_loss(pred_flat, target_flat)
        
        # Dice損失
        dice_loss = self.compute_dice_loss(predictions, targets)
        
        # OHEM適用（BCE損失に対して）
        if apply_ohem and bce_losses.numel() > 0:
            dummy_preds = torch.sigmoid(pred_flat)
            hard_mask = self.ohem_selector.select_hard_examples(
                bce_losses, dummy_preds, target_flat
            )
            
            if hard_mask.sum() > 0:
                bce_loss_final = bce_losses[hard_mask].mean()
            else:
                bce_loss_final = bce_losses.mean()
        else:
            bce_loss_final = bce_losses.mean()
        
        # 統合SAM損失
        total_sam_loss = (
            self.bce_loss_weight * bce_loss_final +
            self.dice_loss_weight * dice_loss
        )
        
        return total_sam_loss
    
    def _compute_consistency_loss(
        self,
        main_predictions: torch.Tensor,     # (B, 1, H, W)
        aux_predictions: torch.Tensor       # (B, 1, H, W)
    ) -> torch.Tensor:
        """デュアルパスウェイ一貫性損失計算"""
        # KL divergence一貫性制約
        main_prob = torch.sigmoid(main_predictions / self.temperature)
        aux_prob = torch.sigmoid(aux_predictions / self.temperature)
        
        # Flatten for KL divergence
        main_flat = main_prob.view(main_prob.size(0), -1)
        aux_flat = aux_prob.view(aux_prob.size(0), -1)
        
        # Add small epsilon for numerical stability
        eps = 1e-8
        main_flat = torch.clamp(main_flat, eps, 1-eps)
        aux_flat = torch.clamp(aux_flat, eps, 1-eps)
        
        # KL divergence
        kl_loss = F.kl_div(
            torch.log(main_flat),
            aux_flat,
            reduction='batchmean'
        )
        
        return kl_loss
    
    def _compute_semantic_loss(
        self,
        logits: torch.Tensor,           # (B, num_classes)
        targets: torch.Tensor,          # (B,)
        apply_ohem: bool
    ) -> torch.Tensor:
        """セマンティック分類損失計算"""
        if self.use_focal_loss:
            losses = self.compute_focal_loss(logits, targets)
        else:
            losses = self.ce_loss(logits, targets)
        
        if apply_ohem and losses.numel() > 0:
            dummy_preds = torch.softmax(logits, dim=1)
            hard_mask = self.ohem_selector.select_hard_examples(
                losses, dummy_preds, targets
            )
            
            if hard_mask.sum() > 0:
                return losses[hard_mask].mean()
        
        return losses.mean()


def create_ohem_loss(
    hard_ratio: float = 0.25,
    config_override: Optional[Dict[str, Any]] = None
) -> DualModalityOHEMLoss:
    """
    OHEM損失関数ファクトリ
    
    Args:
        hard_ratio: 困難例選択比率
        config_override: config_linux設定上書き
        
    Returns:
        DualModalityOHEMLoss instance
    """
    print(f"🔄 OHEM損失関数作成中...")
    
    # config_linux設定取得
    loss_config = config_linux.get_loss_config()
    
    # 設定上書き適用
    if config_override:
        loss_config.update(config_override)
    
    ohem_loss = DualModalityOHEMLoss(
        hard_ratio=hard_ratio,
        ce_loss_weight=loss_config['ce_loss_weight'],
        bce_loss_weight=loss_config['bce_loss_weight'],
        dice_loss_weight=loss_config['dice_loss_weight']
    )
    
    print(f"✅ OHEM損失関数作成完了")
    
    return ohem_loss


if __name__ == "__main__":
    print("=== OHEM損失関数テスト ===")
    
    # テスト用ダミーデータ
    batch_size = 2
    seq_len = 16
    vocab_size = 32000
    height, width = 448, 448
    num_classes = 1000
    
    # Llama-4ダミー出力
    llama_logits = torch.randn(batch_size, seq_len, vocab_size)
    llama_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    llama_attention_mask = torch.ones(batch_size, seq_len)
    
    # SAM2ダミー出力
    sam_predictions = torch.randn(batch_size, 1, height, width)
    sam_targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # 補助出力（デュアルパスウェイ）
    aux_predictions = torch.randn(batch_size, 1, height, width)
    
    # セマンティック分類
    semantic_logits = torch.randn(batch_size, num_classes)
    semantic_targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"テストデータ準備完了:")
    print(f"  - Llama logits: {llama_logits.shape}")
    print(f"  - SAM予測: {sam_predictions.shape}")
    print(f"  - 補助予測: {aux_predictions.shape}")
    print(f"  - セマンティック: {semantic_logits.shape}")
    
    # OHEM損失関数作成
    ohem_loss = create_ohem_loss(hard_ratio=0.25)
    
    # 損失計算テスト
    print(f"\n損失計算テスト:")
    
    with torch.no_grad():
        # 個別損失取得
        losses = ohem_loss(
            llama_logits=llama_logits,
            llama_targets=llama_targets,
            llama_attention_mask=llama_attention_mask,
            sam_predictions=sam_predictions,
            sam_targets=sam_targets,
            aux_predictions=aux_predictions,
            semantic_logits=semantic_logits,
            semantic_targets=semantic_targets,
            apply_ohem=True,
            return_individual=True
        )
        
        print(f"損失結果:")
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                print(f"  - {loss_name}: {loss_value.item():.6f}")
            else:
                print(f"  - {loss_name}: {loss_value}")
    
    # OHEM無効テスト
    print(f"\nOHEM無効時比較:")
    with torch.no_grad():
        total_loss_ohem = ohem_loss(
            llama_logits=llama_logits,
            llama_targets=llama_targets,
            llama_attention_mask=llama_attention_mask,
            sam_predictions=sam_predictions,
            sam_targets=sam_targets,
            apply_ohem=True,
            return_individual=False
        )
        
        total_loss_normal = ohem_loss(
            llama_logits=llama_logits,
            llama_targets=llama_targets,
            llama_attention_mask=llama_attention_mask,
            sam_predictions=sam_predictions,
            sam_targets=sam_targets,
            apply_ohem=False,
            return_individual=False
        )
        
        print(f"  - OHEM有効: {total_loss_ohem.item():.6f}")
        print(f"  - OHEM無効: {total_loss_normal.item():.6f}")
        print(f"  - 差異: {abs(total_loss_ohem.item() - total_loss_normal.item()):.6f}")
    
    print(f"\n✅ OHEM損失関数テスト完了")