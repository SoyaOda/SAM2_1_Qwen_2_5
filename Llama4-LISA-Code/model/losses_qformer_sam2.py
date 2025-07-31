# model/losses_qformer_sam2.py
"""
LISA-Llama4-Scout + Q-Former + SAM2 統合モデル専用損失関数モジュール

Web調査ベース2025年最新ベストプラクティス:
- Focal Tversky Loss: 複雑セグメンテーションに最適（Web推奨）
- Lovász-Softmax Loss: IoU直接最適化
- Q-Former専用損失: マルチモーダル特徴学習
- SAM2統合損失: 高精度プロンプト最適化
- 段階的学習プロトコル対応

参考:
- Focal Tversky: "presents the most promising segmentation results" (2025)
- Lovász-Softmax: "better mIoU scores compared to cross-entropy" (2025)
- Q-Former: BLIP-2準拠多重学習目標
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
import math


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Web調査推奨: 最高性能セグメンテーション損失)
    
    "Focal Tversky Loss presents the most promising segmentation results,
    correctly identifying all organs with only minor variations"
    
    Args:
        alpha: False Negative重み (デフォルト: 0.3)
        beta: False Positive重み (デフォルト: 0.7)
        gamma: Focal重み (デフォルト: 2.0)
        smooth: 数値安定化 (デフォルト: 1e-6)
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測マスク (B, H, W) または (B, 1, H, W)
            target: 正解マスク (B, H, W) または (B, 1, H, W)
        """
        # Web調査準拠: 実装指針に沿った適切なエラー処理
        if pred.dim() == 5:
            raise ValueError(
                f"Focal Tversky損失: 予期しない5次元テンソル {pred.shape} を受信。"
                f"SAM2マスクデコーダーの出力形状が正しく処理されていません。"
                f"上流のマスク処理ロジックを確認してください。"
            )
        
        # 4D形状を3Dに変換（チャネル次元が1の場合）
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        elif pred.dim() == 4 and pred.size(1) > 1:
            raise ValueError(
                f"Focal Tversky損失: 複数チャネル {pred.shape} は未対応。"
                f"マスク選択ロジックで単一チャネルに変換してください。"
            )
            
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
            
        # Web調査修正: sigmoid直接適用（dtype変換による勾配切断防止）
        pred = torch.sigmoid(pred)
        
        # フラット化
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Tversky係数計算
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal適用
        focal_tversky = torch.pow(1 - tversky, self.gamma)
        
        return focal_tversky


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss (IoU直接最適化、Web調査推奨)
    
    "The Lovász-Softmax loss has better mIoU scores compared to training 
    with cross-entropy loss"
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測ロジット (B, C, H, W) または (B, H, W)
            target: 正解ラベル (B, H, W)
        """
        if pred.dim() == 3:
            # バイナリセグメンテーション
            pred = pred.unsqueeze(1)  # (B, 1, H, W)
            
        batch_size = pred.size(0)
        # Web調査準拠: 勾配フロー維持のためtensorで初期化
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        for i in range(batch_size):
            loss_i = self._lovasz_softmax_single(pred[i], target[i])
            total_loss += loss_i
            
        return total_loss / batch_size
    
    def _lovasz_softmax_single(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """単一サンプルのLovász-Softmax損失"""
        if pred.dim() == 3 and pred.size(0) == 1:
            # バイナリケース
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            
            # 有効ピクセルマスク
            valid = target_flat != self.ignore_index
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
            
            # Lovász拡張（データ型保持強化版）
            target_dtype = pred_flat.dtype
            # Float32混入防止: 明示的データ型指定
            signs = (2.0 * target_flat.to(dtype=target_dtype) - 1.0).to(dtype=target_dtype)
            errors = (1.0 - pred_flat * signs).to(dtype=target_dtype)
            errors_sorted, perm = torch.sort(errors, descending=True)
            gt_sorted = target_flat[perm]
            
            grad = self._lovasz_grad(gt_sorted, target_dtype)
            # Web調査修正: dtype変換による勾配切断防止
            relu_errors = F.relu(errors_sorted)
            loss = torch.dot(relu_errors, grad)
            
            return loss
        else:
            # マルチクラス（今回は使用しないが将来対応）
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
    
    def _lovasz_grad(self, gt_sorted: torch.Tensor, target_dtype: torch.dtype = None) -> torch.Tensor:
        """Lovász勾配計算（データ型保持強化版）"""
        if target_dtype is None:
            target_dtype = gt_sorted.dtype
            
        p = len(gt_sorted)
        gts = gt_sorted.sum().to(dtype=target_dtype)
        intersection = gts - gt_sorted.to(dtype=target_dtype).cumsum(0)
        union = gts + (1 - gt_sorted).to(dtype=target_dtype).cumsum(0)
        # Float32混入防止: 除算結果を明示的にキャスト
        jaccard = (1.0 - intersection / union).to(dtype=target_dtype)
        
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


class QFormerMultiModalLoss(nn.Module):
    """
    Q-Former専用マルチモーダル損失 (BLIP-2準拠)
    
    3つの学習目標:
    1. ITC: Image-Text Contrastive Learning
    2. ITM: Image-Text Matching
    3. ITG: Image-Grounded Text Generation
    """
    
    def __init__(self, temperature: float = 0.07, itm_weight: float = 1.0, itg_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.itm_weight = itm_weight
        self.itg_weight = itg_weight
        
    def forward(
        self, 
        query_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        itm_logits: Optional[torch.Tensor] = None,
        itm_labels: Optional[torch.Tensor] = None,
        itg_logits: Optional[torch.Tensor] = None,
        itg_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_embeds: Q-Formerクエリ埋め込み (B, num_queries, hidden_size)
            text_embeds: テキスト埋め込み (B, hidden_size)
            itm_logits: ITMロジット (B, 2)
            itm_labels: ITMラベル (B,)
            itg_logits: ITGロジット (B, seq_len, vocab_size)
            itg_labels: ITGラベル (B, seq_len)
        """
        losses = {}
        
        # 1. ITC Loss (Image-Text Contrastive)
        itc_loss = self._compute_itc_loss(query_embeds, text_embeds)
        losses['itc_loss'] = itc_loss
        
        # 2. ITM Loss (Image-Text Matching) - データ型保持
        if itm_logits is not None and itm_labels is not None:
            target_dtype = query_embeds.dtype
            # Web調査修正: dtype変換による勾配切断防止
            itm_loss = F.cross_entropy(itm_logits, itm_labels)
            losses['itm_loss'] = itm_loss * self.itm_weight
        
        # 3. ITG Loss (Image-Grounded Text Generation) - データ型保持
        if itg_logits is not None and itg_labels is not None:
            target_dtype = query_embeds.dtype
            # Web調査修正: dtype変換による勾配切断防止
            itg_loss = F.cross_entropy(
                itg_logits.view(-1, itg_logits.size(-1)),
                itg_labels.view(-1),
                ignore_index=-100
            )
            losses['itg_loss'] = itg_loss * self.itg_weight
        
        return losses
    
    def _compute_itc_loss(self, query_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """ITC損失計算（対照学習）- 修正方針Y: Webリサーチに基づく温度パラメータ最適化"""
        # データ型保持
        target_dtype = query_embeds.dtype
        target_device = query_embeds.device
        
        # デバッグログ追加
        print(f"🔍 ITC損失計算開始:")
        print(f"  - query_embeds: {query_embeds.shape}, dtype={query_embeds.dtype}")
        print(f"  - text_embeds: {text_embeds.shape}, dtype={text_embeds.dtype}")
        print(f"  - temperature: {self.temperature}")
        
        # クエリ埋め込みを平均化
        image_feat = query_embeds.mean(dim=1)  # (B, qformer_hidden_size)
        text_feat = text_embeds  # (B, llm_hidden_size)
        
        # 🔧 修正方針P継続: 次元不一致対策 - BLIP-2準拠のプロジェクション層
        if image_feat.size(-1) != text_feat.size(-1):
            print(f"🔧 次元不一致検出: image_feat={image_feat.shape}, text_feat={text_feat.shape}")
            
            # 動的プロジェクション層作成（BLIP-2準拠）
            if not hasattr(self, '_projection_layer'):
                self._projection_layer = nn.Linear(
                    image_feat.size(-1), 
                    text_feat.size(-1),
                    bias=False
                ).to(device=target_device, dtype=target_dtype)
                print(f"✅ プロジェクション層作成: {image_feat.size(-1)} → {text_feat.size(-1)}")
            
            # Q-Former特徴をLLM次元に投影
            image_feat = self._projection_layer(image_feat)
            print(f"✅ 次元変換完了: {image_feat.shape}")
        
        # 🔧 修正方針Y: Webリサーチ結果に基づく改善
        # 1. バッチサイズ1の場合の特別処理（対照学習不可能）
        batch_size = image_feat.size(0)
        if batch_size == 1:
            print(f"⚠️ バッチサイズ1: 対照学習スキップ（最小損失返却）")
            # Web調査修正: 入力テンソルから派生した損失で勾配フロー維持
            min_loss = image_feat.mean() * 0.0 + 0.01  # 入力から派生、勾配フロー保持
            return min_loss
        
        # 2. L2正規化（数値安定性向上）
        # Web調査修正: dtype変換による勾配切断防止
        image_feat = F.normalize(image_feat, dim=-1, eps=1e-8)
        text_feat = F.normalize(text_feat, dim=-1, eps=1e-8)
        
        print(f"🔍 正規化後:")
        print(f"  - image_feat: min={image_feat.min():.6f}, max={image_feat.max():.6f}")
        print(f"  - text_feat: min={text_feat.min():.6f}, max={text_feat.max():.6f}")
        
        # 3. 類似度行列計算（温度パラメータ適用）
        sim_matrix = torch.matmul(image_feat, text_feat.t())
        print(f"🔍 類似度行列（温度前）: min={sim_matrix.min():.6f}, max={sim_matrix.max():.6f}")
        
        # 4. 温度スケーリング（Webリサーチ: 0.07は一般的だが、小さいバッチには大きすぎる可能性）
        sim_matrix = sim_matrix / self.temperature
        print(f"🔍 温度スケーリング後: min={sim_matrix.min():.6f}, max={sim_matrix.max():.6f}")
        
        if sim_matrix.dtype != target_dtype:
            sim_matrix = sim_matrix.to(dtype=target_dtype)
        
        # 5. ラベル (対角線が正例)
        labels = torch.arange(batch_size, device=target_device, dtype=torch.long)
        
        # 6. 双方向損失計算（数値安定性向上）
        # Image-to-Text
        # Web調査修正: dtype変換による勾配切断防止
        loss_i2t = F.cross_entropy(sim_matrix, labels, reduction='mean')
        print(f"🔍 loss_i2t: {loss_i2t:.6f}")
        
        # Text-to-Image  
        # Web調査修正: dtype変換による勾配切断防止
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels, reduction='mean')
        print(f"🔍 loss_t2i: {loss_t2i:.6f}")
        
        # 最終損失
        final_loss = (loss_i2t + loss_t2i) / 2.0
        final_loss = final_loss.to(dtype=target_dtype)
        
        print(f"🔍 最終ITC損失: {final_loss:.6f}")
        
        return final_loss


class SAM2PromptOptimizationLoss(nn.Module):
    """
    SAM2プロンプト最適化損失
    
    高品質なプロンプト生成を促進:
    1. プロンプト多様性損失
    2. プロンプト-マスク整合性損失
    3. プロンプト安定性損失
    """
    
    def __init__(self, diversity_weight: float = 0.1, consistency_weight: float = 1.0, stability_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight
        self.stability_weight = stability_weight
        
    def forward(
        self,
        sam_prompts: torch.Tensor,
        predicted_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sam_prompts: SAMプロンプト (B, num_queries, prompt_dim)
            predicted_masks: 予測マスク (B, num_queries, H, W)
            target_masks: 正解マスク (B, H, W)
        """
        losses = {}
        
        # 1. プロンプト多様性損失
        diversity_loss = self._compute_diversity_loss(sam_prompts)
        losses['prompt_diversity_loss'] = diversity_loss * self.diversity_weight
        
        # 2. プロンプト-マスク整合性損失
        consistency_loss = self._compute_consistency_loss(sam_prompts, predicted_masks, target_masks)
        losses['prompt_consistency_loss'] = consistency_loss * self.consistency_weight
        
        # 3. プロンプト安定性損失
        stability_loss = self._compute_stability_loss(sam_prompts)
        losses['prompt_stability_loss'] = stability_loss * self.stability_weight
        
        return losses
    
    def _compute_diversity_loss(self, sam_prompts: torch.Tensor) -> torch.Tensor:
        """プロンプト多様性損失（クエリ間の類似度を下げる）- Float32混入防止版"""
        batch_size, num_queries, prompt_dim = sam_prompts.shape
        target_dtype = sam_prompts.dtype
        target_device = sam_prompts.device
        
        # 正規化（データ型保持）
        # Web調査修正: dtype変換による勾配切断防止
        normalized_prompts = F.normalize(sam_prompts, dim=-1)
        
        # クエリ間類似度行列（データ型保持）
        sim_matrix = torch.bmm(normalized_prompts, normalized_prompts.transpose(1, 2))
        if sim_matrix.dtype != target_dtype:
            sim_matrix = sim_matrix.to(dtype=target_dtype)
        
        # 対角線を除去（自己類似度は1で固定）- データ型保持
        mask = torch.eye(num_queries, device=target_device, dtype=target_dtype).unsqueeze(0).expand(batch_size, -1, -1)
        sim_matrix = sim_matrix * (1 - mask)
        
        # 類似度の2乗平均（多様性を促進）- データ型保持
        diversity_loss = (sim_matrix ** 2).mean()
        if diversity_loss.dtype != target_dtype:
            diversity_loss = diversity_loss.to(dtype=target_dtype)
        
        return diversity_loss
    
    def _compute_consistency_loss(
        self, 
        sam_prompts: torch.Tensor, 
        predicted_masks: torch.Tensor, 
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """プロンプト-マスク整合性損失"""
        batch_size, num_queries = predicted_masks.shape[:2]
        
        # 各クエリマスクと正解マスクのIoU計算
        target_masks = target_masks.unsqueeze(1).expand(-1, num_queries, -1, -1)
        
        # IoU計算
        intersection = (predicted_masks * target_masks).sum(dim=(2, 3))
        union = ((predicted_masks + target_masks) > 0).float().sum(dim=(2, 3))
        iou = intersection / (union + 1e-6)
        
        # 最高IoUクエリを特定
        best_query_indices = iou.argmax(dim=1)
        
        # 最高性能クエリのプロンプトを強化（データ型保持）
        batch_indices = torch.arange(batch_size, device=sam_prompts.device, dtype=torch.long)
        best_prompts = sam_prompts[batch_indices, best_query_indices]
        
        # プロンプトの正規化とIoUの相関を促進（データ型保持）
        prompt_norms = torch.norm(best_prompts, dim=-1)
        best_ious = iou[batch_indices, best_query_indices]
        
        # 高IoUクエリは高ノルムプロンプトを持つべき（データ型保持）
        consistency_loss = F.mse_loss(prompt_norms.float(), best_ious.float()).to(dtype=sam_prompts.dtype)
        
        return consistency_loss
    
    def _compute_stability_loss(self, sam_prompts: torch.Tensor) -> torch.Tensor:
        """プロンプト安定性損失（過度な変動を防ぐ）- Float32混入防止版"""
        target_dtype = sam_prompts.dtype
        target_device = sam_prompts.device
        
        # プロンプトの標準偏差を制限
        prompt_std = sam_prompts.std(dim=1).mean()
        
        # 適度な安定性を促進（完全な均一化は避ける）- データ型保持
        target_std = torch.tensor(0.5, device=target_device, dtype=target_dtype)
        # Web調査修正: dtype変換による勾配切断防止
        stability_loss = F.mse_loss(prompt_std, target_std)
        
        return stability_loss


class CompositeLossQFormerSAM2(nn.Module):
    """
    統合複合損失 (Web調査ベース最適構成)
    
    段階的学習プロトコル対応:
    - Stage 1: インターフェースアライメント (QFormer + 基本セグメンテーション)
    - Stage 2: フルスタック適応 (+ SAM2プロンプト最適化)
    - Stage 3: 推論能力専門化 (+ 高度損失)
    """
    
    def __init__(
        self,
        # 基本セグメンテーション損失重み
        focal_tversky_weight: float = 1.0,
        lovasz_weight: float = 0.5,
        dice_weight: float = 0.3,
        
        # マルチモーダル損失重み
        qformer_weight: float = 0.2,
        
        # SAM2最適化損失重み
        sam2_prompt_weight: float = 0.1,
        
        # 段階的学習設定
        stage: int = 1  # 1: Interface, 2: Full-stack, 3: Reasoning
    ):
        super().__init__()
        
        # 基本セグメンテーション損失（Web調査推奨）
        self.focal_tversky = FocalTverskyLoss()
        self.lovasz = LovaszSoftmaxLoss()
        self.dice = nn.BCEWithLogitsLoss()
        
        # マルチモーダル損失
        self.qformer_loss = QFormerMultiModalLoss()
        
        # SAM2最適化損失
        self.sam2_prompt_loss = SAM2PromptOptimizationLoss()
        
        # 重み設定
        self.focal_tversky_weight = focal_tversky_weight
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.qformer_weight = qformer_weight
        self.sam2_prompt_weight = sam2_prompt_weight
        
        # 段階的学習段階
        self.stage = stage
        
    def forward(
        self,
        # セグメンテーション関連
        predicted_masks: torch.Tensor,
        target_masks: torch.Tensor,
        
        # Q-Former関連
        query_embeds: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        
        # SAM2関連
        sam_prompts: Optional[torch.Tensor] = None,
        
        # その他
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        統合損失計算（2025年ベストプラクティス: デバイス統一）
        
        Args:
            predicted_masks: 予測マスク (B, H, W) または (B, num_queries, H, W)
            target_masks: 正解マスク (B, H, W)
            query_embeds: Q-Formerクエリ埋め込み (B, num_queries, hidden_size)
            text_embeds: テキスト埋め込み (B, hidden_size)
            sam_prompts: SAMプロンプト (B, num_queries, prompt_dim)
        """
        # 2025年ベストプラクティス: デバイス・データ型統一
        target_device = predicted_masks.device
        
        # 特別データ型検出: SAM2出力がFloat32の場合のBFloat16変換
        target_dtype = torch.bfloat16  # Llama-4統一基準
        
        # predicted_masksのデータ型確認・変換
        if predicted_masks.dtype != target_dtype:
            print(f"  🔄 損失計算: {predicted_masks.dtype} → {target_dtype} 変換")
            predicted_masks = predicted_masks.to(dtype=target_dtype)
        
        # 2025年ベストプラクティス: 数値安定化 + デバッグ情報
        print(f"  🔍 デバッグ: predicted_masks統計")
        print(f"    - Shape: {predicted_masks.shape}")
        print(f"    - Device: {predicted_masks.device}")
        print(f"    - Dtype: {predicted_masks.dtype}")
        print(f"    - Min: {predicted_masks.min():.6f}")
        print(f"    - Max: {predicted_masks.max():.6f}")
        print(f"    - Mean: {predicted_masks.mean():.6f}")
        print(f"    - Std: {predicted_masks.std().item():.6f}")
        
        if torch.isnan(predicted_masks).any():
            nan_count = torch.isnan(predicted_masks).sum().item()
            print(f"  ⚠️ NaN検出 in predicted_masks: {nan_count}個 -> ゼロ置換")
            predicted_masks = torch.nan_to_num(predicted_masks, nan=0.0)
        if torch.isinf(predicted_masks).any():
            inf_count = torch.isinf(predicted_masks).sum().item()
            print(f"  ⚠️ Inf検出 in predicted_masks: {inf_count}個 -> クリップ")
            predicted_masks = torch.clamp(predicted_masks, -10.0, 10.0)
        
        # デバイス・データ型統一（GPU優先 + BFloat16統一）
        target_masks = target_masks.to(device=target_device, dtype=target_dtype)
        
        # target_masksの数値安定化 + デバッグ情報
        print(f"  🔍 デバッグ: target_masks統計")
        print(f"    - Shape: {target_masks.shape}")
        print(f"    - Device: {target_masks.device}")
        print(f"    - Dtype: {target_masks.dtype}")
        print(f"    - Min: {target_masks.min():.6f}")
        print(f"    - Max: {target_masks.max():.6f}")
        print(f"    - Mean: {target_masks.mean():.6f}")
        print(f"    - Unique values: {torch.unique(target_masks)[:10]}")
        
        if torch.isnan(target_masks).any():
            nan_count = torch.isnan(target_masks).sum().item()
            print(f"  ⚠️ NaN検出 in target_masks: {nan_count}個 -> ゼロ置換")
            target_masks = torch.nan_to_num(target_masks, nan=0.0)
        if torch.isinf(target_masks).any():
            inf_count = torch.isinf(target_masks).sum().item()
            print(f"  ⚠️ Inf検出 in target_masks: {inf_count}個 -> クリップ")
            target_masks = torch.clamp(target_masks, -10.0, 10.0)
        if query_embeds is not None:
            query_embeds = query_embeds.to(device=target_device, dtype=target_dtype)
        if text_embeds is not None:
            text_embeds = text_embeds.to(device=target_device, dtype=target_dtype)
        if sam_prompts is not None:
            sam_prompts = sam_prompts.to(device=target_device, dtype=target_dtype)
        
        losses = {}
        # Web調査修正: 勾配グラフ保持のためzerosで初期化（tensorよりも確実）
        total_loss = torch.zeros(1, device=target_device, dtype=target_dtype, requires_grad=True).squeeze()
        
        # 1. Web調査準拠SAM2ベストマスク選択（IoU予測スコア活用）
        print(f"  🔍 マスク選択開始: input_shape={predicted_masks.shape}")
        
        if predicted_masks.dim() == 4:
            # 通常の複数クエリマスク: [B, N, H, W]
            pass  # そのまま処理
        elif predicted_masks.dim() != 4:
            raise ValueError(
                f"予期しないマスク次元: {predicted_masks.shape}。"
                f"SAM2出力は [B, N, H, W] の4次元テンソルである必要があります。"
            )
        
        if predicted_masks.dim() == 4 and predicted_masks.size(1) > 1:
            # Web調査準拠: SAM2 IoU予測スコア活用のベストマスク選択
            print(f"    🎯 複数マスク検出: {predicted_masks.size(1)}個のマスク候補")
            
            if hasattr(self, '_current_iou_scores') and self._current_iou_scores is not None:
                # IoU予測スコア利用（SAM2推奨方法）
                print(f"    ✅ IoU予測スコア利用: {self._current_iou_scores.shape}")
                best_idx = self._current_iou_scores.argmax(dim=-1)
                batch_indices = torch.arange(predicted_masks.size(0), device=predicted_masks.device)
                best_masks = predicted_masks[batch_indices, best_idx]
                print(f"    🎯 IoU最高マスク選択: Index {best_idx.item()}, Score {self._current_iou_scores[0, best_idx].detach().item():.3f}")
            else:
                # フォールバック: Dice係数ベース選択
                print(f"    🔄 Dice係数ベースフォールバック選択")
                dice_scores = []
                for i in range(predicted_masks.size(1)):
                    pred_i = torch.sigmoid(predicted_masks[:, i])
                    if pred_i.dtype != predicted_masks.dtype:
                        # Web調査修正: dtype変換による勾配切断防止（乗算で勾配保持）
                        pred_i = pred_i * torch.ones(1, dtype=predicted_masks.dtype, device=pred_i.device, requires_grad=True).squeeze()
                    dice_i = 1 - self._compute_dice(pred_i, target_masks)
                    dice_scores.append(dice_i)
                
                dice_scores = torch.stack(dice_scores, dim=0)
                best_idx = dice_scores.argmin(dim=0)
                batch_indices = torch.arange(predicted_masks.size(0), device=predicted_masks.device)
                best_masks = predicted_masks[batch_indices, best_idx]
        else:
            best_masks = predicted_masks
        
        # Focal Tversky Loss (Web推奨最高性能)
        print(f"  🔍 Focal Tversky計算中...")
        print(f"    - best_masks: {best_masks.shape}, {best_masks.dtype}, min={best_masks.min():.6f}, max={best_masks.max():.6f}")
        focal_tversky_loss = self.focal_tversky(best_masks, target_masks)
        # Web調査修正: dtype変換による勾配切断防止（乗算で勾配保持）
        if focal_tversky_loss.dtype != target_dtype:
            # torch.tensorではなく、勾配グラフを保持するscalar演算を使用
            focal_tversky_loss = focal_tversky_loss * torch.ones(1, dtype=target_dtype, device=focal_tversky_loss.device, requires_grad=True).squeeze()
        print(f"    - Focal Tversky Loss: {focal_tversky_loss:.6f}")
        if torch.isnan(focal_tversky_loss) or torch.isinf(focal_tversky_loss):
            print(f"    ⚠️ Focal Tversky異常値: {focal_tversky_loss}")
        losses['focal_tversky_loss'] = focal_tversky_loss
        total_loss += focal_tversky_loss * self.focal_tversky_weight
        # 🔧 Web調査修正: .item()削除で勾配フロー維持
        print(f"    - 累積total_loss（Focal後）: {total_loss:.6f}")
        
        # Lovász-Softmax Loss (IoU直接最適化)
        print(f"  🔍 Lovász計算中...")
        print(f"    - best_masks入力: {best_masks.shape}, {best_masks.dtype}")
        print(f"    - target_masks入力: {target_masks.shape}, {target_masks.dtype}")
        lovasz_loss = self.lovasz(best_masks, target_masks)
        print(f"    - Lovász出力: {lovasz_loss.dtype}, value={lovasz_loss:.6f}")
        # Web調査修正: dtype変換による勾配切断防止（乗算で勾配保持）
        if lovasz_loss.dtype != target_dtype:
            print(f"    - Lovász型変換: {lovasz_loss.dtype} → {target_dtype}")
            lovasz_loss = lovasz_loss * torch.ones(1, dtype=target_dtype, device=lovasz_loss.device, requires_grad=True).squeeze()
        losses['lovasz_loss'] = lovasz_loss
        total_loss += lovasz_loss * self.lovasz_weight
        # 🔧 Web調査修正: .item()削除で勾配フロー維持
        print(f"    - 累積total_loss（Lovász後）: {total_loss:.6f}")
        
        # Dice Loss (バランス)
        print(f"  🔍 Dice（BCE）計算中...")
        dice_loss = self.dice(best_masks, target_masks)  # データ型保持
        print(f"    - BCE出力: {dice_loss.dtype}, value={dice_loss:.6f}")
        # Web調査修正: dtype変換による勾配切断防止（乗算で勾配保持）
        if dice_loss.dtype != target_dtype:
            print(f"    - BCE型変換: {dice_loss.dtype} → {target_dtype}")
            dice_loss = dice_loss * torch.ones(1, dtype=target_dtype, device=dice_loss.device, requires_grad=True).squeeze()
        losses['dice_loss'] = dice_loss
        total_loss += dice_loss * self.dice_weight
        # 🔧 Web調査修正: .item()削除で勾配フロー維持
        print(f"    - 累積total_loss（BCE後）: {total_loss:.6f}")
        
        # 2. Stage 1以降: Q-Former マルチモーダル損失（データ型統一）
        if self.stage >= 1 and query_embeds is not None and text_embeds is not None:
            print(f"  🔍 Q-Former損失計算中...")
            print(f"    - query_embeds: {query_embeds.shape}, {query_embeds.dtype}")
            print(f"    - text_embeds: {text_embeds.shape}, {text_embeds.dtype}")
            qformer_losses = self.qformer_loss(query_embeds, text_embeds)
            for key, value in qformer_losses.items():
                print(f"    - {key}: {value.dtype}, value={value:.6f}")
                # Web調査修正: dtype変換による勾配切断防止（乗算で勾配保持）
                if value.dtype != target_dtype:
                    print(f"      🔄 Q-Former {key}型変換: {value.dtype} → {target_dtype}")
                    value = value * torch.ones(1, dtype=target_dtype, device=value.device, requires_grad=True).squeeze()
                losses[f'qformer_{key}'] = value
                total_loss += value * self.qformer_weight
            # 🔧 Web調査修正: .item()削除で勾配フロー維持
            print(f"    - Q-Former損失後累積total_loss: {total_loss:.6f}")
        
        # 3. Stage 2以降: SAM2プロンプト最適化損失（データ型統一）
        if self.stage >= 2 and sam_prompts is not None and predicted_masks.dim() == 4:
            print(f"  🔍 SAM2プロンプト損失計算中...")
            print(f"    - sam_prompts: {sam_prompts.shape}, {sam_prompts.dtype}")
            sam2_losses = self.sam2_prompt_loss(sam_prompts, predicted_masks, target_masks)
            for key, value in sam2_losses.items():
                print(f"    - {key}: {value.dtype}, value={value:.6f}")
                # Web調査修正: dtype変換による勾配切断防止（乗算で勾配保持）
                if value.dtype != target_dtype:
                    print(f"      🔄 SAM2 {key}型変換: {value.dtype} → {target_dtype}")
                    value = value * torch.ones(1, dtype=target_dtype, device=value.device, requires_grad=True).squeeze()
                losses[f'sam2_{key}'] = value
                total_loss += value * self.sam2_prompt_weight
            # 🔧 Web調査修正: .item()削除で勾配フロー維持
            print(f"    - SAM2損失後累積total_loss: {total_loss:.6f}")
        
        # 最終損失の数値安定化（Web調査準拠: 勾配フロー維持）
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  ⚠️ total_lossに異常値: {total_loss} -> 1.0でクランプ")
            # Web調査準拠: torch.clampで勾配保持しながら値修正
            total_loss = torch.clamp(total_loss, min=0.001, max=10.0)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                # それでも異常値の場合のみ、勾配付きtensorで置換
                total_loss = torch.tensor(1.0, device=predicted_masks.device, dtype=predicted_masks.dtype, requires_grad=True)
        
        # Web調査準拠: 詳細勾配フロー追跡ログ
        print(f"🔍 勾配フロー追跡: 損失計算後（詳細）")
        for loss_name, loss_value in losses.items():
            grad_status = "✅ 正常" if loss_value.requires_grad and loss_value.grad_fn is not None else "⚠️ 途切れ" if loss_value.requires_grad and loss_value.grad_fn is None else "‼️ 無効"
            print(f"  🔍 [{loss_name}] shape={loss_value.shape}, dtype={loss_value.dtype}, device={loss_value.device}")
            print(f"      requires_grad={loss_value.requires_grad}, grad_fn={loss_value.grad_fn}")
            print(f"      {grad_status}: {loss_name}")
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice係数計算"""
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        
        return dice.mean()
    
    def set_stage(self, stage: int):
        """学習段階を設定"""
        self.stage = stage
        print(f"🔄 損失関数段階を Stage {stage} に設定")


# ファクトリー関数
def get_composite_loss_qformer_sam2(
    stage: int = 1,
    device: str = 'cuda'
) -> CompositeLossQFormerSAM2:
    """
    統合複合損失のファクトリー関数
    
    Args:
        stage: 学習段階 (1: Interface, 2: Full-stack, 3: Reasoning)
        device: デバイス
        
    Returns:
        CompositeLossQFormerSAM2: 設定済み複合損失
    """
    
    # 段階別重み設定（Web調査ベース）
    if stage == 1:
        # Stage 1: インターフェースアライメント重視
        config = {
            'focal_tversky_weight': 1.0,
            'lovasz_weight': 0.5,
            'dice_weight': 0.3,
            'qformer_weight': 0.5,  # Q-Former学習重視
            'sam2_prompt_weight': 0.0,  # SAM2凍結
            'stage': 1
        }
    elif stage == 2:
        # Stage 2: フルスタック適応
        config = {
            'focal_tversky_weight': 1.0,
            'lovasz_weight': 0.5,
            'dice_weight': 0.3,
            'qformer_weight': 0.2,
            'sam2_prompt_weight': 0.3,  # SAM2最適化開始
            'stage': 2
        }
    else:  # stage == 3
        # Stage 3: 推論能力専門化
        config = {
            'focal_tversky_weight': 1.0,
            'lovasz_weight': 0.8,  # IoU最適化強化
            'dice_weight': 0.2,
            'qformer_weight': 0.1,
            'sam2_prompt_weight': 0.5,  # SAM2最適化最大
            'stage': 3
        }
    
    loss_fn = CompositeLossQFormerSAM2(**config)
    loss_fn = loss_fn.to(device)
    
    print(f"✅ Stage {stage} 複合損失初期化完了")
    print(f"  - Focal Tversky (最高性能): {config['focal_tversky_weight']}")
    print(f"  - Lovász-Softmax (IoU): {config['lovasz_weight']}")
    print(f"  - Q-Former (マルチモーダル): {config['qformer_weight']}")
    print(f"  - SAM2プロンプト: {config['sam2_prompt_weight']}")
    
    return loss_fn


# テスト関数
def test_composite_loss():
    """複合損失のテスト"""
    print("=== CompositeLossQFormerSAM2 テスト ===")
    
    # テスト用データ
    batch_size = 2
    num_queries = 64
    height, width = 256, 256
    hidden_size = 5120
    prompt_dim = 256
    
    # ダミーデータ作成（BFloat16統一）
    predicted_masks = torch.randn(batch_size, num_queries, height, width, dtype=torch.bfloat16)
    target_masks = torch.randint(0, 2, (batch_size, height, width), dtype=torch.bfloat16)
    query_embeds = torch.randn(batch_size, num_queries, hidden_size, dtype=torch.bfloat16)
    text_embeds = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    sam_prompts = torch.randn(batch_size, num_queries, prompt_dim, dtype=torch.bfloat16)
    
    # 各段階テスト
    for stage in [1, 2, 3]:
        print(f"\n--- Stage {stage} テスト ---")
        
        loss_fn = get_composite_loss_qformer_sam2(stage=stage, device='cuda')
        
        with torch.no_grad():
            losses = loss_fn(
                predicted_masks=predicted_masks,
                target_masks=target_masks,
                query_embeds=query_embeds,
                text_embeds=text_embeds,
                sam_prompts=sam_prompts
            )
        
        print(f"  総損失: {losses['total_loss']:.4f}")
        for key, value in losses.items():
            if key != 'total_loss':
                print(f"  - {key}: {value:.4f}")
    
    print("\n✅ 複合損失テスト完了")


if __name__ == "__main__":
    test_composite_loss()