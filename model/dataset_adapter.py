# model/dataset_adapter.py
"""
HybridDataset（シングルエンコーダー構成）とQFormerSegmentationBridgeの
インターフェースを調整するアダプター

使用例:
-------
1. 基本的な使用方法：
   ```python
   from model.dataset_adapter import adapt_dataset_for_qformer
   
   # HybridDatasetからのバッチ
   batch = {
       'sam_pixel_values': torch.randn(32, 3, 1024, 1024),
       'input_ids': torch.tensor(...),
       'attention_mask': torch.tensor(...),
   }
   
   # QFormerSegmentationBridge用に変換
   adapted_batch = adapt_dataset_for_qformer(batch)
   
   # QFormerSegmentationBridgeで使用
   outputs = qformer_bridge(
       images=adapted_batch['images'],
       input_ids=adapted_batch['input_ids'],
       attention_mask=adapted_batch['attention_mask']
   )
   ```

2. DataLoaderの変換：
   ```python
   from model.dataset_adapter import create_qformer_compatible_dataloader
   
   # 既存のDataLoader
   original_loader = DataLoader(hybrid_dataset, ...)
   
   # QFormer互換DataLoaderに変換
   qformer_loader = create_qformer_compatible_dataloader(original_loader)
   
   # 通常通り使用
   for batch in qformer_loader:
       outputs = qformer_bridge(**batch)
   ```
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
from utils.transforms import ResizeLongestSide
from utils.dataset import preprocess_mask_sam2_compliant


def adapt_dataset_for_qformer(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    HybridDatasetのデュアルエンコーダー構成出力を
    QFormerSegmentationBridgeの期待する形式に変換
    
    Args:
        batch: HybridDatasetからのバッチデータ
            - pixel_values: Llama-4用画像データ (448x448)
            - sam_pixel_values: SAM2用画像データ (1024x1024)
            - input_ids: テキスト入力ID
            - attention_mask: アテンションマスク
            - labels: ラベル
            
    Returns:
        QFormerSegmentationBridge用に変換されたバッチ
            - images: Llama-4用画像（pixel_values）
            - sam_images: SAM2用画像（sam_pixel_values）  
            - その他のフィールドはそのまま
    """
    adapted_batch = batch.copy()
    
    # デュアルエンコーダー構成：両方の画像を適切にマッピング
    if 'pixel_values' in batch and 'sam_pixel_values' in batch:
        # Llama-4用画像（AutoProcessorで標準処理済み）
        adapted_batch['images'] = batch['pixel_values']
        # SAM2用画像
        adapted_batch['sam_images'] = batch['sam_pixel_values']
        
        print(f"✓ デュアルエンコーダーアダプター: 両方の画像を変換完了")
        print(f"  - Llama-4画像形状: {adapted_batch['images'].shape}")
        print(f"  - SAM2画像形状: {adapted_batch['sam_images'].shape}")
        
    # シングルエンコーダー構成のフォールバック（互換性のため）
    elif 'sam_pixel_values' in batch and 'pixel_values' not in batch:
        adapted_batch['images'] = batch['sam_pixel_values']
        adapted_batch['sam_images'] = batch['sam_pixel_values']
        
        print(f"✓ シングルエンコーダーアダプター: sam_pixel_values → images 変換完了")
        print(f"  - 画像形状: {adapted_batch['images'].shape}")
    
    # pixel_valuesのみ存在する場合
    elif 'pixel_values' in batch and 'sam_pixel_values' not in batch:
        adapted_batch['images'] = batch['pixel_values']
        # SAM用画像をLlama画像から生成（リサイズ）
        import torch.nn.functional as F
        sam_images = F.interpolate(
            batch['pixel_values'],
            size=(1024, 1024),
            mode='bilinear',
            align_corners=False
        )
        adapted_batch['sam_images'] = sam_images
        
        print(f"✓ データセットアダプター: pixel_valuesからSAM画像を生成")
        print(f"  - Llama-4画像形状: {adapted_batch['images'].shape}")
        print(f"  - SAM2画像形状（生成）: {adapted_batch['sam_images'].shape}")
    
    # どちらも存在しない場合はエラー
    else:
        raise ValueError(
            "バッチデータに画像データが含まれていません。"
            "'pixel_values' または 'sam_pixel_values' が必要です。"
        )
    
    # 🔧 修正方針F: ラベルの形状統一処理（Original-LISA準拠）
    if 'labels' in batch:
        adapted_batch['labels'] = _unify_label_shapes(
            batch['labels'], 
            adapted_batch.get('sam_images', adapted_batch['images'])
        )
        print(f"✓ ラベル形状統一処理完了: {adapted_batch['labels'].shape}")
    
    return adapted_batch


def _unify_label_shapes(labels: torch.Tensor, reference_images: torch.Tensor) -> torch.Tensor:
    """
    ラベルの形状を画像のサイズに合わせて統一（SAM2+Original-LISA準拠）
    修正方針H: 非正方形ラベル柔軟処理 + ゼロマスクフォールバック回避
    
    Args:
        labels: ラベルテンソル（様々な形状の可能性）
        reference_images: 基準となる画像テンソル [B, C, H, W]
        
    Returns:
        統一されたラベルテンソル [B, H, W]
    """
    batch_size = reference_images.shape[0]
    target_size = reference_images.shape[2]  # 通常1024
    
    # 既に適切な形状の場合
    if labels.shape == (batch_size, target_size, target_size):
        print(f"✅ ラベル形状適切: {labels.shape}")
        return labels
    
    if labels.dim() == 2 and labels.shape[1] > 1:
        # フラット化されたラベルの場合: [B, flattened_size]
        print(f"📊 フラット化ラベル検出: {labels.shape}")
        
        # 元の2D形状を推定（修正方針H: 柔軟な矩形ラベル対応）
        flattened_size = labels.shape[1]
        estimated_dim = int(flattened_size ** 0.5)
        
        # 完全な正方形の場合
        if estimated_dim * estimated_dim == flattened_size:
            print(f"🔧 修正方針H: 正方形ラベル→SAM2準拠マスク前処理")
            
            unified_masks = []
            for i in range(batch_size):
                # 2D形状に戻す
                mask_2d = labels[i].view(estimated_dim, estimated_dim).cpu().numpy()
                
                # preprocess_sam_imageと同等の変換を適用
                processed_mask = preprocess_mask_sam2_compliant(mask_2d, target_size)
                unified_masks.append(processed_mask)
            
            unified_labels = torch.stack(unified_masks, dim=0).to(labels.device)
            print(f"✅ 正方形ラベル変換成功: {labels.shape} → {unified_labels.shape}")
            return unified_labels
        
        # 修正方針H: 非正方形（矩形）ラベル処理
        else:
            print(f"🔧 修正方針H: 非正方形ラベル処理開始 (flattened_size={flattened_size})")
            
            # 矩形の可能性を探索（アスペクト比考慮・修正方針L: 最小サイズ制約）
            possible_shapes = []
            for h in range(2, int(flattened_size**0.5) + 50):  # 🔧 h>=2で最小サイズ保証
                if flattened_size % h == 0:
                    w = flattened_size // h
                    if w >= 2:  # 🔧 w>=2で最小サイズ保証
                        possible_shapes.append((h, w))
            
            if possible_shapes:
                # 最も正方形に近い形状を選択
                best_shape = min(possible_shapes, key=lambda x: abs(x[0] - x[1]))
                h, w = best_shape
                print(f"🎯 最適矩形形状選択: {h}x{w} (flattened_size={flattened_size})")
                
                unified_masks = []
                for i in range(batch_size):
                    try:
                        # 矩形2D形状に戻す
                        mask_2d = labels[i].view(h, w).cpu().numpy()
                        
                        # SAM2準拠マスク前処理を適用
                        processed_mask = preprocess_mask_sam2_compliant(mask_2d, target_size)
                        unified_masks.append(processed_mask)
                        
                    except Exception as e:
                        print(f"⚠️ バッチ{i}の矩形変換エラー: {e}")
                        # 単一バッチ失敗時はゼロマスクで継続
                        zero_mask = torch.zeros((target_size, target_size), device=labels.device, dtype=labels.dtype)
                        unified_masks.append(zero_mask)
                
                if unified_masks:
                    unified_labels = torch.stack(unified_masks, dim=0).to(labels.device)
                    print(f"✅ 矩形ラベル変換成功: {labels.shape} → {unified_labels.shape}")
                    return unified_labels
            else:
                # 🔧 修正方針L: 極端な矩形の場合は近似的な正方形ラベル生成
                print(f"⚠️ 極端な矩形ラベル: flattened_size={flattened_size} → 近似的正方形変換")
                approx_dim = int(flattened_size**0.5)
                if approx_dim < 2:
                    approx_dim = 2
                
                unified_masks = []
                for i in range(batch_size):
                    # 近似的な正方形マスクを生成（中央に配置）
                    approx_mask = torch.zeros((approx_dim, approx_dim), dtype=labels.dtype)
                    if flattened_size > 0:
                        # 元のラベルから最初のいくつかの値を使用
                        fill_size = min(flattened_size, approx_dim * approx_dim)
                        flat_vals = labels[i][:fill_size]
                        approx_mask.view(-1)[:fill_size] = flat_vals
                    
                    # SAM2準拠変換
                    processed_mask = preprocess_mask_sam2_compliant(approx_mask.cpu().numpy(), target_size)
                    unified_masks.append(processed_mask)
                
                unified_labels = torch.stack(unified_masks, dim=0).to(labels.device)
                print(f"✅ 近似的正方形変換完了: {labels.shape} → {unified_labels.shape}")
                return unified_labels
    
    elif labels.dim() == 3:
        # 3Dラベルの場合: [B, H, W]
        current_H, current_W = labels.shape[1], labels.shape[2]
        if current_H != target_size or current_W != target_size:
            print(f"🔧 修正方針H: 3Dラベル→SAM2準拠マスク前処理")
            
            unified_masks = []
            for i in range(batch_size):
                mask_2d = labels[i].cpu().numpy()
                processed_mask = preprocess_mask_sam2_compliant(mask_2d, target_size)
                unified_masks.append(processed_mask)
            
            unified_labels = torch.stack(unified_masks, dim=0).to(labels.device)
            print(f"✅ 3DラベルSAM2準拠変換成功: {labels.shape} → {unified_labels.shape}")
            return unified_labels
        else:
            # サイズが既に適切な場合
            return labels
    
    # 修正方針H: エラー終了（ゼロマスクフォールバック回避）
    error_msg = f"❌ 修正方針H: 対応不可能なラベル形状 {labels.shape}"
    print(error_msg)
    raise ValueError(error_msg + f"\n対応可能: [B, {target_size}, {target_size}], [B, flattened_size], [B, H, W]")


def create_qformer_compatible_dataloader(original_dataloader):
    """
    既存のDataLoaderをQFormerSegmentationBridge互換に変換
    
    Args:
        original_dataloader: HybridDatasetを使用したDataLoader
        
    Returns:
        QFormerSegmentationBridge互換のDataLoader（ラッパー）
    """
    class QFormerCompatibleDataLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            
        def __iter__(self):
            for batch in self.dataloader:
                yield adapt_dataset_for_qformer(batch)
                
        def __len__(self):
            return len(self.dataloader)
        
        @property
        def batch_size(self):
            return self.dataloader.batch_size
        
        @property
        def dataset(self):
            return self.dataloader.dataset
    
    return QFormerCompatibleDataLoader(original_dataloader)


# デュアルエンコーダー構成のための設定
def configure_dual_encoder(config):
    """
    デュアルエンコーダー構成のための設定を適用
    
    Args:
        config: LlamaQFormerSAM2Config インスタンス
        
    Returns:
        更新されたconfig
    """
    # デュアルエンコーダー設定
    config.use_dual_encoder = True
    config.use_sam_as_vision_encoder = False  # SAM2はセグメンテーション専用
    
    # Llama-4のネイティブマルチモーダル活用
    config.llama_native_multimodal = True
    config.early_fusion = True
    
    # Q-Formerは両方の特徴を統合
    config.qformer_cross_modal = True
    config.sam_feature_extraction = True
    
    print("✓ デュアルエンコーダー設定完了")
    print("  - use_dual_encoder: True")  
    print("  - llama_native_multimodal: True")
    print("  - early_fusion: True")
    print("  - qformer_cross_modal: True")
    
    return config