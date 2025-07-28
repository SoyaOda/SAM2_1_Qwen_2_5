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
from typing import Dict, Any


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
    
    return adapted_batch


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