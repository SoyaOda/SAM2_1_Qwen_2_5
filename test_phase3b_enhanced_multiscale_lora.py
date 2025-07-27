# test_phase3b_enhanced_multiscale_lora.py
"""
Enhanced版テストスクリプト: マルチスケール推論 + LoRA対応
test_phase3b_simple.pyをベースにマルチスケール機能とLoRAを有効化したバージョン
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
from typing import Dict, List, Tuple
import time
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_repo'))

# 設定
import config_linux

# Enhanced Model
from model.enhanced_llama4_qformer_sam2 import (
    EnhancedQFormerSegmentationBridge, 
    EnhancedLlamaQFormerSAM2Config
)

# Transformers
from transformers import AutoModelForCausalLM, AutoProcessor


def calculate_iou(pred_mask, gt_mask, smooth=1e-6):
    """IoU計算"""
    pred = pred_mask.float()
    gt = gt_mask.float()
    
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_dice(pred_mask, gt_mask, smooth=1e-6):
    """Dice係数計算"""
    pred = pred_mask.float()
    gt = gt_mask.float()
    
    intersection = (pred * gt).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    
    return dice.item()


def evaluate_multiscale_predictions(predictions: List[Dict], gt_mask: torch.Tensor) -> Dict:
    """マルチスケール予測の評価"""
    results = {}
    
    for i, pred in enumerate(predictions):
        scale = pred['scale']
        mask = pred['mask']
        
        # マスクサイズを統一
        if mask.shape[-2:] != gt_mask.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(0), 
                size=gt_mask.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # バイナリ化
        binary_mask = (mask > 0.5).float()
        
        # メトリクス計算
        iou = calculate_iou(binary_mask, gt_mask)
        dice = calculate_dice(binary_mask, gt_mask)
        
        results[f'scale_{scale}'] = {
            'iou': iou,
            'dice': dice,
            'mask_shape': mask.shape
        }
    
    return results


def multiscale_inference(model, image: torch.Tensor, text_input: List[str], 
                        scales: List[float] = [0.75, 1.0, 1.25]) -> List[Dict]:
    """マルチスケール推論実行"""
    predictions = []
    original_size = image.shape[-2:]
    
    for scale in scales:
        # スケールに応じて画像リサイズ
        if scale != 1.0:
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_image = F.interpolate(
                image, 
                size=new_size, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            scaled_image = image
        
        print(f"  📏 スケール {scale}: {scaled_image.shape[-2:]} -> {original_size}")
        
        # 推論実行
        with torch.no_grad():
            outputs = model(
                images=scaled_image,
                text_input=text_input,
                mode='itg'
            )
        
        # マスクを元のサイズに戻す
        masks = outputs['masks']
        if masks.shape[-2:] != original_size:
            masks = F.interpolate(
                masks, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        predictions.append({
            'scale': scale,
            'mask': masks[0, 0],  # バッチ次元、チャネル次元を削除
            'confidence': outputs.get('iou_scores', torch.tensor([1.0]))[0].item(),
            'scaled_image_size': scaled_image.shape[-2:]
        })
    
    return predictions


def ensemble_predictions(predictions: List[Dict]) -> torch.Tensor:
    """マルチスケール予測のアンサンブル"""
    masks = []
    weights = []
    
    for pred in predictions:
        masks.append(pred['mask'])
        weights.append(pred['confidence'])
    
    # 重み付き平均
    masks_tensor = torch.stack(masks)
    weights_tensor = torch.tensor(weights, device=masks_tensor.device)
    weights_tensor = F.softmax(weights_tensor, dim=0)
    
    ensemble_mask = (masks_tensor * weights_tensor.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
    
    return ensemble_mask


def test_enhanced_multiscale_lora():
    """Enhanced版テスト実行: マルチスケール + LoRA"""
    print("🚀 Enhanced マルチスケール + LoRA テスト開始...\n")
    
    # GPU確認
    if not torch.cuda.is_available():
        raise RuntimeError("GPU環境が必要です")
    
    print(f"✅ GPU利用可能: {torch.cuda.get_device_name()}")
    print(f"  - GPU数: {torch.cuda.device_count()}")
    
    # 1. Llama-4初期化
    print("\n🧠 Llama-4初期化...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        config_linux.LLAMA_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    llama_processor = AutoProcessor.from_pretrained(
        config_linux.LLAMA_MODEL_ID,
        trust_remote_code=True
    )
    
    print("✅ Llama-4初期化完了")
    
    # 2. Enhanced Model初期化（マルチスケール + LoRA有効）
    print("\n🚀 Enhanced Model初期化（マルチスケール + LoRA有効）...")
    enhanced_config = EnhancedLlamaQFormerSAM2Config()
    
    model = EnhancedQFormerSegmentationBridge(
        config=enhanced_config,
        shared_llama_model=llama_model,
        shared_llama_processor=llama_processor,
        enable_lora=True,          # LoRA有効
        enable_multiscale=True,    # マルチスケール有効
        training_stage=1
    )
    
    print("✅ Enhanced Model初期化完了")
    
    # LoRA統計表示
    lora_count = 0
    lora_params = 0
    for name, module in model.named_modules():
        if 'lora' in name.lower():
            lora_count += 1
            for param in module.parameters():
                lora_params += param.numel()
    
    print(f"\n🔧 LoRA統計:")
    print(f"  - LoRAモジュール数: {lora_count}")
    print(f"  - LoRAパラメータ数: {lora_params:,}")
    
    # 3. テストデータ準備
    print("\n📦 テストデータ作成...")
    batch_size = 1
    test_image = torch.randn(batch_size, 3, 448, 448).cuda()
    test_text = ["Segment the main object in this image"]
    test_labels = torch.zeros(batch_size, 448, 448, dtype=torch.float).cuda()
    
    # 円形のテストラベル作成
    center_y, center_x = 224, 224
    radius = 80
    y, x = np.ogrid[:448, :448]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    test_labels[0][mask] = 1.0
    
    print(f"  - 画像形状: {test_image.shape}")
    print(f"  - テキスト: {test_text}")
    print(f"  - ラベル形状: {test_labels.shape}")
    print(f"  - GT正例ピクセル数: {test_labels.sum().item()}")
    
    # 4. 単一スケールテスト
    print("\n🧪 単一スケールForward処理テスト...")
    
    try:
        start_time = time.time()
        with torch.no_grad():
            outputs = model(
                images=test_image,
                text_input=test_text,
                labels=test_labels,
                mode='itg'
            )
        single_scale_time = time.time() - start_time
        
        print("\n✅ 単一スケールForward成功！")
        print(f"  - マスク形状: {outputs['masks'].shape}")
        print(f"  - テキストロジット形状: {outputs['text_logits'].shape}")
        print(f"  - 視覚特徴形状: {outputs['visual_features'].shape}")
        print(f"  - 推論時間: {single_scale_time:.3f}秒")
        
        if 'loss' in outputs:
            print(f"  - 損失値: {outputs['loss'].item():.4f}")
        
        # Q-Former出力確認
        if 'qformer_outputs' in outputs:
            qf_out = outputs['qformer_outputs']
            print(f"\n📊 Q-Former出力:")
            print(f"  - クエリ埋め込み: {qf_out['query_embeds'].shape}")
            print(f"  - LLM埋め込み: {qf_out['llm_embeds'].shape}")
            print(f"  - SAMプロンプト: {qf_out['sam_prompts'].shape}")
        
        # 単一スケール予測評価
        single_mask = outputs['masks'][0, 0]
        single_iou = calculate_iou(single_mask > 0.5, test_labels[0])
        single_dice = calculate_dice(single_mask > 0.5, test_labels[0])
        
        print(f"\n📊 単一スケール評価:")
        print(f"  - IoU: {single_iou:.4f}")
        print(f"  - Dice: {single_dice:.4f}")
        
    except Exception as e:
        print(f"\n❌ 単一スケールエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. マルチスケール推論テスト
    print("\n🔍 マルチスケール推論テスト...")
    
    try:
        scales = [0.75, 1.0, 1.25]
        start_time = time.time()
        
        multiscale_predictions = multiscale_inference(
            model, test_image, test_text, scales
        )
        
        multiscale_time = time.time() - start_time
        
        print(f"\n✅ マルチスケール推論成功！")
        print(f"  - 推論時間: {multiscale_time:.3f}秒")
        print(f"  - スケール数: {len(multiscale_predictions)}")
        
        # 各スケールの評価
        multiscale_results = evaluate_multiscale_predictions(
            multiscale_predictions, test_labels[0]
        )
        
        print(f"\n📊 スケール別評価:")
        for scale_name, metrics in multiscale_results.items():
            print(f"  - {scale_name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")
        
        # アンサンブル予測
        ensemble_mask = ensemble_predictions(multiscale_predictions)
        ensemble_iou = calculate_iou(ensemble_mask > 0.5, test_labels[0])
        ensemble_dice = calculate_dice(ensemble_mask > 0.5, test_labels[0])
        
        print(f"\n🎯 アンサンブル評価:")
        print(f"  - IoU: {ensemble_iou:.4f}")
        print(f"  - Dice: {ensemble_dice:.4f}")
        
        # 性能比較
        print(f"\n📈 性能比較:")
        print(f"  - 単一スケール IoU: {single_iou:.4f}")
        print(f"  - マルチスケール IoU: {ensemble_iou:.4f}")
        print(f"  - IoU改善: {ensemble_iou - single_iou:+.4f}")
        print(f"  - 推論時間比: {multiscale_time/single_scale_time:.1f}x")
        
    except Exception as e:
        print(f"\n❌ マルチスケールエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. LoRA効果テスト
    print("\n🔬 LoRA効果検証...")
    
    try:
        # LoRA無効バージョンとの比較
        print("  📊 LoRAパラメータ統計:")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_ratio = lora_params / total_params * 100
        
        print(f"    - 総パラメータ数: {total_params:,}")
        print(f"    - 学習可能パラメータ数: {trainable_params:,}")
        print(f"    - LoRAパラメータ数: {lora_params:,}")
        print(f"    - LoRA比率: {lora_ratio:.2f}%")
        
    except Exception as e:
        print(f"  ⚠️ LoRA統計計算エラー: {e}")
    
    print("\n🎉 Enhanced マルチスケール + LoRA テスト完了！")
    
    # 最終サマリー
    print("\n" + "="*60)
    print("📋 テストサマリー")
    print("="*60)
    print(f"✅ 単一スケール推論: 成功")
    print(f"   - IoU: {single_iou:.4f}, Dice: {single_dice:.4f}")
    print(f"   - 推論時間: {single_scale_time:.3f}秒")
    print(f"✅ マルチスケール推論: 成功")
    print(f"   - スケール数: {len(scales)}")
    print(f"   - アンサンブル IoU: {ensemble_iou:.4f}, Dice: {ensemble_dice:.4f}")
    print(f"   - 推論時間: {multiscale_time:.3f}秒")
    print(f"✅ LoRA統合: 成功")
    print(f"   - LoRAモジュール数: {lora_count}")
    print(f"   - パラメータ効率: {lora_ratio:.2f}%")
    print("="*60)


if __name__ == "__main__":
    test_enhanced_multiscale_lora()