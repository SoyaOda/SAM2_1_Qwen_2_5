"""
SAM2.1 + Qwen2.5-VL Integration Test Script
新しい統合モデルのテスト用スクリプト
"""
import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
from typing import Dict, List, Tuple
import time
from PIL import Image
import warnings

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 設定とモデルのインポート
from config_qwen_sam import get_config, validate_config
from model.sam_qwen_model import SAMQwenModel, create_sam_qwen_model


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


def create_test_image_and_mask(size=(1024, 1024)) -> Tuple[np.ndarray, np.ndarray]:
    """テスト用の画像とマスクを作成"""
    # シンプルなテスト画像を作成（グラデーション + 円形オブジェクト）
    H, W = size
    image = np.zeros((H, W, 3), dtype=np.uint8)
    
    # グラデーション背景
    for i in range(H):
        for j in range(W):
            image[i, j] = [int(255 * i / H), int(255 * j / W), 128]
    
    # 円形オブジェクトを追加
    center_y, center_x = H // 2, W // 2
    radius = min(H, W) // 6
    
    y, x = np.ogrid[:H, :W]
    circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
    # 円形領域を明るくする
    image[circle_mask] = [255, 255, 255]
    
    # Ground truth マスク
    gt_mask = circle_mask.astype(np.uint8)
    
    return image, gt_mask


def test_model_initialization():
    """モデル初期化テスト"""
    print("\n🚀 SAM-Qwen統合モデル初期化テスト...\n")
    
    # 設定を取得・検証
    config = get_config('test')  # テスト用設定を使用
    validate_config(config)
    print("✅ 設定検証完了")
    
    # GPU確認
    if not torch.cuda.is_available():
        print("⚠️ GPU未検出、CPUで実行します")
        config.TORCH_DTYPE = torch.float32
        config.DEVICE_MAP = "cpu"
    else:
        print(f"✅ GPU利用可能: {torch.cuda.get_device_name()}")
    
    try:
        # モデル初期化
        print("\n🔧 統合モデル初期化中...")
        model_config = config.get_model_config()
        model = create_sam_qwen_model(model_config)
        
        print("✅ モデル初期化成功！")
        print(f"  - デバイス: {model.device}")
        print(f"  - Qwenモデル: {model.model_name}")
        print(f"  - SAMモデル: {model.sam_model_name}")
        print(f"  - 視覚プロジェクター: {'Q-Former' if model.use_qformer else 'Linear'}")
        print(f"  - クエリ数: {model.num_queries}")
        
        return model, config
        
    except Exception as e:
        print(f"❌ モデル初期化失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_visual_projector(model, config):
    """視覚プロジェクターのテスト"""
    print("\n🔍 視覚プロジェクターテスト...")
    
    try:
        # テスト用の視覚特徴
        batch_size = 1
        num_patches = 64  # 8x8のパッチグリッド
        image_dim = 256  # SAM2.1のデフォルト次元
        
        visual_features = torch.randn(batch_size, num_patches, image_dim).to(model.device)
        
        # プロジェクター実行
        with torch.no_grad():
            projected_features = model.visual_projector(visual_features)
        
        print("✅ 視覚プロジェクター動作確認")
        print(f"  - 入力形状: {visual_features.shape}")
        print(f"  - 出力形状: {projected_features.shape}")
        print(f"  - 期待クエリ数: {model.num_queries}")
        print(f"  - 期待次元: {model.llm.config.hidden_size}")
        
        # 形状確認
        expected_shape = (batch_size, model.num_queries, model.llm.config.hidden_size)
        if projected_features.shape == expected_shape:
            print("✅ 出力形状正常")
        else:
            print(f"⚠️ 出力形状不一致: 期待 {expected_shape}, 実際 {projected_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 視覚プロジェクターテスト失敗: {str(e)}")
        return False


def test_segmentation_only(model, config):
    """セグメンテーションのみのテスト"""
    print("\n🎯 セグメンテーション機能テスト...")
    
    try:
        # テスト画像とマスクを作成
        test_image, gt_mask = create_test_image_and_mask((512, 512))  # 小さめのサイズでテスト
        
        # 点プロンプトを作成（円の中心）
        center_point = torch.tensor([[256.0, 256.0]])  # 画像中心
        point_labels = torch.tensor([1])  # 前景ラベル
        
        print(f"  - テスト画像サイズ: {test_image.shape}")
        print(f"  - GTマスクサイズ: {gt_mask.shape}")
        print(f"  - 点プロンプト: {center_point.numpy()}")
        
        # セグメンテーション実行
        start_time = time.time()
        
        with torch.no_grad():
            results = model.forward(
                image=test_image,
                question="",  # セグメンテーションのみなので空
                point_coords=center_point,
                point_labels=point_labels,
                return_text_only=False,
                return_mask_only=True  # マスクのみ取得
            )
        
        seg_time = time.time() - start_time
        
        print("✅ セグメンテーション成功！")
        print(f"  - 処理時間: {seg_time:.3f}秒")
        print(f"  - 予測マスク形状: {results['mask'].shape}")
        print(f"  - IoUスコア数: {len(results['iou_scores'])}")
        print(f"  - 最良マスクIndex: {results['best_mask_idx']}")
        print(f"  - 最良IoUスコア: {results['iou_scores'][results['best_mask_idx']]:.4f}")
        
        # 評価メトリクス計算
        pred_mask = results['mask'].cpu().numpy().astype(bool)
        iou = calculate_iou(torch.from_numpy(pred_mask), torch.from_numpy(gt_mask.astype(bool)))
        dice = calculate_dice(torch.from_numpy(pred_mask), torch.from_numpy(gt_mask.astype(bool)))
        
        print(f"\n📊 セグメンテーション評価:")
        print(f"  - IoU: {iou:.4f}")
        print(f"  - Dice: {dice:.4f}")
        
        return True, {'iou': iou, 'dice': dice, 'time': seg_time}
        
    except Exception as e:
        print(f"❌ セグメンテーションテスト失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_text_generation_only(model, config):
    """テキスト生成のみのテスト"""
    print("\n💬 テキスト生成機能テスト...")
    
    try:
        # テスト画像を作成
        test_image, _ = create_test_image_and_mask((512, 512))
        test_question = "この画像に何が写っていますか？"
        
        print(f"  - 質問: {test_question}")
        
        # テキスト生成実行
        start_time = time.time()
        
        with torch.no_grad():
            results = model.forward(
                image=test_image,
                question=test_question,
                return_text_only=True,  # テキストのみ取得
                return_mask_only=False,
                max_new_tokens=64
            )
        
        gen_time = time.time() - start_time
        
        print("✅ テキスト生成成功！")
        print(f"  - 処理時間: {gen_time:.3f}秒")
        print(f"  - 生成テキスト: '{results['generated_text']}'")
        print(f"  - 入力長: {results['input_length']}")
        print(f"  - 生成ID形状: {results['generated_ids'].shape}")
        
        return True, {'text': results['generated_text'], 'time': gen_time}
        
    except Exception as e:
        print(f"❌ テキスト生成テスト失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_unified_inference(model, config):
    """統合推論テスト（セグメンテーション + テキスト生成）"""
    print("\n🔗 統合推論テスト...")
    
    try:
        # テスト画像とマスク作成
        test_image, gt_mask = create_test_image_and_mask((512, 512))
        test_question = "画像の中心にある白い円について説明してください。"
        
        # 点プロンプト（円の中心）
        center_point = torch.tensor([[256.0, 256.0]])
        point_labels = torch.tensor([1])
        
        print(f"  - 質問: {test_question}")
        print(f"  - 点プロンプト: {center_point.numpy()}")
        
        # 統合推論実行
        start_time = time.time()
        
        with torch.no_grad():
            results = model.forward(
                image=test_image,
                question=test_question,
                point_coords=center_point,
                point_labels=point_labels,
                return_text_only=False,
                return_mask_only=False,  # 両方取得
                max_new_tokens=128
            )
        
        unified_time = time.time() - start_time
        
        print("✅ 統合推論成功！")
        print(f"  - 処理時間: {unified_time:.3f}秒")
        print(f"  - 生成テキスト: '{results['generated_text']}'")
        print(f"  - マスク形状: {results['mask'].shape}")
        print(f"  - 視覚特徴形状: {results['visual_features'].shape}")
        
        # セグメンテーション評価
        pred_mask = results['mask'].cpu().numpy().astype(bool)
        iou = calculate_iou(torch.from_numpy(pred_mask), torch.from_numpy(gt_mask.astype(bool)))
        dice = calculate_dice(torch.from_numpy(pred_mask), torch.from_numpy(gt_mask.astype(bool)))
        
        print(f"\n📊 統合推論評価:")
        print(f"  - セグメンテーションIoU: {iou:.4f}")
        print(f"  - セグメンテーションDice: {dice:.4f}")
        print(f"  - テキスト長: {len(results['generated_text'])}文字")
        
        return True, {
            'iou': iou,
            'dice': dice,
            'text': results['generated_text'],
            'time': unified_time
        }
        
    except Exception as e:
        print(f"❌ 統合推論テスト失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    """メインテスト実行"""
    print("🧪 SAM2.1 + Qwen2.5-VL 統合モデルテスト開始")
    print("=" * 60)
    
    # 警告を抑制
    warnings.filterwarnings("ignore", category=UserWarning)
    
    results = {}
    
    # 1. モデル初期化テスト
    model, config = test_model_initialization()
    if model is None:
        print("❌ モデル初期化に失敗したため、テストを中止します")
        return
    
    results['initialization'] = True
    
    # 2. 視覚プロジェクターテスト
    projector_success = test_visual_projector(model, config)
    results['visual_projector'] = projector_success
    
    # 3. セグメンテーション機能テスト
    seg_success, seg_metrics = test_segmentation_only(model, config)
    results['segmentation'] = seg_success
    results['seg_metrics'] = seg_metrics
    
    # 4. テキスト生成機能テスト
    text_success, text_metrics = test_text_generation_only(model, config)
    results['text_generation'] = text_success
    results['text_metrics'] = text_metrics
    
    # 5. 統合推論テスト
    unified_success, unified_metrics = test_unified_inference(model, config)
    results['unified_inference'] = unified_success
    results['unified_metrics'] = unified_metrics
    
    # 最終サマリー
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")
    print("=" * 60)
    
    success_count = sum([
        results['initialization'],
        results['visual_projector'],
        results['segmentation'],
        results['text_generation'],
        results['unified_inference']
    ])
    
    print(f"✅ 成功したテスト: {success_count}/5")
    print(f"{'✅' if results['initialization'] else '❌'} モデル初期化")
    print(f"{'✅' if results['visual_projector'] else '❌'} 視覚プロジェクター")
    print(f"{'✅' if results['segmentation'] else '❌'} セグメンテーション機能")
    print(f"{'✅' if results['text_generation'] else '❌'} テキスト生成機能")
    print(f"{'✅' if results['unified_inference'] else '❌'} 統合推論")
    
    if success_count == 5:
        print("\n🎉 全テスト成功！SAM2.1 + Qwen2.5-VL統合モデルが正常に動作しています。")
    else:
        print(f"\n⚠️ {5-success_count}個のテストが失敗しました。ログを確認してください。")
    
    print("=" * 60)


if __name__ == "__main__":
    main()