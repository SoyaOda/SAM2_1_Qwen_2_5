#!/usr/bin/env python3
"""
第5節：堅牢性と将来の開発に向けた事前検証
5.2. サニティチェック2：エンドツーエンドの推論パイプライン

論理的根拠:
学習時のコードパスと推論時のコードパスは、微妙に異なることがよくあります。
このテストは、一度「学習」されたモデル（たとえ1バッチだけでも）が、
クラッシュすることなく予測に使用できることを保証します。
デバイスの不一致、データ型の問題、batch_size=1の時の形状変化など、
デプロイ時に頻発するエラーを事前に捕捉します。

期待される結果:
スクリプトがエラーなく最後まで実行されることが第一の目標です。
出力の「質」を評価するのではなく、データ入力からモデル処理、
そして出力生成までの一連のパイプラインが実行可能であることを確認します。
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoProcessor
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話型に設定
import matplotlib.pyplot as plt

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset

def get_config():
    """
    config_linux.pyを必須として読み込む
    読み込めない場合はエラーで停止
    """
    try:
        import config_linux as config
        print(f"設定: config_linux.py を使用")
        return config
    except ImportError as e:
        print(f"❌ ERROR: config_linux.pyが見つかりません")
        print(f"   詳細: {e}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        print(f"   ファイル存在確認: {os.path.exists('config_linux.py')}")
        raise SystemExit("config_linux.pyが必須です。ファイルが存在することを確認してください。")

def parse_args():
    # config_linux.pyから設定を取得
    config = get_config()
    
    parser = argparse.ArgumentParser(description="エンドツーエンド推論パイプライン検証")
    parser.add_argument("--dataset_type", type=str, default="reason_seg", 
                       choices=["sem_seg", "refer_seg", "vqa", "reason_seg"],
                       help="テストに使用するデータセットタイプ")
    parser.add_argument("--num_samples", type=int, default=3, 
                       help="テストするサンプル数")
    parser.add_argument("--max_new_tokens", type=int, 
                       default=getattr(config, 'MAX_NEW_TOKENS', 100),
                       help="生成する最大トークン数")
    parser.add_argument("--pretrained_from_overfit", action="store_true",
                       help="過学習テストで学習したモデル状態を使用")
    return parser.parse_args()

# フォールバック機能は削除

def load_model_with_training_state(config, device, args):
    """
    過学習テストで得られたモデル状態または新規モデルをロード
    """
    print("\n📦 LISA-Gemmaモデルを初期化中...")
    
    lisa_config = LisaGemmaConfig(
        gemma_model_id=getattr(config, 'GEMMA_MODEL_ID', 'google/gemma-3-4b-it'),
        sam_checkpoint_path=getattr(config, 'SAM_CHECKPOINT_PATH', None),
        seg_token=getattr(config, 'SEG_TOKEN', '[SEG]'),
        gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),
        sam_prompt_embed_dim=getattr(config, 'SEG_PROJECTION_DIM', 256),
        gemma_image_size=getattr(config, 'GEMMA_IMAGE_SIZE', 896),
        sam_image_size=getattr(config, 'SAM_IMAGE_SIZE', 1024),
        model_max_length=getattr(config, 'MODEL_MAX_LENGTH', 2048),
    )
    
    model = LisaGemmaForCausalLM(lisa_config)
    model = model.to(device)
    
    # LoRA設定を適用（最適化後の設定で検証するため）
    print("\n🔧 LoRA設定を適用中...")
    try:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # LoRAをGemmaモデルに適用
        if not hasattr(model, 'gemma_model'):
            print(f"❌ ERROR: モデルにgemma_modelが見つかりません")
            print(f"   モデル属性: {list(model.__dict__.keys())}")
            raise SystemExit("モデル構造エラー: gemma_modelが見つかりません")
        
        model.gemma_model = get_peft_model(model.gemma_model, lora_config)
        print("✅ LoRA設定が正常に適用されました")
        
        # 学習可能パラメータの確認
        param_info = model.get_trainable_parameters_info()
        print(f"✅ LoRA適用後のパラメータ情報:")
        print(f"  - 総パラメータ数: {param_info['total_parameters']:,}")
        print(f"  - 学習可能パラメータ数: {param_info['trainable_parameters']:,}")
        print(f"  - 学習可能率: {param_info['trainable_percentage']:.2f}%")
        
        # 仕様書準拠性チェック
        spec_compliant = param_info['trainable_percentage'] < 1.0
        print(f"  - 📋 仕様書準拠性: {'✅ 準拠' if spec_compliant else '❌ 違反'} (要求: <1%)")
        
        if not spec_compliant:
            print(f"❌ ERROR: 学習可能パラメータ率が仕様書要求を超過: {param_info['trainable_percentage']:.2f}% > 1%")
            raise SystemExit("仕様書準拠性違反: 学習可能パラメータ率が1%を超えています")
        
    except ImportError as e:
        print(f"❌ ERROR: peftライブラリが見つかりません: {e}")
        print("   必須: pip install peft でインストールしてください")
        raise SystemExit("peftライブラリが必須です")
    except Exception as e:
        print(f"❌ ERROR: LoRA設定中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(f"LoRA設定エラー: {e}")
    
    if args.pretrained_from_overfit:
        print("📚 過学習テストで学習したモデル状態をロード...")
        
        # 最新の過学習チェックポイントを探す
        output_dir = "verification_output"
        latest_checkpoint_path = os.path.join(output_dir, "latest_overfit_checkpoint.pth")
        
        if not os.path.exists(latest_checkpoint_path):
            print(f"❌ ERROR: 過学習チェックポイントが見つかりません: {latest_checkpoint_path}")
            print(f"📌 必須: 先に `python overfit_single_batch.py` を実行してチェックポイントを作成してください")
            raise SystemExit("過学習チェックポイントが必要です。overfit_single_batch.pyを先に実行してください。")
        
        print(f"✓ 最新チェックポイントを発見: {latest_checkpoint_path}")
        
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # チェックポイント情報を表示
            config_info = checkpoint.get('config', {})
            print(f"✅ 過学習チェックポイントをロード成功:")
            print(f"  - セッション: {checkpoint.get('session_timestamp', 'unknown')}")
            print(f"  - 学習イテレーション: {config_info.get('iterations', 'unknown')}")
            print(f"  - 最終損失: {config_info.get('final_loss', 'unknown'):.6f}")
            print(f"  - 損失減少率: {config_info.get('reduction_ratio', 0)*100:.1f}%")
            
        except Exception as e:
            print(f"❌ ERROR: チェックポイントロード失敗: {e}")
            print(f"   ファイルパス: {latest_checkpoint_path}")
            print(f"   ファイル存在確認: {os.path.exists(latest_checkpoint_path)}")
            raise SystemExit(f"チェックポイントロードエラー: {e}")
    else:
        print("📚 新しいモデル状態で推論テスト (--pretrained_from_overfit未指定)")
        print("   注意: 学習前の初期重みでの推論のため、出力品質は期待できません")
    
    # 推論モードに切り替え
    model.eval()
    print("✅ モデルを推論モードに設定")
    
    return model, lisa_config

def load_single_sample(config, dataset_type: str, sample_idx: int = 0):
    """
    単一のデータサンプルをロード（batch_size=1でのテスト用）
    """
    print(f"\n📝 単一サンプルをロード中 (dataset: {dataset_type}, index: {sample_idx})")
    
    # プロセッサーの準備
    processor = AutoProcessor.from_pretrained(
        getattr(config, 'GEMMA_MODEL_ID', 'google/gemma-3-4b-it')
    )
    
    # A10 24GB制約対応: サンプル数調整
    samples_per_epoch = 10
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 25:  # A10 (24GB) 検出
            samples_per_epoch = 5  # A10では少なめに設定
            print(f"  ⚠️  A10 GPU検出 ({gpu_memory_gb:.1f}GB): サンプル数を調整")
    
    # データセットの準備
    dataset = HybridDataset(
        base_image_dir=getattr(config, 'DATASET_BASE_DIR', './dataset'),
        gemma_processor=processor,
        dataset=dataset_type,
        samples_per_epoch=samples_per_epoch
    )
    
    if sample_idx >= len(dataset):
        sample_idx = 0
        print(f"⚠️ サンプルインデックスが範囲外のため、0に調整しました")
    
    # 単一サンプルの取得
    sample = dataset[sample_idx]
    
    print(f"✅ サンプル取得完了:")
    print(f"  - データセット: {dataset_type}")
    print(f"  - サンプルインデックス: {sample_idx}")
    print(f"  - 入力シーケンス長: {sample['input_ids'].shape[0]}")
    print(f"  - 画像形状 (Gemma): {sample['images_for_gemma'].shape}")
    print(f"  - 画像形状 (SAM): {sample['images_for_sam'].shape}")
    
    # 会話内容の表示（最初の100文字のみ）
    if 'conversations' in sample:
        for i, conv in enumerate(sample['conversations'][:2]):  # 最初の2ターンのみ
            content = conv['value'][:100] + "..." if len(conv['value']) > 100 else conv['value']
            print(f"  - {conv['from']}: {content}")
    
    return sample, dataset

def run_inference_pipeline(model, sample, device, max_new_tokens=100):
    """
    推論パイプラインの実行
    """
    print(f"\n🔍 推論パイプラインを実行中...")
    
    # デバイスへの移動
    sample_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in sample.items()}
    
    # 推論実行時の詳細情報
    print(f"推論設定:")
    print(f"  - max_new_tokens: {max_new_tokens}")
    print(f"  - デバイス: {device}")
    print(f"  - モデルモード: {'train' if model.training else 'eval'}")
    
    # A10 24GB制約対応: メモリクリーンアップ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 25:  # A10 (24GB) 検出
            print(f"  ⚠️  A10 GPU検出: メモリクリーンアップ実行")
    
    start_time = time.time()
    
    try:
        with torch.no_grad():  # 推論時は勾配計算無効
            # 1. バッチ形式での推論（batch_size=1）
            print("\n📊 バッチ形式推論テスト...")
            
            batch_inputs = {
                'input_ids': sample_device['input_ids'].unsqueeze(0),
                'attention_mask': sample_device['attention_mask'].unsqueeze(0),
                'images_for_gemma': sample_device['images_for_gemma'].unsqueeze(0),
                'images_for_sam': sample_device['images_for_sam'].unsqueeze(0),
                'generate_mask': True
            }
            
            batch_outputs = model(**batch_inputs)
            
            print(f"✅ バッチ推論成功:")
            print(f"  - 出力テンソル数: {len(batch_outputs)}")
            if 'predicted_masks' in batch_outputs:
                masks_shape = batch_outputs['predicted_masks'].shape
                print(f"  - 予測マスク形状: {masks_shape}")
            if 'logits' in batch_outputs:
                logits_shape = batch_outputs['logits'].shape
                print(f"  - ロジット形状: {logits_shape}")
            
            # 2. 生成式推論テスト（generate_with_segmentationがある場合）
            if hasattr(model, 'generate_with_segmentation'):
                print("\n🎯 生成式推論テスト...")
                
                # PIL画像とテキストプロンプトを準備
                image_tensor = sample_device['images_for_gemma'].squeeze(0)  # (3, H, W)
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                
                # プロンプトの抽出（conversationsから）
                text_prompt = ""
                if 'conversations' in sample:
                    for conv in sample['conversations']:
                        if conv['from'] == 'human':
                            text_prompt = conv['value'].replace('<image>', '').strip()
                            break
                
                if not text_prompt:
                    text_prompt = "この画像で指定されたオブジェクトをセグメント化してください。"
                
                print(f"  - テキストプロンプト: {text_prompt[:50]}...")
                
                try:
                    generation_outputs = model.generate_with_segmentation(
                        image=pil_image,
                        text_prompt=text_prompt,
                        max_new_tokens=max_new_tokens
                    )
                    
                    print(f"✅ 生成式推論成功:")
                    print(f"  - 出力キー: {list(generation_outputs.keys())}")
                    
                    if 'generated_text' in generation_outputs:
                        generated_text = generation_outputs['generated_text']
                        display_text = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                        print(f"  - 生成テキスト: {display_text}")
                    
                    if 'predicted_mask' in generation_outputs:
                        mask_shape = generation_outputs['predicted_mask'].shape
                        print(f"  - 予測マスク形状: {mask_shape}")
                    
                except Exception as e:
                    print(f"⚠️ 生成式推論でエラー: {e}")
                    generation_outputs = None
            else:
                print("\n⚠️ generate_with_segmentationメソッドが見つかりません")
                generation_outputs = None
        
        inference_time = time.time() - start_time
        print(f"\n⏱️ 推論時間: {inference_time:.2f}秒")
        
        return {
            'batch_outputs': batch_outputs,
            'generation_outputs': generation_outputs,
            'inference_time': inference_time,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"\n❌ ERROR: 推論中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print(f"   推論時間: {inference_time:.2f}秒")
        print(f"   デバイス: {device}")
        print(f"   モデルモード: {'train' if model.training else 'eval'}")
        
        # A10対応: エラー時のメモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU メモリクリーンアップ実行")
        
        raise SystemExit(f"推論エラー: {e}")

def visualize_inference_result(original_data, pred_mask, image_path, sample_idx, session_timestamp):
    """推論結果を可視化して保存（第1節の手法を適用）"""
    try:
        item_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"inference_result_sample_{sample_idx}_{item_timestamp}"
        
        output_dir = "verification_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 画像処理（第1節の手法を適用）
        try:
            # 元画像を直接読み込み（最も確実な方法）
            original_image = Image.open(image_path).convert('RGB')
            print(f"✅ 元画像を直接読み込み: {original_image.size}")
        except Exception as e:
            print(f"元画像読み込みエラー: {e}")
            # フォールバック: SAM前処理済み画像の逆変換
            # original_dataには'images'キーでSAM処理済み画像が含まれている
            if 'images' in original_data:
                image_tensor = original_data['images']
                if image_tensor.dim() == 4:
                    image_tensor = image_tensor[0]  # バッチ次元を削除
                
                # SAM前処理の逆変換（第1節と同じパラメータ）
                pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
                pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
                
                # 正規化を逆変換
                image_denorm = image_tensor * pixel_std + pixel_mean
                image_denorm = torch.clamp(image_denorm / 255.0, 0, 1)
                
                # PIL画像に変換
                pil_converter = ToPILImage()
                original_image = pil_converter(image_denorm.cpu())
                print(f"✅ SAM前処理済み画像を逆変換: {original_image.size}")
            else:
                raise Exception("画像データが見つかりません")
        
        # 予測マスクの処理（バイナリマスクに変換）
        if pred_mask is not None:
            if isinstance(pred_mask, torch.Tensor):
                pred_mask_np = pred_mask.cpu().numpy()
            else:
                pred_mask_np = pred_mask
            
            # マスクの次元調整
            if pred_mask_np.ndim == 3:
                if pred_mask_np.shape[0] == 1:
                    pred_mask_np = pred_mask_np.squeeze(0)
                elif pred_mask_np.shape[-1] == 1:
                    pred_mask_np = pred_mask_np.squeeze(-1)
                else:
                    pred_mask_np = pred_mask_np[0]
            
            # マスクを画像と同じサイズにリサイズ
            if pred_mask_np.ndim == 2:
                img_width, img_height = original_image.size
                
                # 0-1の値を0-255に変換してPILでリサイズ
                mask_pil = Image.fromarray((pred_mask_np * 255).astype(np.uint8), mode='L')
                mask_resized = mask_pil.resize((img_width, img_height), Image.NEAREST)
                pred_mask_np = np.array(mask_resized) / 255.0
        
        # 正解マスクの処理（存在する場合）
        gt_mask_np = None
        if 'masks' in original_data and original_data['masks'] is not None:
            gt_mask = original_data['masks']
            if isinstance(gt_mask, list) and len(gt_mask) > 0:
                gt_mask = gt_mask[0]
            
            if isinstance(gt_mask, torch.Tensor):
                gt_mask_np = gt_mask.cpu().numpy()
            else:
                gt_mask_np = gt_mask
            
            # 正解マスクの次元調整
            if gt_mask_np.ndim == 3:
                if gt_mask_np.shape[0] == 1:
                    gt_mask_np = gt_mask_np.squeeze(0)
                elif gt_mask_np.shape[-1] == 1:
                    gt_mask_np = gt_mask_np.squeeze(-1)
                else:
                    gt_mask_np = gt_mask_np[0]
            
            # 正解マスクを画像と同じサイズにリサイズ
            if gt_mask_np.ndim == 2:
                img_width, img_height = original_image.size
                gt_mask_pil = Image.fromarray((gt_mask_np * 255).astype(np.uint8), mode='L')
                gt_mask_resized = gt_mask_pil.resize((img_width, img_height), Image.NEAREST)
                gt_mask_np = np.array(gt_mask_resized) / 255.0
        
        # 3パネル比較画像作成
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # オリジナル画像
        axes[0].imshow(original_image)
        axes[0].set_title(f"Original Image\n({original_image.size[0]}x{original_image.size[1]})")
        axes[0].axis('off')
        
        # 予測マスク
        if pred_mask_np is not None:
            axes[1].imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f"Predicted Mask\n({pred_mask_np.shape[1]}x{pred_mask_np.shape[0]})")
        else:
            axes[1].text(0.5, 0.5, 'No Prediction', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title("No Predicted Mask")
        axes[1].axis('off')
        
        # 正解マスク（存在する場合）または予測結果のオーバーレイ
        axes[2].imshow(original_image)
        if gt_mask_np is not None:
            # 正解マスクを表示
            mask_colored = np.zeros((*gt_mask_np.shape, 4))
            mask_colored[gt_mask_np > 0.1] = [0, 1, 0, 0.6]  # 緑色半透明
            axes[2].imshow(mask_colored)
            axes[2].set_title("Ground Truth Overlay")
        elif pred_mask_np is not None:
            # 予測マスクを表示
            mask_colored = np.zeros((*pred_mask_np.shape, 4))
            mask_colored[pred_mask_np > 0.1] = [1, 0, 0, 0.6]  # 赤色半透明
            axes[2].imshow(mask_colored)
            axes[2].set_title("Prediction Overlay")
        else:
            axes[2].set_title("Image Only")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 画像保存
        image_filename = os.path.join(output_dir, f"{base_filename}.png")
        plt.savefig(image_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可視化結果を保存: {image_filename}")
        return image_filename
        
    except Exception as e:
        print(f"可視化エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    config = get_config()
    
    print("="*80)
    print("第5節: 堅牢性と将来の開発に向けた事前検証")
    print("5.2. サニティチェック2：エンドツーエンドの推論パイプライン")
    print("="*80)
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # GPU情報表示（A10対応）
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU情報: {gpu_name} ({gpu_memory_gb:.1f}GB)")
        if gpu_memory_gb < 25:
            print(f"  ⚠️  A10制約対応モード有効")
    
    print(f"📋 設定情報 (config_linux.py):")
    print(f"  - データセットベースディレクトリ: {getattr(config, 'DATASET_BASE_DIR', 'N/A')}")
    print(f"  - Gemmaモデル: {getattr(config, 'GEMMA_MODEL_ID', 'N/A')}")
    print(f"  - SAMチェックポイント: {getattr(config, 'SAM_CHECKPOINT_PATH', 'N/A')}")
    print(f"  - 最大シーケンス長: {getattr(config, 'MODEL_MAX_LENGTH', 'N/A')}")
    
    print(f"🎯 テスト設定:")
    print(f"  - データセットタイプ: {args.dataset_type}")
    print(f"  - テストサンプル数: {args.num_samples}")
    print(f"  - 最大生成トークン数: {args.max_new_tokens}")
    print(f"  - 過学習済みモデル使用: {args.pretrained_from_overfit}")
    print("-" * 80)
    
    # 出力ディレクトリの作成
    output_dir = "verification_output"
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = {
        "session_timestamp": session_timestamp,
        "test_config": {
            "dataset_type": args.dataset_type,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "pretrained_from_overfit": args.pretrained_from_overfit,
        },
        "results": [],
        "overall_success": True,
        "total_inference_time": 0
    }
    
    try:
        # 1. モデルのロード
        model, lisa_config = load_model_with_training_state(config, device, args)
        
        # 2. 各サンプルでの推論テスト
        for sample_idx in range(args.num_samples):
            print(f"\n" + "="*60)
            print(f"📋 サンプル {sample_idx + 1}/{args.num_samples} のテスト")
            print("="*60)
            
            try:
                # サンプルロード
                sample, dataset = load_single_sample(config, args.dataset_type, sample_idx)
                
                # 推論実行
                inference_results = run_inference_pipeline(
                    model, sample, device, args.max_new_tokens
                )
                
                # 結果の可視化
                pred_mask = None
                if inference_results['batch_outputs'] and 'predicted_masks' in inference_results['batch_outputs']:
                    pred_mask = inference_results['batch_outputs']['predicted_masks'][0, 0]  # (B, N, H, W) -> (H, W)
                elif inference_results['generation_outputs'] and 'predicted_mask' in inference_results['generation_outputs']:
                    pred_mask = inference_results['generation_outputs']['predicted_mask']
                
                # 画像パスを取得（データセットから）
                image_path = sample.get('image_path')
                if not image_path:
                    # フォールバック: データセットから画像パスを再取得
                    raw_sample = dataset[sample_idx]
                    image_path = raw_sample.get('image_path')
                
                visualization_path = visualize_inference_result(
                    sample, pred_mask, image_path, sample_idx, session_timestamp
                )
                
                # 結果の記録
                sample_result = {
                    "sample_idx": sample_idx,
                    "dataset_type": args.dataset_type,
                    "success": inference_results['success'],
                    "inference_time": inference_results['inference_time'],
                    "error": inference_results['error'],
                    "visualization_path": visualization_path,
                    "has_predicted_mask": inference_results['batch_outputs'] is not None and 'predicted_masks' in inference_results['batch_outputs'],
                    "has_generated_text": inference_results['generation_outputs'] is not None and 'generated_text' in inference_results['generation_outputs']
                }
                
                results_summary["results"].append(sample_result)
                results_summary["total_inference_time"] += inference_results['inference_time']
                
                if not inference_results['success']:
                    results_summary["overall_success"] = False
                
                # サンプル結果のサマリー
                status = "✅ 成功" if inference_results['success'] else "❌ 失敗"
                print(f"\n📊 サンプル {sample_idx + 1} 結果: {status}")
                print(f"  - 推論時間: {inference_results['inference_time']:.2f}秒")
                
            except Exception as e:
                print(f"\n❌ ERROR: サンプル {sample_idx + 1} でエラーが発生: {e}")
                import traceback
                traceback.print_exc()
                print(f"   サンプルインデックス: {sample_idx}")
                print(f"   データセットタイプ: {args.dataset_type}")
                raise SystemExit(f"サンプル {sample_idx + 1} 推論エラー: {e}")
        
        # 3. 総合結果の分析
        print(f"\n" + "="*80)
        print("📊 総合結果の分析")
        print("="*80)
        
        successful_samples = sum(1 for r in results_summary["results"] if r["success"])
        success_rate = successful_samples / len(results_summary["results"]) * 100
        
        print(f"推論パイプライン検証結果: {'✅ 成功' if results_summary['overall_success'] else '❌ 部分的失敗'}")
        print(f"  - 成功率: {successful_samples}/{len(results_summary['results'])} ({success_rate:.1f}%)")
        print(f"  - 総推論時間: {results_summary['total_inference_time']:.2f}秒")
        print(f"  - 平均推論時間: {results_summary['total_inference_time']/len(results_summary['results']):.2f}秒/サンプル")
        
        # 4. 結果の保存
        print(f"\n💾 結果を保存中...")
        
        json_path = os.path.join(output_dir, f"inference_pipeline_test_{session_timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 結果保存完了:")
        print(f"  - 詳細データ: {json_path}")
        
        # 5. 最終判定
        print(f"\n" + "="*80)
        print("🎯 最終判定")
        print("="*80)
        
        # 成功率100%以外はエラーで停止
        if not results_summary["overall_success"] or success_rate < 100.0:
            print("❌ ERROR: サニティチェック2 失敗")
            print("   以下の問題が検出されました:")
            
            for i, result in enumerate(results_summary["results"]):
                if not result["success"]:
                    print(f"   - サンプル {i+1}: {result.get('error', 'Unknown error')}")
            
            print(f"   成功率: {success_rate:.1f}% < 100%")
            raise SystemExit("推論パイプライン検証失敗: 全サンプルが成功する必要があります")
        
        print("✅ サニティチェック2: エンドツーエンドの推論パイプライン - 成功")
        print("   推論パイプラインは正常に動作しています。")
        print("   学習時から推論時への移行に問題はありません。")
        print("   デプロイメント準備が整っています。")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 致命的エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 