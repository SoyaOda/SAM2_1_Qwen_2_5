#!/usr/bin/env python3
"""
LISA-Llama4 Dataset Integrity Verification (Single-Encoder Configuration)
Lambda Cloud最適化版 - 軽量化とライブラリ遅延読み込み

HybridDatasetがシングルエンコーダー構成で正しく動作し、
意図した通りの学習サンプルを生成しているかを検証するスクリプト。

主な検証内容:
- sam_pixel_valuesが正しく生成されているか
- pixel_values（Llama4用）が生成されていないか
- Llama-4ネイティブフォーマットでテキストが処理されているか
- <|image|>トークンが自動挿入されているか
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# 軽量なライブラリのみ先に読み込み
import json

# 重いライブラリは遅延読み込み（必要時のみ）
# import torch  # 遅延読み込み
# import numpy as np  # 遅延読み込み
# import matplotlib.pyplot as plt  # 遅延読み込み
# from PIL import Image  # 遅延読み込み
# from torchvision.transforms import ToPILImage  # 遅延読み込み

print("🚀 LISA-Llama4 Dataset Integrity Verification (Single-Encoder Configuration)")
print("📦 基本ライブラリ読み込み完了")

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_config():
    """
    config_linux.pyを必須として読み込む
    読み込めない場合はエラーで停止
    """
    try:
        import config_linux as config
        print(f"✅ 設定ファイルを読み込み: config_linux.py")
        return config
    except ImportError as e:
        print(f"❌ ERROR: config_linux.pyが見つかりません")
        print(f"   詳細: {e}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        print(f"   ファイル存在確認: {os.path.exists('config_linux.py')}")
        raise SystemExit("config_linux.pyが必須です。ファイルが存在することを確認してください。")

def parse_args():
    parser = argparse.ArgumentParser(description="Verify HybridDataset Integrity")
    parser.add_argument(
        "--datasets", 
        nargs='+', 
        default=["sem_seg", "refer_seg", "vqa", "reason_seg"],
        help="List of sub-dataset names to inspect"
    )
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to inspect per dataset")
    return parser.parse_args()

def save_comparison_image(image_tensor, mask_tensor, dataset_name, sample_idx, 
                         image_path, conversations, questions, class_names, session_timestamp,
                         formatted_prompt=None, is_single_encoder=True):
    """比較画像とメタデータを保存（画面表示なし）- シングルエンコーダー構成対応"""
    try:
        item_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ミリ秒まで
        base_filename = f"{dataset_name}_{sample_idx}_{item_timestamp}"
        
        # セッション用のサブディレクトリ作成
        output_dir = os.path.join("verification_output", f"session_{session_timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 画像処理（元画像から直接読み込み）
        try:
            # 元画像を直接読み込み（最も確実な方法）
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"元画像読み込みエラー: {e}")
            # フォールバック: SAM前処理済み画像の逆変換
            pil_converter = ToPILImage()
            
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]
            
            # SAM前処理の逆変換（正規化とパディングを元に戻す）
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
            
            # 正規化を逆変換
            image_denorm = image_tensor * pixel_std + pixel_mean
            image_denorm = torch.clamp(image_denorm / 255.0, 0, 1)
            
            # PIL画像に変換（1024x1024のまま）
            image = pil_converter(image_denorm.cpu())
        
        # マスク処理 - 画像と同じサイズにリサイズ
        mask = None
        if mask_tensor is not None:
            if isinstance(mask_tensor, list) and len(mask_tensor) > 0:
                mask_tensor = mask_tensor[0]
            
            if isinstance(mask_tensor, torch.Tensor):
                mask = mask_tensor.cpu().numpy()
            
            if mask is not None and mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                else:
                    mask = mask[0]
            
            # マスクを画像と同じサイズにリサイズ（シンプルな方法）
            if mask is not None and mask.ndim == 2:
                # 画像のサイズを取得
                img_width, img_height = image.size
                
                # マスクをPILでリサイズ
                mask_pil = Image.fromarray((mask * 255).astype('uint8'), mode='L')
                mask_resized = mask_pil.resize((img_width, img_height), Image.NEAREST)
                import numpy as np
                mask = np.array(mask_resized) / 255.0
        
        # 比較画像作成（画面表示なし）
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # SAM処理済み画像
        axes[0].imshow(image)
        axes[0].set_title(f"{dataset_name} - SAM Processed Image\n({image.size[0]}x{image.size[1]})")
        axes[0].axis('off')
        
        # マスクのみ
        if mask is not None and mask.ndim == 2:
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f"{dataset_name} - Resized Mask\n({mask.shape[1]}x{mask.shape[0]})")
        else:
            axes[1].text(0.5, 0.5, 'No Mask\n(VQA Task)', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title(f"{dataset_name} - No Mask")
        axes[1].axis('off')
        
        # マスク付き画像
        axes[2].imshow(image)
        if mask is not None and mask.ndim == 2:
            # マスクと画像のサイズが一致することを確認
            if mask.shape[:2] == (image.size[1], image.size[0]):
                import numpy as np
                mask_colored = np.zeros((*mask.shape, 4))
                mask_colored[mask > 0.1] = [1, 0, 0, 0.6]  # 赤色半透明
                axes[2].imshow(mask_colored)
                axes[2].set_title(f"{dataset_name} - Synchronized Overlay")
            else:
                axes[2].set_title(f"{dataset_name} - Size Mismatch: img{image.size} vs mask{mask.shape}")
        else:
            axes[2].set_title(f"{dataset_name} - Image Only")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 画像保存（画面表示なし）
        image_filename = os.path.join(output_dir, f"{base_filename}_comparison.png")
        plt.savefig(image_filename, dpi=150, bbox_inches='tight')
        plt.close()  # メモリリークを防ぐためにfigureを閉じる
        
        # メタデータ保存
        metadata = {
            "dataset_name": dataset_name,
            "sample_idx": sample_idx,
            "session_timestamp": session_timestamp,
            "item_timestamp": item_timestamp,
            "original_image_path": image_path,
            "conversations": conversations,
            "questions": questions,
            "class_names": class_names,
            "comparison_image": image_filename,
            "processing_note": "Single-encoder configuration: SAM processes images",
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "mask_size": f"{mask.shape[1]}x{mask.shape[0]}" if mask is not None else "None",
            "formatted_prompt": formatted_prompt,
            "is_single_encoder": is_single_encoder
        }
        
        metadata_filename = os.path.join(output_dir, f"{base_filename}_metadata.json")
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return image_filename, metadata_filename
        
    except Exception as e:
        print(f"保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_heavy_libraries():
    """重いライブラリを必要時に読み込む"""
    print("📦 重いライブラリを読み込み中...")
    global torch, np, plt, Image, ToPILImage
    
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # バックエンドを非対話型に設定
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms import ToPILImage
    
    print("✅ PyTorch, NumPy, Matplotlib読み込み完了")

def main():
    args = parse_args()
    config = get_config()
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n🔍 LISA-Llama4 Dataset Integrity Verification (Single-Encoder Configuration)")
    print("=" * 60)
    print(f"🚀 検証セッション開始: {session_timestamp}")
    print(f"📁 出力ディレクトリ: verification_output/session_{session_timestamp}/")
    print(f"🔧 シングルエンコーダー構成: SAM専用画像処理")
    print("=" * 60)
    
    # 重いライブラリを必要時に読み込み
    load_heavy_libraries()
    
    # HybridDatasetを使用した検証（シングルエンコーダー構成）
    try:
        print("📦 HybridDatasetクラスを読み込み中...")
        from utils.dataset import HybridDataset, setup_seg_token
        from transformers import AutoProcessor
        
        print("✅ データセットクラス読み込み完了")
        
        # Llama4 Processorを読み込み（トークナイザーとして使用）
        print(f"📦 Llama4 Processor初期化中...")
        processor = AutoProcessor.from_pretrained(config.LLAMA_MODEL_ID)
        tokenizer = processor.tokenizer
        
        # [SEG]トークンを追加
        seg_token_idx = setup_seg_token(tokenizer, config.SEG_TOKEN)
        print(f"✅ Processorとトークナイザーの準備完了")
        
        # データセット設定（config_linux.pyから統一管理）
        print(f"📋 設定情報:")
        print(f"   データセットベースディレクトリ: {config.DATASET_BASE_DIR}")
        print(f"   SAMチェックポイント: {config.SAM_CHECKPOINT_PATH}")
        print(f"   バッチサイズ: {config.BATCH_SIZE_PER_GPU}")
        print(f"   勾配蓄積ステップ: {config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Llama4モデル: {config.LLAMA_MODEL_ID}")
        print(f"   SEGトークンID: {seg_token_idx}")
        
        # HybridDatasetを初期化
        print(f"📦 HybridDataset初期化中...")
        
        # デバッグ: 引数を表示
        print("  デバッグ: HybridDataset初期化引数:")
        print(f"    base_image_dir: {config.DATASET_BASE_DIR}")
        print(f"    llama_processor: {type(processor)}")
        print(f"    samples_per_epoch: {getattr(config, 'SAMPLES_PER_EPOCH', 500)}")
        print(f"    llama_image_size: {config.LLAMA_IMAGE_SIZE}")
        print(f"    sam_image_size: {config.SAM_IMAGE_SIZE}")
        
        try:
            hybrid_dataset = HybridDataset(
                base_image_dir=config.DATASET_BASE_DIR,
                llama_processor=processor,
                samples_per_epoch=getattr(config, 'SAMPLES_PER_EPOCH', 500),
                precision="bf16",
                llama_image_size=config.LLAMA_IMAGE_SIZE,
                sam_image_size=config.SAM_IMAGE_SIZE,
                num_classes_per_sample=3,
                exclude_val=False,
                dataset="sem_seg||refer_seg||vqa||reason_seg",
                sample_rate=[9, 3, 3, 1],
                sem_seg_data=config.SEM_SEG_DATA,
                refer_seg_data=config.REFER_SEG_DATA,
                vqa_data=config.VQA_DATA,
                reason_seg_data=config.REASON_SEG_DATA,
                explanatory=0.1  # 追加: 必須引数
            )
            print(f"✅ HybridDataset初期化完了 (サンプル数: {len(hybrid_dataset)})")
        except Exception as e:
            print(f"❌ HybridDataset初期化エラー: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # HybridDatasetを検証
        print("\n🔬 HybridDataset検証開始（シングルエンコーダー構成）")
        
        # サンプル検証
        for i in range(min(args.num_samples, len(hybrid_dataset))):
            print(f"\n--- HybridDataset Sample {i+1}/{args.num_samples} ---")
            
            try:
                # HybridDatasetからサンプル取得
                sample = hybrid_dataset[i]
                dataset_name = sample.get('dataset_name', 'unknown')
                
                print(f"📂 ソースデータセット: {dataset_name}")
                
                # データ構造確認
                print(f"📦 返却キー: {list(sample.keys())}")
                
                # シングルエンコーダー構成の確認
                if 'sam_pixel_values' in sample:
                    print(f"✅ SAM画像入力: shape={sample['sam_pixel_values'].shape}")
                else:
                    print(f"❌ SAM画像入力が存在しません")
                
                if 'pixel_values' in sample:
                    print(f"⚠️  pixel_valuesが存在します（シングルエンコーダーでは不要）")
                
                # テキスト入力の確認
                if 'input_ids' in sample:
                    print(f"📝 入力ID: shape={sample['input_ids'].shape}")
                    
                    # Llama-4ネイティブフォーマットの確認
                    if 'formatted_prompt' in sample:
                        prompt_preview = sample['formatted_prompt'][:200] + "..." if len(sample['formatted_prompt']) > 200 else sample['formatted_prompt']
                        print(f"💬 フォーマット済みプロンプト: {prompt_preview}")
                        
                        # <|image|>トークンの確認
                        if '<|image|>' in sample['formatted_prompt']:
                            print(f"✅ Llama-4ネイティブ<|image|>トークン検出")
                        else:
                            print(f"⚠️ <|image|>トークンが見つかりません")
                
                # マスクデータの確認
                if 'ground_truth_mask' in sample and sample['ground_truth_mask'] is not None:
                    mask = sample['ground_truth_mask']
                    print(f"🎯 マスク: shape={mask.shape}")
                    
                    # 元画像サイズの確認
                    if 'original_size' in sample:
                        orig_h, orig_w = sample['original_size']
                        print(f"📷 元画像サイズ: {orig_w}x{orig_h}")
                else:
                    print(f"ℹ️  マスクなし（VQAタスクなど）")
                
                # SEGトークンマスクの確認
                if 'seg_token_mask' in sample:
                    seg_positions = torch.nonzero(sample['seg_token_mask']).squeeze()
                    if seg_positions.numel() > 0:
                        print(f"🎯 [SEG]トークン位置: {seg_positions.tolist()}")
                    else:
                        print(f"ℹ️  [SEG]トークンなし")
                
                # 可視化
                if 'sam_pixel_values' in sample:
                    image_path = sample.get('image_path', '')
                    conversations = []
                    if 'text_prompt' in sample:
                        conversations = [{"from": "human", "value": sample['text_prompt']}]
                    
                    save_comparison_image(
                        sample['sam_pixel_values'],
                        sample.get('ground_truth_mask'),
                        dataset_name,
                        i,
                        image_path=image_path,
                        conversations=conversations,
                        questions=sample.get('questions'),
                        class_names=sample.get('sampled_classes'),
                        session_timestamp=session_timestamp,
                        formatted_prompt=sample.get('formatted_prompt'),
                        is_single_encoder=True
                    )
                    print(f"📸 比較画像を保存しました")
                
                print(f"  ✅ サンプル {i+1} 検証完了")
                
            except Exception as e:
                print(f"  ❌ サンプル {i+1} 検証失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # セッション情報のサマリー作成
        session_dir = os.path.join("verification_output", f"session_{session_timestamp}")
        session_summary = {
            "session_info": {
                "session_timestamp": session_timestamp,
                "start_time": session_timestamp,
                "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "datasets_verified": args.datasets,
                "samples_per_dataset": args.num_samples,
                "config_used": "config_linux.py",
                "single_encoder_mode": True
            },
            "verification_results": "See individual metadata files for detailed results"
        }
        
        # セッションサマリーを保存
        summary_file = os.path.join(session_dir, "session_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(session_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 検証完了: データセット構築と完全性の検証（シングルエンコーダー構成）")
        print(f"📁 保存された比較画像とメタデータ: verification_output/session_{session_timestamp}/")
        print(f"📋 セッションサマリー: {summary_file}")
        print(f"🕐 セッション終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 検証失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()