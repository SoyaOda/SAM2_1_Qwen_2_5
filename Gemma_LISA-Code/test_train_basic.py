 #!/usr/bin/env python3
"""
LISA-Gemma3 学習スクリプトの基本テスト
第4章の実装をテストする
"""

import os
import sys
import torch
from transformers import AutoProcessor

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_linux import *
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn
from functools import partial
from peft import LoraConfig, get_peft_model

def test_model_initialization():
    """モデルの初期化テスト"""
    print("=== モデル初期化テスト ===")
    
    try:
        # 設定作成
        config = LisaGemmaConfig(
            gemma_model_id=GEMMA_MODEL_ID,
            sam_checkpoint_path="",  # SAMなしでテスト
            seg_token="<SEG>",
            gemma_hidden_size=2560,
            sam_prompt_embed_dim=256,
        )
        
        # モデル初期化
        print("モデルを初期化中...")
        model = LisaGemmaForCausalLM(config)
        
        # 仕様書第4章.2に従ったLoRA設定
        print("LoRA設定を適用中...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 内部のGemmaモデルにのみLoRAアダプタを適用
        model.gemma_model = get_peft_model(model.gemma_model, lora_config)
        
        # パラメータの凍結設定（仕様書第2章.5）
        # Gemma本体は凍結（LoRAで間接的に学習）
        for param in model.gemma_model.base_model.model.parameters():
            param.requires_grad = False
        
        # MLPプロジェクタは訓練可能
        for param in model.mlp_projector.parameters():
            param.requires_grad = True
        
        # パラメータ数確認
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ モデル初期化成功")
        print(f"総パラメータ数: {total_params:,}")
        print(f"訓練可能パラメータ数: {trainable_params:,}")
        print(f"訓練可能な割合: {trainable_params/total_params*100:.2f}%")
        
        # 仕様書の期待値チェック
        expected_percentage = 1.0  # 1%未満
        if trainable_params/total_params*100 < expected_percentage:
            print(f"✅ 仕様書要件を満たしています（{expected_percentage}%未満）")
        else:
            print(f"⚠️ 訓練可能パラメータが多すぎます（{expected_percentage}%以上）")
        
        return model
        
    except Exception as e:
        print(f"❌ モデル初期化失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_loading():
    """データローディングテスト"""
    print("\n=== データローディングテスト ===")
    
    try:
        # Gemma-3プロセッサーの初期化
        print("Gemma-3プロセッサーを初期化中...")
        gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
        
        # データセット作成（小さなサンプル数でテスト）
        print("データセットを作成中...")
        dataset = HybridDataset(
            base_image_dir=DATASET_BASE_DIR,
            gemma_processor=gemma_processor,
            samples_per_epoch=10,  # 小さなサンプル数
            precision="bf16",
            gemma_image_size=GEMMA_IMAGE_SIZE,
            sam_image_size=SAM_IMAGE_SIZE,
        )
        
        print(f"✅ データセット作成成功: {len(dataset)} サンプル")
        
        # データローダー作成
        from torch.utils.data import DataLoader
        # collate_fnはgemma_processorを受け取らない単純な関数
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # テスト用
        )
        
        # 1バッチ取得してテスト
        print("1バッチを取得中...")
        batch = next(iter(dataloader))
        
        print("バッチ内容:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
            else:
                print(f"  {key}: {type(value)}")
        
        print("✅ データローディング成功")
        return dataloader, batch
        
    except Exception as e:
        print(f"❌ データローディング失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_forward_pass(model, batch):
    """フォワードパステスト"""
    print("\n=== フォワードパステスト ===")
    
    if model is None or batch is None:
        print("❌ モデルまたはバッチが無効です")
        return None
    
    try:
        # モデルを評価モードに設定
        model.eval()
        
        # バッチをGPUに移動（利用可能な場合）
        device = next(model.parameters()).device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        print("フォワードパスを実行中...")
        with torch.no_grad():
            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                labels=batch.get("input_ids"),  # テキスト損失計算のためラベルを設定
                generate_mask=False,  # SAMなしでテスト
            )
        
        print("出力内容:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        print("✅ フォワードパス成功")
        return outputs
        
    except Exception as e:
        print(f"❌ フォワードパス失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_loss_computation(outputs, batch):
    """損失計算テスト"""
    print("\n=== 損失計算テスト ===")
    
    if outputs is None or batch is None:
        print("❌ 出力またはバッチが無効です")
        return None
    
    try:
        # 損失関数初期化
        loss_fn = CompositeLoss(
            ce_loss_weight=1.0,
            dice_loss_weight=0.5,
            bce_loss_weight=2.0
        )
        
        # 損失計算用のバッチデータを準備
        # labelsは既にoutputsに含まれているtext_lossを使用
        batch_for_loss = {
            "labels": batch.get("input_ids"),  # テキストのラベル
            "ground_truth_mask": batch.get("masks")  # セグメンテーションマスク
        }
        
        print("損失を計算中...")
        losses = loss_fn(outputs, batch_for_loss)
        
        print("損失値:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("✅ 損失計算成功")
        return losses
        
    except Exception as e:
        print(f"❌ 損失計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メインテスト関数"""
    print("LISA-Gemma3 学習スクリプト基本テスト開始")
    print("=" * 60)
    
    # 1. モデル初期化テスト
    model = test_model_initialization()
    
    # 2. データローディングテスト
    dataloader, batch = test_data_loading()
    
    # 3. フォワードパステスト
    outputs = test_forward_pass(model, batch)
    
    # 4. 損失計算テスト
    losses = test_loss_computation(outputs, batch)
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー:")
    print(f"✅ モデル初期化: {'成功' if model is not None else '失敗'}")
    print(f"✅ データローディング: {'成功' if dataloader is not None else '失敗'}")
    print(f"✅ フォワードパス: {'成功' if outputs is not None else '失敗'}")
    print(f"✅ 損失計算: {'成功' if losses is not None else '失敗'}")
    
    if all([model, dataloader, outputs, losses]):
        print("\n🎉 全てのテストが成功しました！")
        print("学習スクリプトの基本的な動作が確認できました。")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
        print("エラーメッセージを確認して修正してください。")

if __name__ == "__main__":
    main()