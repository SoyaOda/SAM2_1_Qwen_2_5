#!/usr/bin/env python3
"""
簡単なDDPテストスクリプト
train_simple_test.pyの成功をベースにDDP機能をテスト
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# プロジェクトパス
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config_linux as config
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model

def test_single_gpu_with_ddp_interface():
    """DDP風のインターフェースで単一GPU学習をテスト"""
    print("🔧 DDP風インターフェースでの単一GPU学習テスト")
    print("=" * 50)
    
    try:
        # 1. Gemmaプロセッサーの初期化
        gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
        
        # SEGトークンの追加
        seg_token = "[SEG]"
        if seg_token not in gemma_processor.tokenizer.get_vocab():
            gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        
        print(f"✅ Gemmaプロセッサー初期化完了")
        print(f"   SEGトークンID: {gemma_processor.tokenizer.convert_tokens_to_ids(seg_token)}")
        
        # 2. モデル設定
        model_config = LisaGemmaConfig(
            gemma_model_id=config.GEMMA_MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            seg_token_idx=gemma_processor.tokenizer.convert_tokens_to_ids(seg_token),
            gemma_hidden_size=config.GEMMA_HIDDEN_SIZE,
            sam_prompt_embed_dim=config.SEG_PROJECTION_DIM,
        )
        
        # 3. モデル初期化
        model = LisaGemmaForCausalLM(model_config)
        model.resize_token_embeddings(len(gemma_processor.tokenizer))
        
        # 4. LoRA適用
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        print(f"✅ モデル初期化・LoRA適用完了")
        
        # 5. パラメータ確認
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   総パラメータ: {total_params:,}")
        print(f"   訓練可能: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # 6. デバイス設定（DDP風）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ モデルを {device} に配置")
        
        # 7. データセット（小さなサイズ）
        dataset = HybridDataset(
            base_image_dir=config.DATASET_BASE_DIR,
            gemma_processor=gemma_processor,
            samples_per_epoch=5,  # 非常に小さく
            precision="bf16",
            gemma_image_size=config.GEMMA_IMAGE_SIZE,
            sam_image_size=config.SAM_IMAGE_SIZE,
            dataset="reason_seg",
            sample_rate=[1.0],
            reason_seg_data="ReasonSeg|train",
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # 最小バッチサイズ
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        print(f"✅ データセット・ローダー作成完了: {len(dataset)} サンプル")
        
        # 8. オプティマイザ（DDP風設定）
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(config.BETA1, config.BETA2),
            weight_decay=config.WEIGHT_DECAY,
        )
        print(f"✅ オプティマイザ設定完了")
        
        # 9. 損失関数
        loss_fn = CompositeLoss(
            ce_loss_weight=config.CE_LOSS_WEIGHT,
            dice_loss_weight=config.DICE_LOSS_WEIGHT,
            bce_loss_weight=config.BCE_LOSS_WEIGHT
        )
        print(f"✅ 損失関数設定完了")
        
        # 10. DDP風学習ループ（1エポック）
        print("\n🔥 DDP風学習開始...")
        model.train()
        
        epoch_loss = 0
        num_steps = 0
        
        for step, batch in enumerate(dataloader):
            if step >= 3:  # 3ステップのみ
                break
                
            # バッチをデバイスに移動
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # フォワードパス
            outputs = model(**batch)
            
            # 損失計算
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict['total_loss']
            
            # バックワードパス
            loss.backward()
            
            # オプティマイザステップ
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_steps += 1
            
            print(f"   Step {step+1}: Loss = {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_steps
        print(f"\n✅ DDP風学習完了!")
        print(f"   平均損失: {avg_loss:.4f}")
        print(f"   実行ステップ数: {num_steps}")
        
        # 11. 成功判定
        if avg_loss > 0 and avg_loss < 10:  # 合理的な損失範囲
            print("\n🎉 DDP準備成功!")
            print("   → train_ddp.py の実装が可能です")
            print("   → DeepSpeedへの移行準備が整いました")
            return True
        else:
            print(f"\n⚠️  損失値が異常: {avg_loss}")
            return False
            
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_next_steps():
    """次のステップを提案"""
    print("\n" + "=" * 50)
    print("📋 次のステップ提案")
    print("=" * 50)
    
    print("1. 🎯 今すぐ実行可能:")
    print("   python train_ddp.py --batch_size 4 --world_size 1")
    print("   → DeepSpeedと同じインターフェースで分散学習")
    
    print("\n2. 🔧 環境を改善したい場合:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   → CUDA 12.1版で互換性向上")
    
    print("\n3. 🌐 将来のクラウド移行:")
    print("   deepspeed train_deepspeed.py --deepspeed_config ds_config_cloud.json")
    print("   → 設定ファイルとコードが既に準備済み")
    
    print("\n4. 📊 実装優先順位:")
    print("   Phase 1: DDP学習の安定化（今）")
    print("   Phase 2: より大きなモデル・データセット（近未来）")  
    print("   Phase 3: DeepSpeed移行（クラウド環境）")

def main():
    """メイン実行"""
    print("🚀 LISA-Gemma3 DDP準備テスト")
    print("=" * 50)
    print("目的: train_simple_test.py の成功をベースに")
    print("     DDPインターフェース対応を確認")
    print("")
    
    # テスト実行
    success = test_single_gpu_with_ddp_interface()
    
    # 次のステップ提案
    suggest_next_steps()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 