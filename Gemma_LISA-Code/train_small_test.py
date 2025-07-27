#!/usr/bin/env python3
"""
LISA-Gemma3 小規模テスト学習スクリプト
本格学習前の動作確認用
Original LISA train_ds.py をベースに、Gemma3統合版として実装
"""

import argparse
import os
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 動的設定管理機能を追加
def get_config():
    """動的設定読み込み（環境変数対応）"""
    config_path = os.environ.get('LISA_CONFIG_PATH', None)
    
    if config_path:
        # 環境変数で指定された設定ファイル
        try:
            config_module = __import__(config_path)
            print(f"✓ カスタム設定ファイルを使用: {config_path}")
            return config_module
        except ImportError:
            print(f"⚠️ カスタム設定ファイル {config_path} が見つかりません")
    
    # デフォルトの設定ファイル検索順序（小規模テスト優先）
    config_candidates = ['config_small_test', 'config_linux']
    
    for config_name in config_candidates:
        try:
            config_module = __import__(config_name)
            print(f"✓ 設定ファイルを使用: {config_name}")
            return config_module
        except ImportError:
            continue
    
    raise ImportError("利用可能な設定ファイルが見つかりません")

# 動的設定読み込み
config = get_config()

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn
from utils.utils import AverageMeter, ProgressMeter, dict_to_cuda, intersectionAndUnionGPU

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="LISA-Gemma3 小規模テスト学習スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 学習設定
    parser.add_argument("--batch_size", type=int, default=getattr(config, 'SMALL_TEST_SAMPLES_PER_EPOCH', 40) // 10, help="グローバルバッチサイズ")
    parser.add_argument("--grad_accumulation_steps", type=int, default=4, help="勾配累積ステップ")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="学習率")
    parser.add_argument("--epochs", type=int, default=getattr(config, 'SMALL_TEST_EPOCHS', 2), help="エポック数")
    parser.add_argument("--steps_per_epoch", type=int, default=getattr(config, 'SMALL_TEST_STEPS_PER_EPOCH', 10), help="エポック毎のステップ数")
    parser.add_argument("--print_freq", type=int, default=5, help="ログ出力間隔")
    
    # パス設定
    parser.add_argument("--exp_name", type=str, default="lisa_gemma3_small_test", help="実験名")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config_small_test.json", help="DeepSpeed設定ファイル")
    parser.add_argument("--config_path", default=None, type=str, help="設定ファイルパス")
    
    # DeepSpeed必須引数
    parser.add_argument("--local_rank", type=int, default=0, help="DeepSpeed local rank")
    
    return parser.parse_args()

def setup_model_and_tokenizer():
    """モデルとトークナイザーのセットアップ"""
    print("🔧 モデルとトークナイザーの初期化中...")
    
    # Gemma-3プロセッサーの初期化
    gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
    print(f"✓ Gemma-3プロセッサー初期化完了: {config.GEMMA_MODEL_ID}")
    
    # <SEG>トークンの追加（まだ存在しない場合）
    seg_token = "[SEG]"
    if seg_token not in gemma_processor.tokenizer.get_vocab():
        print(f"🔧 {seg_token}トークンを語彙に追加中...")
        gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        print(f"✓ 語彙サイズ: {len(gemma_processor.tokenizer)}")
    else:
        print(f"✓ {seg_token}トークンは既に存在")
    
    # カスタムモデル設定
    model_config = LisaGemmaConfig(
        gemma_model_id=config.GEMMA_MODEL_ID,
        sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
        seg_token_idx=gemma_processor.tokenizer.convert_tokens_to_ids(seg_token),
        gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),  # 設定ファイルから取得
        sam_prompt_embed_dim=getattr(config, 'SEG_PROJECTION_DIM', 256),  # 設定ファイルから取得
    )
    
    # カスタムモデルの初期化
    print("🔧 LISA-Gemmaモデル初期化中...")
    model = LisaGemmaForCausalLM(model_config)
    
    # DeepSpeed分散テンソル競合回避: トークン拡張を後で実行
    # model.resize_token_embeddings(len(gemma_processor.tokenizer))
    print(f"✓ モデル初期化完了（トークン拡張は後で実行）")
    
    return model, gemma_processor

def setup_lora(model):
    """LoRA設定の適用"""
    print("🔧 LoRA設定の適用中...")
    
    # LoRA設定
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    
    # LoRAの適用
    model = get_peft_model(model, lora_config)
    print("✓ LoRA設定適用完了")
    
    # 訓練可能なパラメータの情報表示
    model.print_trainable_parameters()
    
    return model

def create_dataset_and_dataloader(gemma_processor, args):
    """データセットとデータローダーの作成"""
    print("🔧 データセットとデータローダーの作成中...")
    
    # test_real_data.pyで動作確認済みの引数構造を使用
    dataset = HybridDataset(
        base_image_dir=config.DATASET_BASE_DIR,
        gemma_processor=gemma_processor,
        samples_per_epoch=getattr(config, 'SMALL_TEST_SAMPLES_PER_EPOCH', 40),  # 40サンプル
        dataset="reason_seg||vqa",  # 利用可能なデータセットのみ
        sample_rate=[1, 1],  # シンプルな比率
        reason_seg_data="ReasonSeg|train",
        vqa_data="llava_instruct_150k",
        refer_seg_data="refcoco",
        sem_seg_data="ade20k",
        precision="bf16",
        gemma_image_size=config.GEMMA_IMAGE_SIZE,
        sam_image_size=config.SAM_IMAGE_SIZE,
    )
    print(f"✓ データセット作成完了 (サンプル数: {len(dataset)})")
    
    # collate_fnはgemma_processorを受け取らない単純な関数
    
    # データローダーの作成
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # 単純にcollate_fnを使用
        num_workers=0,  # デバッグ用
        pin_memory=True,
    )
    print(f"✓ データローダー作成完了 (バッチサイズ: {args.batch_size})")
    
    return dataset, dataloader

def create_deepspeed_config(args):
    """DeepSpeed設定の生成"""
    ds_config = {
        "train_batch_size": args.batch_size * args.grad_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "steps_per_print": args.print_freq,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [config.BETA1, config.BETA2],
                "eps": 1e-8,
                "weight_decay": config.WEIGHT_DECAY,
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": args.lr * 0.1,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 5,
            }
        },
        
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},  # CPUオフロードを無効化
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
        },
        
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }
    
    return ds_config

def train_step(model_engine, batch, loss_fn, step, args):
    """1ステップの学習"""
    batch = dict_to_cuda(batch)
    
    # フォワードパス
    outputs = model_engine(**batch)
    
    # 損失計算
    losses = loss_fn(outputs, batch)
    total_loss = losses["total_loss"]
    
    # バックワードパス
    model_engine.backward(total_loss)
    model_engine.step()
    
    # ログ情報の準備
    log_info = {
        "step": step,
        "total_loss": total_loss.item(),
        "text_loss": losses.get("text_loss", torch.tensor(0.0)).item(),
        "mask_loss": losses.get("mask_loss", torch.tensor(0.0)).item(),
        "lr": model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else args.lr,
    }
    
    return log_info

def main():
    """メイン関数"""
    args = parse_args()
    
    print(f"🚀 LISA-Gemma3 小規模テスト学習開始")
    print(f"実験名: {args.exp_name}")
    print(f"設定: バッチサイズ={args.batch_size}, エポック={args.epochs}, ステップ/エポック={args.steps_per_epoch}")
    
    # DeepSpeedの初期化（単一GPU用）
    # deepspeed.init_distributed()  # 単一GPUテストでは不要
    
    # モデルとプロセッサーのセットアップ
    model, gemma_processor = setup_model_and_tokenizer()
    
    # LoRAの適用
    model = setup_lora(model)
    
    # データセットの作成
    dataset, dataloader = create_dataset_and_dataloader(gemma_processor, args)
    
    # 損失関数の初期化
    loss_fn = CompositeLoss(
        ce_loss_weight=config.CE_LOSS_WEIGHT,
        dice_loss_weight=config.DICE_LOSS_WEIGHT,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
    )
    
    # DeepSpeed設定
    ds_config = create_deepspeed_config(args)
    
    # DeepSpeed初期化前にトークン埋め込み調整を実行
    print("🔧 トークン埋め込み層のサイズ調整...")
    vocab_size = len(gemma_processor.tokenizer)
    if hasattr(model.gemma_model, 'embed_tokens'):
        current_size = model.gemma_model.embed_tokens.num_embeddings
        if current_size != vocab_size:
            print(f"   トークン埋め込み層を {current_size} → {vocab_size} に拡張")
            # 手動でembedding層を拡張
            old_embeddings = model.gemma_model.embed_tokens
            new_embeddings = torch.nn.Embedding(vocab_size, old_embeddings.embedding_dim, device=old_embeddings.weight.device)
            new_embeddings.weight.data[:current_size, :] = old_embeddings.weight.data[:current_size, :]
            model.gemma_model.embed_tokens = new_embeddings
            
            # lm_headも同様に拡張
            if hasattr(model.gemma_model, 'lm_head') and model.gemma_model.lm_head.out_features != vocab_size:
                old_lm_head = model.gemma_model.lm_head
                new_lm_head = torch.nn.Linear(old_lm_head.in_features, vocab_size, bias=old_lm_head.bias is not None, device=old_lm_head.weight.device)
                new_lm_head.weight.data[:current_size, :] = old_lm_head.weight.data[:current_size, :]
                if old_lm_head.bias is not None:
                    new_lm_head.bias.data[:current_size] = old_lm_head.bias.data[:current_size]
                model.gemma_model.lm_head = new_lm_head
            print("✓ トークン埋め込み層拡張完了")
    
    # DeepSpeedエンジンの初期化（単一GPU用）
    # 分散学習を明示的に無効化
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config_params=ds_config,
    )
    
    # TensorBoardの設定
    log_dir = os.path.join(config.LOG_BASE_DIR, args.exp_name)
    writer = SummaryWriter(log_dir)
    print(f"📝 TensorBoardログ: {log_dir}")
    
    # 学習ループ
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n📚 エポック {epoch+1}/{args.epochs} 開始")
        model_engine.train()
        
        # エポックごとのメトリクス
        epoch_losses = AverageMeter("EpochLoss", ":.4f")
        
        data_iter = iter(dataloader)
        
        for step in range(args.steps_per_epoch):
            try:
                batch = next(data_iter)
            except StopIteration:
                # データローダーが終了した場合、新しいイテレータを作成
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # 学習ステップ
            log_info = train_step(model_engine, batch, loss_fn, global_step, args)
            
            # メトリクスの更新
            epoch_losses.update(log_info["total_loss"])
            
            # ログ出力
            if global_step % args.print_freq == 0:
                print(f"  ステップ [{global_step:3d}] "
                      f"損失: {log_info['total_loss']:.4f} "
                      f"(テキスト:{log_info['text_loss']:.4f}, マスク:{log_info['mask_loss']:.4f}) "
                      f"学習率: {log_info['lr']:.2e}")
                
                # TensorBoardログ
                for key, value in log_info.items():
                    if key != "step":
                        writer.add_scalar(f"Train/{key}", value, global_step)
            
            global_step += 1
        print(f"📊 エポック {epoch+1} 完了 - 平均損失: {epoch_losses.avg:.4f}")
        writer.add_scalar("Epoch/avg_loss", epoch_losses.avg, epoch)
    
    print("\n🎉 小規模テスト学習完了！")
    print(f"総ステップ数: {global_step}")
    print(f"TensorBoardログ: {log_dir}")
    
    writer.close()

if __name__ == "__main__":
    main() 