#!/usr/bin/env python3
"""
LISA-Gemma3 簡易DDP学習スクリプト
quick_ddp_test.pyで動作確認済みのロジックをベース
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
import time

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
    
    # デフォルトの設定ファイル検索順序
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

class SimpleProgressTracker:
    """簡単な進捗追跡"""
    def __init__(self, name):
        self.name = name
        self.total_loss = 0
        self.count = 0
    
    def update(self, loss):
        self.total_loss += loss
        self.count += 1
    
    def average(self):
        return self.total_loss / self.count if self.count > 0 else 0

def parse_args():
    """引数解析"""
    parser = argparse.ArgumentParser(description="LISA-Gemma3 簡易DDP学習")
    
    # 基本設定
    parser.add_argument("--batch_size", default=4, type=int, help="バッチサイズ")
    parser.add_argument("--steps_per_epoch", default=10, type=int, help="エポックあたりステップ数")
    parser.add_argument("--epochs", default=1, type=int, help="エポック数")
    parser.add_argument("--lr", default=1e-4, type=float, help="学習率")
    parser.add_argument("--exp_name", default="ddp_simple", type=str, help="実験名")
    
    # 設定ファイル
    parser.add_argument("--config_path", default=None, type=str, help="設定ファイルパス")
    
    # DDP設定
    parser.add_argument("--world_size", default=1, type=int, help="プロセス数")
    parser.add_argument("--local_rank", default=0, type=int, help="ローカルランク")
    
    return parser.parse_args()

def setup_model_and_data(args, rank=0):
    """モデルとデータの設定"""
    if rank == 0:
        print("=== モデルとデータの設定 ===")
    
    # Gemmaプロセッサーの初期化
    gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
    
    # SEGトークンの追加
    seg_token = "[SEG]"
    if seg_token not in gemma_processor.tokenizer.get_vocab():
        gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
    
    if rank == 0:
        print(f"✅ Gemmaプロセッサー初期化完了")
    
    # モデル設定
    model_config = LisaGemmaConfig(
        gemma_model_id=config.GEMMA_MODEL_ID,
        sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
        seg_token_idx=gemma_processor.tokenizer.convert_tokens_to_ids(seg_token),
        gemma_hidden_size=config.GEMMA_HIDDEN_SIZE,
        sam_prompt_embed_dim=config.SEG_PROJECTION_DIM,
    )
    
    # モデル初期化
    model = LisaGemmaForCausalLM(model_config)
    model.resize_token_embeddings(len(gemma_processor.tokenizer))
    
    # LoRA適用
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    if rank == 0:
        print(f"✅ モデル初期化・LoRA適用完了")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   総パラメータ: {total_params:,}")
        print(f"   訓練可能: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # データセット
    dataset = HybridDataset(
        base_image_dir=config.DATASET_BASE_DIR,
        gemma_processor=gemma_processor,
        samples_per_epoch=args.batch_size * args.steps_per_epoch * args.world_size,
        precision="bf16",
        gemma_image_size=config.GEMMA_IMAGE_SIZE,
        sam_image_size=config.SAM_IMAGE_SIZE,
        dataset="reason_seg",
        sample_rate=[1.0],
        reason_seg_data="ReasonSeg|train",
    )
    
    if rank == 0:
        print(f"✅ データセット作成完了: {len(dataset)} サンプル")
    
    return model, dataset, gemma_processor

def single_gpu_train(args):
    """単一GPU学習"""
    print(f"🚀 単一GPU DDP学習開始")
    print(f"実験名: {args.exp_name}")
    
    # デバイス設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # モデルとデータの設定
    model, dataset, gemma_processor = setup_model_and_data(args)
    model = model.to(device)
    
    # データローダー
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )
    
    # 損失関数
    loss_fn = CompositeLoss(
        ce_loss_weight=config.CE_LOSS_WEIGHT,
        dice_loss_weight=config.DICE_LOSS_WEIGHT,
        bce_loss_weight=config.BCE_LOSS_WEIGHT
    )
    
    # TensorBoard
    log_dir = os.path.join(config.LOG_BASE_DIR, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"📝 TensorBoardログ: {log_dir}")
    
    # 学習ループ
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        model.train()
        
        progress_tracker = SimpleProgressTracker("Loss")
        
        for step, batch in enumerate(dataloader):
            if step >= args.steps_per_epoch:
                break
            
            # バッチをGPUに移動
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
            
            # 進捗更新
            progress_tracker.update(loss.item())
            
            # ログ出力
            if step % 5 == 0:
                avg_loss = progress_tracker.average()
                print(f"  Step {step}/{args.steps_per_epoch} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
                
                # TensorBoard記録
                global_step = epoch * args.steps_per_epoch + step
                writer.add_scalar("Loss/total", loss.item(), global_step)
                writer.add_scalar("Loss/avg", avg_loss, global_step)
        
        # エポック終了
        avg_loss = progress_tracker.average()
        print(f"Epoch {epoch+1} 完了 | 平均損失: {avg_loss:.4f}")
    
    # 学習完了
    writer.close()
    print("🎉 学習完了!")

def main():
    """メイン関数"""
    args = parse_args()
    
    if args.world_size == 1:
        # 単一GPU実行
        single_gpu_train(args)
    else:
        # マルチGPU実行（将来対応）
        print("⚠️  マルチGPU学習は将来対応予定です。現在は単一GPU (--world_size 1) のみサポート。")
        sys.exit(1)

if __name__ == "__main__":
    main() 