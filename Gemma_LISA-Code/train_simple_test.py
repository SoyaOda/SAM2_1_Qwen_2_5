#!/usr/bin/env python3
"""
LISA-Gemma3 シンプル学習スクリプト
DeepSpeedなしの単一GPU用
"""

import argparse
import os
import sys
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model

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
from utils.dataset import LisaGemma3Dataset, collate_fn_gemma3
from utils.utils import dict_to_cuda

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="LISA-Gemma3 シンプル学習スクリプト（DeepSpeedなし）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 学習設定
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="学習率")
    parser.add_argument("--exp_name", type=str, default="lisa_gemma3_simple_test", help="実験名")
    
    # 設定ファイル
    parser.add_argument("--config_path", default=None, type=str, help="設定ファイルパス")
    
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
    
    # トークン埋め込み層のサイズを新しい語彙サイズに合わせて拡張
    model.resize_token_embeddings(len(gemma_processor.tokenizer))
    print(f"✓ トークン埋め込み層をサイズ {len(gemma_processor.tokenizer)} に拡張")
    
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
    
    # より少ないデータでメモリ効率的なテスト
    dataset = LisaGemma3Dataset(
        base_image_dir=config.DATASET_BASE_DIR,
        gemma_processor=gemma_processor,
        samples_per_epoch=10,  # さらに少なくしてメモリ使用量を削減
        dataset="reason_seg",  # セグメンテーションのみに絞る
        sample_rate=[1],  # シンプルな比率
        reason_seg_data="ReasonSeg|train",
        vqa_data="llava_instruct_150k",
        refer_seg_data="refcoco",
        sem_seg_data="ade20k",
        precision="bf16",
        gemma_image_size=config.GEMMA_IMAGE_SIZE,
        sam_image_size=config.SAM_IMAGE_SIZE,
    )
    print(f"✓ データセット作成完了 (サンプル数: {len(dataset)})")
    
    # collate_fn_gemma3はgemma_processorを受け取らない単純な関数
    # test_real_data.pyでも正常に動作していた実装を使用
    
    # データローダーの作成
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_gemma3,  # 単純にcollate_fn_gemma3を使用
        num_workers=0,  # デバッグ用
        pin_memory=True,
    )
    print(f"✓ データローダー作成完了 (バッチサイズ: {args.batch_size})")
    
    return dataset, dataloader

def main():
    """メイン関数"""
    args = parse_args()
    
    print(f"🚀 LISA-Gemma3 シンプル学習開始")
    print(f"実験名: {args.exp_name}")
    print(f"設定: バッチサイズ={args.batch_size}, 学習率={args.lr}")
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # モデルとプロセッサーのセットアップ
    model, gemma_processor = setup_model_and_tokenizer()
    
    # LoRAの適用
    model = setup_lora(model)
    
    # モデルをGPUに移動
    model.to(device)
    print(f"✓ モデルを{device}に移動")
    
    # データセットの作成
    dataset, dataloader = create_dataset_and_dataloader(gemma_processor, args)
    
    # 損失関数の初期化
    loss_fn = CompositeLoss(
        ce_loss_weight=config.CE_LOSS_WEIGHT,
        dice_loss_weight=config.DICE_LOSS_WEIGHT,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
    )
    
    # オプティマイザの設定
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )
    print(f"✓ AdamWオプティマイザ初期化完了 (学習率: {args.lr})")
    
    # TensorBoardの設定
    log_dir = os.path.join(config.LOG_BASE_DIR, args.exp_name)
    writer = SummaryWriter(log_dir)
    print(f"📝 TensorBoardログ: {log_dir}")
    
    print("\n🎯 学習開始...")
    
    # 1バッチだけでテスト
    model.train()
    data_iter = iter(dataloader)
    batch = next(data_iter)
    
    print("📦 バッチ情報:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    # バッチをGPUに移動
    batch = dict_to_cuda(batch)
    
    print("\n🔄 フォワードパス実行中...")
    try:
        outputs = model(**batch)
        print("✓ フォワードパス成功")
        
        print("📊 モデル出力:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        # 損失計算
        print("\n🧮 損失計算中...")
        losses = loss_fn(outputs, batch)
        print("✓ 損失計算成功")
        
        print("📈 損失値:")
        for key, value in losses.items():
            print(f"  {key}: {value.item():.4f}")
        
        print("\n🎉 テスト成功！全ての処理が正常に動作しました。")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
    
    writer.close()

if __name__ == "__main__":
    main() 