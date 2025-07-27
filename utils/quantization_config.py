"""
量子化設定ユーティリティ（edit_config.md推奨値準拠）
"""

import os
import sys
import torch
from transformers import BitsAndBytesConfig

# config_linuxをインポート
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_linux


def get_bnb_config():
    """
    edit_config.md推奨のBitsAndBytes量子化設定を取得
    
    Returns:
        BitsAndBytesConfig: 4ビットNormalFloat量子化設定
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,                              # 4ビットでのロードを有効化
        bnb_4bit_quant_type="nf4",                     # NormalFloat4量子化（精度維持に優れる）
        bnb_4bit_use_double_quant=True,                # 二次量子化（追加で0.4ビット節約）
        bnb_4bit_compute_dtype=torch.bfloat16          # 計算はbfloat16で実行（精度維持）
    )


def get_training_args_template():
    """
    edit_config.md推奨のTrainingArguments設定テンプレートを取得
    
    Returns:
        dict: TrainingArguments用パラメータ辞書
    """
    # config_linuxから設定を取得
    training_config = config_linux.get_training_config()
    
    return {
        "output_dir": "./lisa-llama4-scout-finetuned",
        "per_device_train_batch_size": training_config["batch_size_per_gpu"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "learning_rate": training_config["learning_rate"],
        "num_train_epochs": training_config["epochs"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "warmup_ratio": training_config["warmup_ratio"],
        "optim": training_config["optim_type"],
        "bf16": training_config["mixed_precision"],
        "logging_steps": 10,                            # ログ頻度
        "evaluation_strategy": "steps",                 # ステップごと評価
        "eval_steps": 50,                               # 評価頻度
        "save_strategy": "steps",                       # 保存戦略
        "save_steps": 50,                               # 保存頻度
        "save_total_limit": 2,                          # 最新2つのみ保存
        "gradient_checkpointing": training_config["gradient_checkpointing"],
        "report_to": "wandb",                           # 監視ツール
    }