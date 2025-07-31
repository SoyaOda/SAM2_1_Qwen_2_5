"""
高度な学習設定（2025年最新のベストプラクティス）
"""

# メモリ最適化設定
MEMORY_OPTIMIZATION = {
    "use_8bit_adam": True,                  # 8bit Optimizer（メモリ25%削減）
    "gradient_checkpointing": True,         # Gradient Checkpointing（メモリ30%削減）
    "cpu_offload": False,                   # CPUオフロード（現在は不要）
    "mixed_precision": "bf16",              # 混合精度（bf16推奨）
    "tf32": True,                          # A100でTF32有効化
}

# 学習率設定（改善版）
LEARNING_RATE_CONFIG = {
    "base_lr": 2e-4,                       # 基本学習率（2e-4推奨）
    "min_lr": 1e-6,                        # 最小学習率
    "warmup_steps": 300,                   # Warmupステップ
    "scheduler_type": "cosine",            # スケジューラタイプ
    "layer_wise_lr_decay": 0.9,           # LLRD減衰率
}

# バッチサイズ設定（改善版）
BATCH_CONFIG = {
    "batch_size": 1,                       # 物理バッチサイズ
    "gradient_accumulation_steps": 16,     # 勾配累積（実効BS=16）
    "dataloader_num_workers": 8,           # DataLoaderワーカー
    "pin_memory": True,                    # メモリピン留め
    "persistent_workers": True,            # ワーカー永続化
}

# SAM特有の設定（2025年最新）
SAM_TRAINING_CONFIG = {
    # マルチスケール学習
    "multiscale_training": True,
    "training_scales": [0.5, 0.75, 1.0, 1.25, 1.5],
    
    # ポイントサンプリング戦略
    "point_sampling_strategy": "importance",  # importance, uniform, hybrid
    "num_points_per_mask": 64,
    
    # 損失関数の重み（調整版）
    "mask_loss_coefficient": 20.0,         # CELoss係数（LISA論文準拠）
    "dice_coefficient": 1.0,               # DICE係数（増加）
    "focal_coefficient": 1.0,              # Focal Loss係数
    "iou_coefficient": 1.0,                # IoU Loss係数
    
    # データ拡張
    "augmentation": {
        "random_flip": True,
        "random_scale": [0.8, 1.2],
        "random_crop": True,
        "color_jitter": 0.4,
    }
}

# 正則化設定
REGULARIZATION_CONFIG = {
    "weight_decay": 0.05,                  # 重み減衰（0.05推奨）
    "dropout": 0.1,                        # ドロップアウト
    "gradient_clip": 0.5,                  # 勾配クリッピング（厳格化）
    "label_smoothing": 0.1,                # ラベルスムージング
}

# モニタリング設定
MONITORING_CONFIG = {
    "log_steps": 10,                       # ログ出力間隔
    "eval_steps": 500,                     # 評価間隔
    "save_steps": 500,                     # 保存間隔
    "save_total_limit": 3,                 # 保存するチェックポイント数
    "load_best_model_at_end": True,        # 最良モデルをロード
    "metric_for_best_model": "eval_loss",  # 最良モデルの指標
    "greater_is_better": False,            # 小さいほど良い
}

# 実験的機能（オプション）
EXPERIMENTAL_FEATURES = {
    # Focal Loss（クラス不均衡対策）
    "use_focal_loss": True,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    
    # Stochastic Depth（正則化）
    "use_stochastic_depth": False,
    "stochastic_depth_rate": 0.1,
    
    # EMA（Exponential Moving Average）
    "use_ema": True,
    "ema_decay": 0.9999,
    
    # Gradient Accumulation with Different LR
    "differential_lr": {
        "llama_lora": 5e-5,      # Llama LoRA層
        "projector": 2e-4,       # プロジェクタ（高め）
        "sam_decoder": 1e-5,     # SAMデコーダ（低め）
    }
}

# データセット特有の設定
DATASET_SPECIFIC_CONFIG = {
    "Semantic_Segmentation": {
        "loss_weight": 1.0,
        "specific_lr": None,
    },
    "Referring_Segmentation": {
        "loss_weight": 1.2,      # 少し重視
        "specific_lr": None,
    },
    "VQA": {
        "loss_weight": 0.8,      # 少し軽視
        "specific_lr": None,
    },
    "Reasoning_Segmentation": {
        "loss_weight": 1.5,      # 最重視
        "specific_lr": None,
    }
}

def get_optimized_config():
    """最適化された設定を統合して返す"""
    return {
        "memory": MEMORY_OPTIMIZATION,
        "learning_rate": LEARNING_RATE_CONFIG,
        "batch": BATCH_CONFIG,
        "sam": SAM_TRAINING_CONFIG,
        "regularization": REGULARIZATION_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "experimental": EXPERIMENTAL_FEATURES,
        "dataset_specific": DATASET_SPECIFIC_CONFIG,
    }