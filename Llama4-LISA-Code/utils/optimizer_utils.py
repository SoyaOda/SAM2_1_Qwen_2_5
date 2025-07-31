"""
最適化関連ユーティリティ
Layer-wise Learning Rate Decay (LLRD)などの高度な最適化手法
"""

import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_llrd_parameters(model, base_lr: float, decay_rate: float = 0.9) -> List[Dict[str, Any]]:
    """
    Layer-wise Learning Rate Decay (LLRD)用のパラメータグループを作成
    深い層ほど学習率を低く設定することで、事前学習済み知識を保持
    
    Args:
        model: モデル
        base_lr: 基本学習率
        decay_rate: 層ごとの減衰率（0.9推奨）
    
    Returns:
        パラメータグループのリスト
    """
    # レイヤー名とパラメータのマッピング
    layer_params = {}
    
    # モデルの全パラメータを層ごとに分類
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 層番号を抽出（例: "base_model.model.llama_model.layers.23.self_attn.q_proj"）
        if "layers." in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
        else:
            # 層番号がない場合は最上位層として扱う
            layer_num = -1
            
        if layer_num not in layer_params:
            layer_params[layer_num] = []
        layer_params[layer_num].append(param)
    
    # 層ごとに学習率を設定
    param_groups = []
    max_layer = max(layer_params.keys())
    
    for layer_num, params in sorted(layer_params.items()):
        # 深い層ほど低い学習率
        if layer_num == -1:
            # 最上位層（プロジェクタなど）は基本学習率
            lr = base_lr
        else:
            # 層が深いほど低い学習率
            lr = base_lr * (decay_rate ** (max_layer - layer_num))
        
        param_groups.append({
            'params': params,
            'lr': lr,
            'layer': layer_num
        })
        
        logger.info(f"Layer {layer_num}: lr={lr:.2e}, params={len(params)}")
    
    return param_groups


def create_optimizer_with_llrd(model, base_lr: float, weight_decay: float = 0.05, 
                              use_8bit: bool = True, decay_rate: float = 0.9):
    """
    LLRD対応のオプティマイザーを作成
    
    Args:
        model: モデル
        base_lr: 基本学習率
        weight_decay: 重み減衰
        use_8bit: 8bit optimizer使用フラグ
        decay_rate: 層ごとの学習率減衰率
    
    Returns:
        オプティマイザー
    """
    # LLRDパラメータグループを取得
    param_groups = get_llrd_parameters(model, base_lr, decay_rate)
    
    # オプティマイザー作成
    try:
        import bitsandbytes as bnb
        if use_8bit:
            logger.info("✅ 8bit AdamW with LLRD")
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        else:
            raise ImportError
    except ImportError:
        logger.info("標準AdamW with LLRD")
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    return optimizer


def get_param_count_by_layer(model) -> Dict[int, int]:
    """層ごとのパラメータ数を取得"""
    layer_counts = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "layers." in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
        else:
            layer_num = -1
            
        if layer_num not in layer_counts:
            layer_counts[layer_num] = 0
        layer_counts[layer_num] += param.numel()
    
    return layer_counts