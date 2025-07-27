#!/usr/bin/env python3
"""
シングルエンコーダー構成のモデルアーキテクチャテスト
第2部フェーズ2の実装が正しく動作することを検証
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
from utils.dataset import preprocess_sam_image
import config_linux

def create_dummy_batch(batch_size: int = 2, device: str = "cpu", seg_token_id: int = 201135) -> Dict[str, torch.Tensor]:
    """ダミーバッチデータの作成"""
    # テキスト入力（[SEG]トークンを含む）
    seq_length = 50
    vocab_size = 202048  # 実際の語彙サイズ（ログから）
    
    # ランダムな入力IDを生成
    input_ids = torch.randint(0, vocab_size-1000, (batch_size, seq_length))  # 特殊トークンを避ける
    # 各サンプルに1つずつ[SEG]トークンを挿入
    for i in range(batch_size):
        seg_position = torch.randint(10, seq_length-10, (1,)).item()
        input_ids[i, seg_position] = seg_token_id  # 実際の[SEG]トークンID
    
    # アテンションマスク（全て1）
    attention_mask = torch.ones(batch_size, seq_length)
    
    # SAM用画像（1024x1024）
    sam_pixel_values = torch.randn(batch_size, 3, 1024, 1024)
    
    # ラベル（言語モデリング用）
    labels = input_ids.clone()
    labels[:, :10] = -100  # 最初の10トークンは無視
    
    # 正解マスク（元画像サイズのリスト）
    ground_truth_masks = []
    original_sizes = []
    for i in range(batch_size):
        # ランダムな元画像サイズ
        h = torch.randint(480, 720, (1,)).item()
        w = torch.randint(640, 960, (1,)).item()
        mask = torch.randint(0, 2, (h, w)).float()
        ground_truth_masks.append(mask)
        original_sizes.append((h, w))
    
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'sam_pixel_values': sam_pixel_values.to(device),
        'labels': labels.to(device),
        'ground_truth_masks': ground_truth_masks,
        'original_sizes': original_sizes,
    }


def test_model_initialization():
    """モデル初期化のテスト"""
    print("=== モデル初期化テスト ===")
    
    # 動的コンパイルを無効化（GPU分散エラー回避）
    torch.compiler.disable()
    print("✅ 動的コンパイル無効化: GPU分散エラー回避のため")
    
    # GPUが利用可能か確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    
    # 利用可能なGPU数を確認
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"利用可能なGPU数: {num_gpus}")
        
        # 各GPUのメモリを表示
        for i in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_memory:.1f} GB")
    
    # 設定を作成
    # 8GPU環境では自動分散を使用
    if device == "cuda":
        device_map = "auto"  # HuggingFaceの自動分散機能を使用
        print("device_map='auto'を使用（自動GPU分散）")
    else:
        device_map = "cpu"
    
    # config_linux.get_lisa_model_config()を使用（verify_llama4_lisa_gradients.py準拠）
    if hasattr(config_linux, 'get_lisa_model_config'):
        config_dict = config_linux.get_lisa_model_config()
        # device_mapを上書き
        config_dict['device_map'] = device_map
        config = LisaLlama4Config(**config_dict)
    else:
        # フォールバック
        config = LisaLlama4Config(
            llama_model_id=config_linux.LLAMA_MODEL_ID,
            sam_checkpoint_path=config_linux.SAM_CHECKPOINT_PATH,
            seg_token="[SEG]",
            llama_hidden_size=5120,
            sam_prompt_embed_dim=256,
            llama_image_size=448,
            sam_image_size=1024,
            model_max_length=2048,
            attn_implementation="eager",
            device_map=device_map,
            torch_dtype="bfloat16" if device == "cuda" else "float32",
        )
    
    # モデルを初期化
    try:
        model = LisaLlama4ForCausalLM(config)
        print("✅ モデル初期化成功")
    except Exception as e:
        print(f"❌ モデル初期化失敗: {e}")
        return None
    
    # パラメータ情報を表示
    param_info = model.get_trainable_parameters_info()
    print(f"\n総パラメータ数: {param_info['total_params']:,}")
    print(f"学習可能パラメータ数: {param_info['trainable_params']:,}")
    print(f"学習可能パラメータ割合: {param_info['trainable_percentage_str']}")
    print(f"メモリ効率的: {param_info['memory_efficient']}")
    
    return model


def test_forward_pass(model: Optional[nn.Module] = None):
    """フォワードパスのテスト"""
    print("\n=== フォワードパステスト ===")
    
    if model is None:
        print("❌ モデルが提供されていません")
        return
    
    # モデルのデバイスを正しく取得（SAMモデルのデバイスを確認）
    if hasattr(model, 'sam_model') and model.sam_model is not None:
        device = next(model.sam_model.parameters()).device
    else:
        device = next(model.parameters()).device
    
    print(f"モデルデバイス: {device}")
    
    # SEGトークンIDを取得
    seg_token_id = model.seg_token_id if hasattr(model, 'seg_token_id') else 201135
    print(f"SEGトークンID: {seg_token_id}")
    
    # ダミーバッチを作成
    batch = create_dummy_batch(batch_size=2, device=str(device), seg_token_id=seg_token_id)
    
    # SEGトークンの位置を確認
    seg_positions = (batch['input_ids'] == seg_token_id).nonzero(as_tuple=True)
    print(f"SEGトークン位置: {list(zip(seg_positions[0].tolist(), seg_positions[1].tolist()))}")
    
    # モデルを評価モードに
    model.eval()
    
    try:
        with torch.no_grad():
            # フォワードパス実行
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                sam_pixel_values=batch['sam_pixel_values'],
                labels=batch['labels'],
                ground_truth_masks=batch['ground_truth_masks'],
                original_sizes=batch['original_sizes'],
            )
        
        print("✅ フォワードパス成功")
        
        # 出力内容を詳細に確認
        print(f"\n出力の型: {type(outputs)}")
        if isinstance(outputs, dict):
            print(f"出力のキー: {list(outputs.keys())}")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} (device: {value.device})")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: List[{len(value)}]")
                else:
                    print(f"  {key}: {type(value)}")
        
        # 出力の検証（エラーを出さずに確認）
        if 'loss' not in outputs:
            print("⚠️ 警告: 損失が出力に含まれていません")
            if 'losses' in outputs:
                print("  'losses'キーが見つかりました:", outputs['losses'].keys() if isinstance(outputs['losses'], dict) else outputs['losses'])
        
        if 'pred_masks' not in outputs:
            print("⚠️ 警告: 予測マスクが出力に含まれていません")
            if 'predicted_masks' in outputs:
                print("  'predicted_masks'キーが見つかりました")
        
        # 損失の表示
        if 'loss' in outputs:
            print(f"\n出力内容:")
            print(f"  - 損失: {outputs['loss'].item():.4f}")
        elif 'losses' in outputs and isinstance(outputs['losses'], dict) and 'total_loss' in outputs['losses']:
            print(f"\n出力内容:")
            print(f"  - 総損失: {outputs['losses']['total_loss'].item():.4f}")
            if 'lm_loss' in outputs['losses']:
                print(f"  - 言語モデリング損失: {outputs['losses']['lm_loss'].item():.4f}")
            if 'seg_loss' in outputs['losses']:
                print(f"  - セグメンテーション損失: {outputs['losses']['seg_loss'].item():.4f}")
        
        # マスクの表示
        mask_key = 'pred_masks' if 'pred_masks' in outputs else 'predicted_masks'
        if mask_key in outputs and outputs[mask_key] is not None:
            masks = outputs[mask_key]
            if isinstance(masks, torch.Tensor):
                print(f"  - 予測マスク形状: {masks.shape}")
            elif isinstance(masks, list):
                print(f"  - 予測マスク数: {len(masks)}")
                for i, mask in enumerate(masks):
                    print(f"  - 予測マスク[{i}]形状: {mask.shape}")
        
    except Exception as e:
        print(f"❌ フォワードパス失敗: {e}")
        import traceback
        traceback.print_exc()


def test_gradient_flow(model: Optional[nn.Module] = None):
    """勾配フローのテスト"""
    print("\n=== 勾配フローテスト ===")
    
    if model is None:
        print("❌ モデルが提供されていません")
        return
    
    # モデルのデバイスを正しく取得（SAMモデルのデバイスを確認）
    if hasattr(model, 'sam_model') and model.sam_model is not None:
        device = next(model.sam_model.parameters()).device
    else:
        device = next(model.parameters()).device
    
    print(f"モデルデバイス: {device}")
    
    # SEGトークンIDを取得
    seg_token_id = model.seg_token_id if hasattr(model, 'seg_token_id') else 201135
    print(f"SEGトークンID: {seg_token_id}")
    
    # ダミーバッチを作成
    batch = create_dummy_batch(batch_size=1, device=str(device), seg_token_id=seg_token_id)
    
    # モデルを訓練モードに
    model.train()
    
    try:
        # フォワードパス実行
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            sam_pixel_values=batch['sam_pixel_values'],
            labels=batch['labels'],
            ground_truth_masks=batch['ground_truth_masks'],
            original_sizes=batch['original_sizes'],
        )
        
        # バックワード実行
        if 'loss' in outputs:
            loss = outputs['loss']
        elif 'losses' in outputs and isinstance(outputs['losses'], dict) and 'total_loss' in outputs['losses']:
            loss = outputs['losses']['total_loss']
        else:
            print("❌ 損失が見つかりません")
            print(f"出力キー: {list(outputs.keys())}")
            return
        
        loss.backward()
        
        print("✅ バックワード成功")
        
        # 勾配フローの検証
        print("\n勾配フロー検証:")
        
        # Llama4ビジョンタワーの勾配確認（バイパスされているはず）
        vision_tower_grad_found = False
        if hasattr(model.llama_model, 'vision_tower'):
            for name, param in model.llama_model.vision_tower.named_parameters():
                if param.grad is not None:
                    vision_tower_grad_found = True
                    print(f"⚠️ ビジョンタワーに勾配発見: {name}")
                    break
        
        if not vision_tower_grad_found:
            print("✅ ビジョンタワーに勾配なし（正しくバイパスされています）")
        
        # プロジェクタの勾配確認（学習されているはず）
        projector_grad_found = False
        for name, param in model.multi_modal_projector.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                projector_grad_found = True
                print(f"✅ プロジェクタに勾配あり: {name}, 勾配ノルム: {param.grad.norm().item():.6f}")
        
        if not projector_grad_found:
            print("❌ プロジェクタに勾配なし（問題があります）")
        
        # LoRAアダプタの勾配確認（もし有効な場合）
        lora_grad_found = False
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad and param.grad is not None:
                lora_grad_found = True
                print(f"✅ LoRAアダプタに勾配あり: {name}, 勾配ノルム: {param.grad.norm().item():.6f}")
                break
        
        if not lora_grad_found:
            print("ℹ️ LoRAアダプタなし、または勾配なし")
        
        # 勾配をクリア
        model.zero_grad()
        
    except Exception as e:
        print(f"❌ 勾配フローテスト失敗: {e}")
        import traceback
        traceback.print_exc()


def test_data_flow():
    """データフローのテスト（テンソル形状とデータ型の確認）"""
    print("\n=== データフローテスト ===")
    
    # 仕様書で期待される形状
    print("期待されるテンソル形状:")
    print("  - sam_pixel_values: [B, 3, 1024, 1024] (float32)")
    print("  - input_ids: [B, seq_length]")
    print("  - image_embeddings (SAM出力): [B, 256, 64, 64]")
    print("  - hidden_states (Llama出力): [B, seq_length, 5120]")
    print("  - predicted_masks: [B, 1, 256, 256] → リサイズ後 [B, 1, H_orig, W_orig]")
    
    # 実際のバッチでテスト
    batch = create_dummy_batch(batch_size=2, device="cpu")
    
    print("\n実際のテンソル形状:")
    print(f"  - sam_pixel_values: {batch['sam_pixel_values'].shape}, dtype: {batch['sam_pixel_values'].dtype}")
    print(f"  - input_ids: {batch['input_ids'].shape}, dtype: {batch['input_ids'].dtype}")
    print(f"  - ground_truth_masks[0]: {batch['ground_truth_masks'][0].shape}")
    print(f"  - original_sizes[0]: {batch['original_sizes'][0]}")


def test_single_encoder_architecture():
    """シングルエンコーダーアーキテクチャの完全テスト"""
    print("=" * 60)
    print("シングルエンコーダーアーキテクチャ統合テスト")
    print("=" * 60)
    
    # 1. モデル初期化
    model = test_model_initialization()
    if model is None:
        print("\n❌ テスト中止：モデル初期化に失敗しました")
        return
    
    # 2. フォワードパステスト
    test_forward_pass(model)
    
    # 3. 勾配フローテスト
    test_gradient_flow(model)
    
    # 4. データフローテスト
    test_data_flow()
    
    print("\n" + "=" * 60)
    print("✅ シングルエンコーダーアーキテクチャテスト完了")
    print("=" * 60)


if __name__ == "__main__":
    # メインテストを実行
    test_single_encoder_architecture()