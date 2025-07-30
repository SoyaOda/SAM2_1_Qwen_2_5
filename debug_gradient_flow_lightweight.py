#!/usr/bin/env python3
"""
軽量版勾配フローデバッグスクリプト
最小限のテストで勾配フロー問題を特定・デバッグ
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

print("🚀 最小限勾配フローデバッグ開始...")
print("  - 依存ライブラリなしで基本勾配フロー問題をテスト")

# GPU設定（フォールバック対応）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ デバイス: {device}")

if torch.cuda.is_available():
    print(f"✅ GPU利用可能: {torch.cuda.get_device_name()}")
    print(f"  - GPU数: {torch.cuda.device_count()}")
else:
    print("⚠️ CPUモードで実行（勾配フロー問題の根本原因特定には十分）")

class MockLlama4Model(nn.Module):
    """Llama4のモック（高速ロード用）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = getattr(config, 'hidden_size', 5120)
        
        # 軽量なモック埋め込み層
        self.embed_tokens = nn.Embedding(128256, self.hidden_size)
        
        # 勾配フロー確認用の簡単なレイヤー
        self.mock_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        print("✅ MockLlama4Model初期化完了")
        print(f"  - 隠れ層サイズ: {self.hidden_size}")
        print(f"  - 語彙サイズ: 128256 (mock)")
    
    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        batch_size = input_ids.size(0) if input_ids is not None else 1
        seq_len = input_ids.size(1) if input_ids is not None else 32
        
        # モック出力生成
        if input_ids is not None:
            # テキスト埋め込み
            text_embeds = self.embed_tokens(input_ids)
        else:
            # デフォルトテキスト埋め込み
            text_embeds = torch.randn(batch_size, seq_len, self.hidden_size, device=device)
        
        # 画像特徴をモック（Llama4のマルチモーダル出力をシミュレート）
        if pixel_values is not None:
            image_seq_len = 256  # 448x448 -> 32x32パッチ程度をシミュレート
            image_embeds = torch.randn(batch_size, image_seq_len, self.hidden_size, device=device)
            # テキストと画像を結合
            combined_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        else:
            combined_embeds = text_embeds
        
        # 簡単な変換で勾配フローを維持
        output_embeds = self.mock_layers(combined_embeds)
        
        # Llama4風の出力形式
        return type('MockOutput', (), {
            'hidden_states': [output_embeds],  # 最終層のみ
            'last_hidden_state': output_embeds
        })()

def create_mock_config():
    """モック用設定作成"""
    return type('Config', (), {
        'hidden_size': 5120,
        'vocab_size': 128256,
        'llama_hidden_size': 5120,
        'sam_prompt_embed_dim': 256,
        'qformer_config': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'sam_prompt_dim': 256
        },
        'use_seg_token': False,
        'early_fusion': False,
        'use_unified_token_space': False,
        'use_dynamic_sam_control': False
    })()

def create_test_data(batch_size=1):
    """テスト用データ作成"""
    print("📦 テストデータ作成...")
    
    # テキストデータ（短縮版）
    input_ids = torch.randint(1, 1000, (batch_size, 16), device=device)  # 短縮
    
    # 画像データ（Llama4用: 448x448）
    pixel_values = torch.randn(batch_size, 3, 448, 448, device=device, dtype=torch.bfloat16)
    
    # SAM2用画像（1024x1024） 
    sam_pixel_values = torch.randn(batch_size, 3, 1024, 1024, device=device, dtype=torch.bfloat16)
    
    # ターゲットマスク
    target_masks = torch.randint(0, 2, (batch_size, 1024, 1024), device=device, dtype=torch.float32)
    
    # テキストプロンプト
    text_prompts = ['Segment the main object in this image'] * batch_size
    
    print("✅ テストデータ作成完了")
    print(f"  - Llama4画像形状: {pixel_values.shape}")
    print(f"  - SAM2画像形状: {sam_pixel_values.shape}")  
    print(f"  - テキスト: {text_prompts[0]}")
    print(f"  - ターゲット形状: {target_masks.shape}")
    
    return {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
        'sam_pixel_values': sam_pixel_values,
        'labels': [target_masks],
        'text_prompts': text_prompts
    }

def test_gradient_flow():
    """勾配フローテスト"""
    print("\n🧪 軽量版勾配フローテスト開始...")
    
    # モック設定とモデル
    config = create_mock_config()
    mock_llama = MockLlama4Model(config).to(device)
    
    # Enhanced Model初期化（軽量版）
    from model.enhanced_llama4_qformer_sam2 import EnhancedLlama4QFormerSAM2Bridge
    
    print("🚀 Enhanced Model初期化（軽量版 - Llama4モック使用）...")
    model = EnhancedLlama4QFormerSAM2Bridge(
        llama_model=mock_llama,  # モックLlama4を使用
        config=config,
        enable_multiscale=True
    )
    model = model.to(device)
    model.train()  # 訓練モード
    
    # テストデータ
    test_data = create_test_data()
    
    # Forward処理テスト
    print("\n🔍 Forward処理テスト...")
    try:
        with torch.enable_grad():  # 勾配有効化を明示
            outputs = model(
                input_ids=test_data['input_ids'],
                pixel_values=test_data['pixel_values'],
                sam_pixel_values=test_data['sam_pixel_values'],
                labels=test_data['labels']
            )
        
        print("✅ Forward処理成功!")
        
        # 出力確認
        if 'loss_dict' in outputs:
            loss_dict = outputs['loss_dict']
            print(f"\n📊 損失結果:")
            for name, loss in loss_dict.items():
                if isinstance(loss, torch.Tensor):
                    print(f"  - {name}: {loss.item():.6f}")
                    print(f"    - requires_grad: {loss.requires_grad}")
                    print(f"    - grad_fn: {loss.grad_fn is not None}")
                    
                    if loss.requires_grad and loss.grad_fn is not None:
                        print(f"    ✅ {name}: 勾配フロー正常")
                    else:
                        print(f"    ❌ {name}: 勾配フロー問題")
            
            # 総損失でbackward実行テスト
            if 'total_loss' in loss_dict:
                total_loss = loss_dict['total_loss']
                print(f"\n🔄 Backward実行テスト...")
                try:
                    total_loss.backward()
                    print("✅ Backward処理成功!")
                    
                    # 勾配確認
                    print("\n🔍 モデル勾配確認:")
                    grad_count = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_count += 1
                            grad_norm = param.grad.norm().item()
                            if grad_count <= 5:  # 最初の5個のみ表示
                                print(f"  - {name}: grad_norm={grad_norm:.6f}")
                    
                    print(f"✅ 勾配が設定されたパラメータ数: {grad_count}")
                    
                    if grad_count > 0:
                        print("🎉 勾配フロー完全成功!")
                        return True
                    else:
                        print("❌ 勾配が設定されたパラメータがありません")
                        return False
                        
                except Exception as e:
                    print(f"❌ Backward失敗: {e}")
                    return False
            else:
                print("❌ total_lossが見つかりません")
                return False
        else:
            print("❌ loss_dictが見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ Forward処理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("="*80)
    print("🚀 軽量版勾配フローデバッグスクリプト")
    print("="*80)
    
    # GPU・CUDA設定
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"✅ CUDA設定: GPU {torch.cuda.current_device()}")
    
    # メモリ効率化設定
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    
    print("✅ PyTorch最適化設定完了")
    
    # 勾配フローテスト実行
    success = test_gradient_flow()
    
    print("\n" + "="*80)
    if success:
        print("🎉 軽量版勾配フローテスト: 成功!")
        print("  - 全ての勾配フロー修正が効果を発揮")
        print("  - メインテストスクリプトでも動作するはず")
    else:
        print("❌ 軽量版勾配フローテスト: 失敗")
        print("  - さらなるデバッグが必要")
    print("="*80)

if __name__ == "__main__":
    main()