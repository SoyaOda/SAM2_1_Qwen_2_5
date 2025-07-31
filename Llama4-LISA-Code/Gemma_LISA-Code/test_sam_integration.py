#!/usr/bin/env python3
"""
SAMチェックポイント統合テスト
実際のSAMモデルを読み込んで、Gemma-3との統合動作を確認
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path

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

class SAMIntegrationTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")
        
        # 設定の確認
        self.check_config()
        
    def check_config(self):
        """設定とSAMチェックポイントの確認"""
        print("=== SAM統合テスト設定確認 ===")
        print(f"SAM checkpoint: {config.SAM_CHECKPOINT_PATH}")
        print(f"Gemma model ID: {config.GEMMA_MODEL_ID}")
        
        # SAMチェックポイントの存在と詳細確認
        if config.SAM_CHECKPOINT_PATH and os.path.exists(config.SAM_CHECKPOINT_PATH):
            size = os.path.getsize(config.SAM_CHECKPOINT_PATH)
            print(f"✓ SAMチェックポイント存在確認: {size:,} bytes ({size/1024/1024/1024:.2f} GB)")
            
            if os.access(config.SAM_CHECKPOINT_PATH, os.R_OK):
                print("✓ SAMチェックポイント読み取り権限確認")
            else:
                raise PermissionError(f"SAMチェックポイントの読み取り権限がありません: {config.SAM_CHECKPOINT_PATH}")
        else:
            print(f"⚠️ SAMチェックポイントが見つかりません: {config.SAM_CHECKPOINT_PATH}")
            print("SAMなしモードでテストを続行します")
    
    def test_sam_checkpoint_loading(self):
        """SAMチェックポイントの読み込みテスト"""
        print("\n=== SAMチェックポイント読み込みテスト ===")
        
        if not config.SAM_CHECKPOINT_PATH or not os.path.exists(config.SAM_CHECKPOINT_PATH):
            print("⚠️ SAMチェックポイントが利用できません。テストをスキップします")
            return None
        
        try:
            print("SAMチェックポイント読み込み中...")
            # segment_anythingライブラリを使用してSAMを読み込み
            from model.segment_anything import sam_model_registry
            
            sam = sam_model_registry["vit_h"](checkpoint=config.SAM_CHECKPOINT_PATH)
            print("✓ SAMモデル読み込み成功")
            
            # SAMコンポーネントの確認
            print(f"  - Image Encoder: {type(sam.image_encoder).__name__}")
            print(f"  - Mask Decoder: {type(sam.mask_decoder).__name__}")
            print(f"  - Prompt Encoder: {type(sam.prompt_encoder).__name__}")
            
            # パラメータ数の確認
            total_params = sum(p.numel() for p in sam.parameters())
            print(f"  - 総パラメータ数: {total_params:,}")
            
            return sam
            
        except Exception as e:
            print(f"✗ SAMチェックポイント読み込み失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_model_with_sam(self, sam_model):
        """SAM統合モデルのテスト"""
        print("\n=== SAM統合モデルテスト ===")
        
        try:
            # SAM統合モデルの設定
            model_config = LisaGemmaConfig(
                gemma_model_id=config.GEMMA_MODEL_ID,
                sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,  # SAMチェックポイント指定
                gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),
                sam_prompt_embed_dim=getattr(config, 'SEG_PROJECTION_DIM', 256)
            )
            
            print("LISA-Gemmaモデル（SAM統合版）の初期化中...")
            model = LisaGemmaForCausalLM(model_config)
            print("✓ SAM統合モデル初期化成功")
            
            # パラメータ情報の確認
            if hasattr(model, 'get_trainable_parameters_info'):
                param_info = model.get_trainable_parameters_info()
                print(f"  総パラメータ数: {param_info['total_parameters']:,}")
                print(f"  訓練可能パラメータ数: {param_info['trainable_parameters']:,}")
                print(f"  訓練可能割合: {param_info['trainable_percentage']:.2f}%")
            else:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  総パラメータ数: {total_params:,}")
                print(f"  訓練可能パラメータ数: {trainable_params:,}")
                print(f"  訓練可能割合: {trainable_params/total_params*100:.2f}%")
            
            # 各コンポーネントの存在確認
            if hasattr(model, 'sam_image_encoder') and model.sam_image_encoder is not None:
                print("✓ SAM Image Encoder 統合確認")
            else:
                print("✗ SAM Image Encoder 未統合")
                
            if hasattr(model, 'sam_prompt_encoder') and model.sam_prompt_encoder is not None:
                print("✓ SAM Prompt Encoder 統合確認")
            else:
                print("⚠️ SAM Prompt Encoder 未統合（仕様によっては正常）")
                
            if hasattr(model, 'sam_mask_decoder') and model.sam_mask_decoder is not None:
                print("✓ SAM Mask Decoder 統合確認")
            else:
                print("✗ SAM Mask Decoder 未統合")
                
            if hasattr(model, 'mlp_projector') and model.mlp_projector is not None:
                print("✓ MLP Projector 存在確認")
            else:
                print("✗ MLP Projector 未作成")
            
            # デバイスに移動
            model.to(self.device)
            print(f"✓ モデルを{self.device}に移動")
            
            return model
            
        except Exception as e:
            print(f"✗ SAM統合モデルテスト失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_sam_forward_pass(self, model):
        """SAM統合フォワードパステスト"""
        print("\n=== SAM統合フォワードパステスト ===")
        
        if model is None:
            print("✗ モデルが初期化されていません")
            return
        
        try:
            # テスト用画像の作成（カラフルなテスト画像）
            test_image = Image.new('RGB', (512, 512))
            # グラデーションを追加
            pixels = []
            for y in range(512):
                for x in range(512):
                    r = int(255 * x / 512)
                    g = int(255 * y / 512)
                    b = 128
                    pixels.append((r, g, b))
            test_image.putdata(pixels)
            
            test_prompt = "この画像の右下の青い部分をセグメント化してください。 <SEG>"
            
            print("SAM統合フォワードパス実行中...")
            with torch.no_grad():
                if hasattr(model, 'forward') and hasattr(model, 'sam_image_encoder'):
                    # カスタムフォワードメソッドがある場合
                    outputs = model(
                        image=test_image,
                        text_prompt=test_prompt,
                        generate_mask=True  # SAMマスク生成を有効化
                    )
                else:
                    # 標準的なフォワードメソッドを使用
                    from transformers import AutoProcessor
                    processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)
                    
                    # 簡単なテスト入力を作成
                    inputs = processor(
                        images=test_image, 
                        text=test_prompt, 
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
            
            print("✓ SAM統合フォワードパス成功")
            print(f"  出力キー: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"    {key}: {type(value)}")
            
            # マスク予測の確認
            if "predicted_masks" in outputs and outputs["predicted_masks"] is not None:
                masks = outputs["predicted_masks"]
                print(f"✓ マスク予測成功: {masks.shape}")
                
                # マスクの統計情報
                print(f"    マスク値範囲: [{masks.min():.4f}, {masks.max():.4f}]")
                print(f"    マスク平均値: {masks.mean():.4f}")
                
                # IoU予測の確認
                if "iou_predictions" in outputs and outputs["iou_predictions"] is not None:
                    iou = outputs["iou_predictions"]
                    print(f"✓ IoU予測: {iou.shape}, 値: {iou.mean():.4f}")
            else:
                print("⚠️ マスク予測なし（<SEG>トークンが検出されなかった可能性）")
                
        except Exception as e:
            print(f"✗ SAM統合フォワードパス失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def test_sam_mask_generation(self, model):
        """SAMマスク生成の詳細テスト"""
        print("\n=== SAMマスク生成詳細テスト ===")
        
        if model is None:
            print("✗ モデルが初期化されていません")
            return
        
        try:
            # 複数の<SEG>トークンを含むプロンプトでテスト
            test_image = Image.new('RGB', (256, 256), color='red')
            test_prompts = [
                "この画像の赤い部分をセグメント化してください。 <SEG>",
                "画像全体をセグメント化してください。 <SEG> さらに詳細に <SEG>",
                "<SEG> 最初にセグメント化し、その後解析してください。"
            ]
            
            for i, prompt in enumerate(test_prompts):
                print(f"\nテストケース {i+1}: {prompt[:30]}...")
                
                with torch.no_grad():
                    outputs = model(
                        image=test_image,
                        text_prompt=prompt,
                        generate_mask=True
                    )
                
                if "predicted_masks" in outputs and outputs["predicted_masks"] is not None:
                    masks = outputs["predicted_masks"]
                    print(f"  ✓ マスク生成成功: {masks.shape}")
                else:
                    print(f"  ✗ マスク生成失敗")
                    
        except Exception as e:
            print(f"✗ SAMマスク生成テスト失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def test_sam_loss_computation(self, model):
        """SAM統合損失計算テスト"""
        print("\n=== SAM統合損失計算テスト ===")
        
        if model is None:
            print("✗ モデルが初期化されていません")
            return
        
        try:
            loss_fn = CompositeLoss(
                ce_loss_weight=1.0,
                dice_loss_weight=0.5,
                bce_loss_weight=2.0
            )
            print("✓ 複合損失関数初期化成功")
            
            # テスト用画像とプロンプト
            test_image = Image.new('RGB', (128, 128), color='blue')
            test_prompt = "青い領域をセグメント化 <SEG>"
            
            # モデル推論
            with torch.no_grad():
                outputs = model(
                    image=test_image,
                    text_prompt=test_prompt,
                    generate_mask=True
                )
            
            # ダミーの正解データ作成（出力と同じ次元に調整）
            logits_shape = outputs["gemma_logits"].shape
            batch = {
                "labels": torch.randint(0, 32000, (logits_shape[0], logits_shape[1])).to(self.device),  # ダミーラベル
                "ground_truth_mask": torch.ones(1, 1, 64, 64).to(self.device)  # ダミーマスク
            }
            
            # 損失計算
            losses = loss_fn(outputs, batch)
            print("✓ SAM統合損失計算成功")
            
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.item():.4f}")
                else:
                    print(f"    {key}: {value}")
                    
        except Exception as e:
            print(f"✗ SAM統合損失計算テスト失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def run_all_tests(self):
        """全SAM統合テストの実行"""
        print("LISA-Gemma3 SAMチェックポイント統合テスト開始")
        print("=" * 60)
        
        # SAMチェックポイント読み込みテスト
        sam_model = self.test_sam_checkpoint_loading()
        
        # SAM統合モデルテスト
        integrated_model = self.test_model_with_sam(sam_model)
        
        # SAM統合フォワードパステスト
        self.test_sam_forward_pass(integrated_model)
        
        # SAMマスク生成詳細テスト
        self.test_sam_mask_generation(integrated_model)
        
        # SAM統合損失計算テスト
        self.test_sam_loss_computation(integrated_model)
        
        print("\n" + "=" * 60)
        print("SAM統合テスト完了")
        
        # 統合成功度の評価
        if integrated_model is not None:
            if (integrated_model.sam_image_encoder is not None and 
                integrated_model.sam_prompt_encoder is not None and
                integrated_model.sam_mask_decoder is not None):
                print("🎉 SAMチェックポイント統合：完全成功")
                print("✅ 次のステップ：LoRA設定とDeepSpeed分散学習")
            else:
                print("⚠️ SAMチェックポイント統合：部分的成功")
                print("🔧 修正が必要：SAMコンポーネントの統合確認")
        else:
            print("❌ SAMチェックポイント統合：失敗")
            print("🔧 修正が必要：モデル初期化またはSAMチェックポイント")


if __name__ == "__main__":
    tester = SAMIntegrationTester()
    tester.run_all_tests() 