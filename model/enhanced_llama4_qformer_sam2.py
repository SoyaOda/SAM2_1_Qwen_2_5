# model/enhanced_llama4_qformer_sam2.py
"""
Enhanced LISA-Llama4-Scout + Q-Former + SAM2 統合モデル

o3-modification20250727.mdに基づく改修版:
1. Enhanced Q-Former (BLIP-2準拠テキスト入力対応)
2. 視覚・言語特徴分離機構
3. LoRAエキスパート統合
4. マルチスケールセグメンテーションヘッド

改修内容:
- Q-Formerのテキスト入力対応
- LLM出力の視覚・言語分離
- マルチモーダル適応のLoRA
- 高精度マルチスケールマスク生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SAM2リポジトリパスも追加
sam2_repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sam2_repo')
if sam2_repo_path not in sys.path:
    sys.path.insert(0, sam2_repo_path)

import config_linux

# 改修版コンポーネント
from model.enhanced_qformer import get_enhanced_qformer_model
from model.modality_separation import VisualLanguageSeparator
from model.lora_experts import inject_lora_to_model, LoRAExpertMoE
from model.multiscale_decoder import MultiScaleSegmentationHead

# 既存コンポーネント
from model.sam2_integration import get_sam2_wrapper
from model.losses_qformer_sam2 import get_composite_loss_qformer_sam2
from model.moe_adapters import create_heterogeneous_moe_adapter

# LLMインポート
try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from transformers import Llama4ForCausalLM
    LLAMA4_MODEL_CLASS = Llama4ForCausalLM
    LLAMA4_AVAILABLE = True
    print("✅ Llama4ForCausalLM利用可能")
except ImportError:
    try:
        from transformers import AutoModelForCausalLM
        LLAMA4_MODEL_CLASS = AutoModelForCausalLM
        LLAMA4_AVAILABLE = True
        print("⚠️ Llama4ForCausalLM未対応、AutoModelForCausalLM使用")
    except ImportError as e:
        LLAMA4_AVAILABLE = False
        LLAMA4_MODEL_CLASS = None
        raise ImportError(f"❌ LLMモデルクラスimport失敗: {e}")


class EnhancedLlamaQFormerSAM2Config:
    """改修版統合モデル設定クラス"""
    
    def __init__(self):
        # Llama-4-Scout設定
        self.llama_model_id = config_linux.LLAMA_MODEL_ID
        self.llama_hidden_size = config_linux.LLAMA_HIDDEN_SIZE
        self.llama_vocab_size = 128256  # Llama-4デフォルト
        
        # GPU設定
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 4:
            self.device_map = "balanced_low_0"
            self.max_memory = {i: "70GB" for i in range(num_gpus)}
        else:
            self.device_map = "auto"
            self.max_memory = None
        
        self.torch_dtype = config_linux.TORCH_DTYPE
        self.attn_implementation = config_linux.ATTN_IMPLEMENTATION
        
        # Enhanced Q-Former設定
        # Webリサーチ結果に基づき、SAM2の実際の出力次元に対応
        self.qformer_config = {
            'num_queries': 32,              # BLIP-2準拠
            'hidden_size': 768,             # BLIP-2準拠
            'num_layers': 12,               # BLIP-2準拠
            'num_heads': 12,                # BLIP-2準拠
            'intermediate_size': 3072,      # BLIP-2準拠
            'dropout': 0.1,
            'sam_prompt_dim': 256,          # SAM2
            'encoder_hidden_size': 256,     # SAM2 FPN出力次元（Webリサーチ結果）
            'llm_hidden_size': self.llama_hidden_size,  # 5120
            'max_txt_len': 64,              # テキスト最大長
        }
        
        # SAM2設定
        self.sam2_model_id = config_linux.SAM2_HF_MODEL_ID
        self.sam2_config = {
            'model_id': self.sam2_model_id,
            'target_dtype': self.torch_dtype,
            'vos_optimized': True,
            'compile_model': True,
            'memory_pathways': 3,
            'mixed_precision': True,
        }
        
        # LoRA設定
        self.lora_config = {
            'rank': 16,
            'alpha': 16.0,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],  # Attention層
            'use_moe': True,        # MoE使用
            'num_experts': 2,       # RGB, Depth等
        }
        
        # マルチスケール設定
        self.multiscale_config = {
            'stages': [3, 6, 9, 12],  # ViT層インデックス
            'hidden_dim': 256,
        }
        
        # 訓練設定
        self.training_stage = 1  # Stage 1: インターフェース学習から開始


class EnhancedQFormerSegmentationBridge(nn.Module):
    """
    改修版統合モデル
    
    o3-modification20250727.mdの全改修項目を実装
    """
    
    def __init__(
        self,
        config: EnhancedLlamaQFormerSAM2Config,
        shared_llama_model: Optional[Any] = None,
        shared_llama_processor: Optional[Any] = None,
        enable_lora: bool = True,
        enable_multiscale: bool = True,
        training_stage: int = 1
    ):
        super().__init__()
        
        self.config = config
        self.training_stage = training_stage
        self.enable_lora = enable_lora
        self.enable_multiscale = enable_multiscale
        
        print("\n" + "="*80)
        print("🚀 Enhanced QFormerSegmentationBridge初期化開始")
        print("="*80)
        
        # 1. Llama-4初期化（共有インスタンス使用）
        self._setup_llama4(shared_llama_model, shared_llama_processor)
        
        # 2. Enhanced Q-Former初期化
        self._init_enhanced_qformer()
        
        # 3. 視覚・言語分離機構初期化
        self._init_modality_separator()
        
        # 4. SAM2初期化（マルチスケール対応）
        self._init_sam2_multiscale()
        
        # 5. LoRAエキスパート初期化
        if enable_lora:
            self._init_lora_experts()
        
        # 6. 損失関数初期化
        self._init_loss_function()
        
        # 7. デバイス配置の統一
        self._ensure_device_consistency()
        
        print("\n✅ Enhanced QFormerSegmentationBridge初期化完了")
        print(f"  - テキスト入力: 対応")
        print(f"  - 視覚・言語分離: 有効")
        print(f"  - LoRA適応: {'有効' if enable_lora else '無効'}")
        print(f"  - マルチスケール: {'有効' if enable_multiscale else '無効'}")
        print(f"  - 訓練ステージ: {training_stage}")
    
    def _setup_llama4(self, shared_model, shared_processor):
        """Llama-4セットアップ"""
        print("\n🧠 Llama-4セットアップ...")
        
        if shared_model is not None and shared_processor is not None:
            self.llama_model = shared_model
            self.llama_processor = shared_processor
            print("✅ 共有Llama-4インスタンス使用")
            
            # トークナイザー取得
            if hasattr(shared_processor, 'tokenizer'):
                self.llama_tokenizer = shared_processor.tokenizer
            else:
                self.llama_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.llama_model_id
                )
        else:
            raise RuntimeError("共有Llama-4インスタンスが必要です")
    
    def _init_enhanced_qformer(self):
        """Enhanced Q-Former初期化"""
        print("\n🔍 Enhanced Q-Former初期化...")
        
        # BLIP-2準拠の設定でQ-Former作成
        self.qformer = get_enhanced_qformer_model(self.config.qformer_config)
        
        # Llama-4接続用の追加プロジェクター
        # Q-Formerの出力(768) -> Llama-4入力(5120)への変換は
        # Enhanced Q-Former内で実装済み
        
        print(f"✅ Enhanced Q-Former初期化完了")
        print(f"  - テキスト入力対応: 有効")
        print(f"  - 学習モード: ITC/ITM/ITG対応")
    
    def _init_modality_separator(self):
        """視覚・言語分離機構初期化"""
        print("\n🔀 視覚・言語分離機構初期化...")
        
        # LLMの埋め込み層を取得
        llm_embed_layer = self.llama_model.get_input_embeddings()
        
        # 分離機構の作成
        self.modality_separator = VisualLanguageSeparator(
            llm_config={
                'hidden_size': self.config.llama_hidden_size,
                'vocab_size': self.config.llama_vocab_size,
                'vision_output_dim': 1024,  # SAM2互換
            },
            llm_tokenizer=self.llama_tokenizer,
            llm_embed_layer=llm_embed_layer
        )
        
        print("✅ 視覚・言語分離機構初期化完了")
    
    def _init_sam2_multiscale(self):
        """SAM2マルチスケール初期化"""
        print("\n🎯 SAM2マルチスケール初期化...")
        
        # 基本SAM2ラッパー取得
        sam_wrapper = get_sam2_wrapper(**self.config.sam2_config)
        
        if self.enable_multiscale:
            # マルチスケールヘッドでラップ
            self.segmentation_head = MultiScaleSegmentationHead(
                sam_wrapper=sam_wrapper,
                stages=self.config.multiscale_config['stages']
            )
            print("✅ マルチスケールセグメンテーションヘッド有効")
        else:
            # 通常のSAM2使用
            self.segmentation_head = sam_wrapper
            print("✅ 標準SAM2使用")
    
    def _init_lora_experts(self):
        """LoRAエキスパート初期化"""
        print("\n🔧 LoRAエキスパート初期化...")
        
        # SAM2エンコーダーにLoRA注入
        if hasattr(self.segmentation_head, 'image_encoder') and self.segmentation_head.image_encoder is not None:
            target_encoder = self.segmentation_head.image_encoder
            print(f"  - LoRA注入対象: MultiScaleSegmentationHead.image_encoder")
        elif hasattr(self.segmentation_head, 'sam_wrapper'):
            sam_wrapper = self.segmentation_head.sam_wrapper
            if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
                actual_model = sam_wrapper.predictor.model
                if hasattr(actual_model, 'image_encoder'):
                    target_encoder = actual_model.image_encoder
                    print(f"  - LoRA注入対象: SAM2.predictor.model.image_encoder")
                else:
                    raise RuntimeError(
                        "LoRA注入用のimage_encoderが見つかりません。"
                        "SAM2 predictor.modelにimage_encoderが存在しません。"
                    )
            else:
                raise RuntimeError(
                    "LoRA注入用のimage_encoderが見つかりません。"
                    "SAM2Wrapperにpredictor.modelが存在しません。"
                )
        else:
            raise RuntimeError(
                "LoRA注入用のエンコーダーが見つかりません。"
                "セグメンテーションヘッドにimage_encoderもsam_wrapperも存在しません。"
            )
        
        # LoRA注入
        inject_lora_to_model(
            model=target_encoder,
            target_modules=self.config.lora_config['target_modules'],
            rank=self.config.lora_config['rank'],
            alpha=self.config.lora_config['alpha'],
            dropout=self.config.lora_config['dropout'],
            use_moe=self.config.lora_config['use_moe'],
            num_experts=self.config.lora_config['num_experts']
        )
        
        print(f"✅ LoRAエキスパート注入完了")
        print(f"  - エキスパート数: {self.config.lora_config['num_experts']}")
        print(f"  - ランク: {self.config.lora_config['rank']}")
    
    def _init_loss_function(self):
        """損失関数初期化"""
        print("\n📊 損失関数初期化...")
        
        device = next(self.llama_model.parameters()).device
        
        self.loss_function = get_composite_loss_qformer_sam2(
            stage=self.training_stage,
            device=device
        )
        
        print(f"✅ 損失関数初期化完了 (Stage {self.training_stage})")
    
    def _ensure_device_consistency(self):
        """デバイス配置の統一"""
        print("\n🔧 デバイス配置確認中...")
        
        # Llama-4のデバイスを基準とする
        llama_device = next(self.llama_model.parameters()).device
        
        # Q-Formerを同じデバイスとデータ型に
        self.qformer = self.qformer.to(llama_device)
        # Llama-4のデータ型に合わせる（通常bfloat16）
        llama_dtype = next(self.llama_model.parameters()).dtype
        self.qformer = self.qformer.to(llama_dtype)
        
        # 分離機構も同じデバイスとデータ型に
        self.modality_separator.output_separator = \
            self.modality_separator.output_separator.to(llama_device).to(llama_dtype)
        
        print(f"✅ デバイス配置統一完了: {llama_device}")
    
    def forward(
        self,
        images: torch.Tensor,
        text_input: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = 'itg',  # Q-Former学習モード
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        統合forward処理
        
        Args:
            images: 入力画像 [B, 3, H, W]
            text_input: テキスト入力のリスト（オプション）
            labels: セグメンテーションラベル [B, H, W]
            mode: Q-Former学習モード ('itc', 'itm', 'itg')
            return_dict: 辞書形式で返すか
            
        Returns:
            Dict containing:
                - masks: セグメンテーションマスク
                - text_logits: テキスト生成ロジット
                - iou_scores: IoUスコア
                - visual_features: 視覚特徴
                - loss: 損失値（訓練時）
        """
        batch_size = images.size(0)
        device = images.device
        
        # 1. 画像特徴抽出（マルチスケール対応）
        if self.enable_multiscale and hasattr(self.segmentation_head, 'feature_extractor'):
            # マルチスケール特徴抽出
            image_encoder = self.segmentation_head.image_encoder
            multiscale_features = self.segmentation_head.feature_extractor(
                images, image_encoder
            )
            # メイン特徴（最終層またはfinal）
            # マルチスケール特徴から適切なものを選択
            # SAM2の最終出力は通常256次元
            if 'final' in multiscale_features:
                image_features = multiscale_features['final']
            else:
                # マルチスケール特徴をデバッグ出力
                print(f"⚠️ マルチスケール特徴一覧:")
                for name, feat in multiscale_features.items():
                    if isinstance(feat, torch.Tensor):
                        print(f"  - {name}: {feat.shape}")
                
                # フォールバック: 直接エンコード（通常のSAM2出力を取得）
                encoded_output = image_encoder(images)
                # SAM2は辞書形式で返すことがある
                if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
                    image_features = encoded_output['vision_features']
                else:
                    image_features = encoded_output
        else:
            # 通常の特徴抽出
            if hasattr(self.segmentation_head, 'image_encoder') and self.segmentation_head.image_encoder is not None:
                encoded_output = self.segmentation_head.image_encoder(images)
                # SAM2は辞書形式で返すことがある
                if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
                    image_features = encoded_output['vision_features']
                else:
                    image_features = encoded_output
            else:
                # マルチスケールが無効の場合、self.segmentation_head自体がSAM2Wrapper
                # SAM2Wrapperから直接image_encoderを取得
                if hasattr(self.segmentation_head, 'predictor') and hasattr(self.segmentation_head.predictor, 'model'):
                    actual_model = self.segmentation_head.predictor.model
                    if hasattr(actual_model, 'image_encoder'):
                        encoded_output = actual_model.image_encoder(images)
                        # SAM2は辞書形式で返すことがある
                        if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
                            image_features = encoded_output['vision_features']
                        else:
                            image_features = encoded_output
                    else:
                        raise RuntimeError(
                            "SAM2 predictor.modelにimage_encoderが見つかりません。"
                            "SAM2の構造を確認してください。"
                        )
                else:
                    raise RuntimeError(
                        "SAM2Wrapperにpredictor.modelが見つかりません。"
                        "SAM2の初期化を確認してください。"
                    )
            multiscale_features = None
        
        # 2. Q-Formerでクエリ生成（テキスト入力対応）
        qformer_outputs = self.qformer(
            image_feats=image_features,
            text_input=text_input,
            mode=mode,
            return_dict=True
        )
        
        # 3. Llama-4での統合処理準備
        # 視覚埋め込み（LLM用に射影済み）
        visual_embeds = qformer_outputs['llm_embeds']  # [B, 32, 5120]
        
        # テキストトークン処理
        if text_input is not None:
            text_tokens = self.llama_tokenizer(
                text_input,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
        else:
            text_tokens = None
        
        # 4. マルチモーダル入力の準備
        multimodal_input = self.modality_separator.prepare_input(
            visual_embeds=visual_embeds,
            text_tokens=text_tokens
        )
        
        # 5. Llama-4実行
        llm_outputs = self.llama_model(
            inputs_embeds=multimodal_input['inputs_embeds'],
            attention_mask=multimodal_input['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        
        # 6. 視覚・言語出力の分離
        # CausalLMモデルの場合、hidden_statesから最後の層を取得
        if hasattr(llm_outputs, 'hidden_states'):
            last_hidden_state = llm_outputs.hidden_states[-1]
        else:
            # フォールバック: logitsから隠れ状態を推定（非推奨）
            raise RuntimeError(
                "Llama-4の出力にhidden_statesがありません。"
                "output_hidden_states=Trueが正しく設定されているか確認してください。"
            )
        
        separated_outputs = self.modality_separator.separate_output(
            llm_outputs=last_hidden_state,
            modality_info=multimodal_input,
            output_hidden_states=True
        )
        
        # 7. セグメンテーションマスク生成
        # SAMプロンプトとしてQ-Formerのクエリ埋め込みを使用
        sam_prompts = qformer_outputs['sam_prompts']  # [B, 32, 256]
        
        if self.enable_multiscale:
            # マルチスケールマスク生成
            masks, iou_scores, _ = self.segmentation_head(
                images=images,
                visual_context=separated_outputs['visual_features'],
                multimask_output=True
            )
        else:
            # 通常のマスク生成
            # マルチスケールが無効の場合、self.segmentation_head自体がSAM2Wrapper
            if hasattr(self.segmentation_head, 'predict_with_prompts'):
                # SAM2の場合（self.segmentation_headが直接SAM2Wrapper）
                sam_wrapper = self.segmentation_head
                
                # 画像設定
                sam_wrapper.set_image(images)
                
                # プロンプトなしで予測（全体マスク）
                # SAM2Wrapper.predict_with_promptsの正しいシグネチャに合わせる
                outputs_dict = sam_wrapper.predict_with_prompts(
                    prompt_embeddings=sam_prompts,  # Q-Formerからのプロンプト埋め込みを使用
                    point_coords=None,
                    point_labels=None,
                    boxes=None,
                    multimask_output=True
                )
                
                # SAM2は[num_masks, H, W]形式で返すので、バッチ次元を追加
                masks = outputs_dict['masks']  # [num_masks, H, W]
                if masks.dim() == 3:
                    masks = masks.unsqueeze(0)  # [1, num_masks, H, W]
                iou_scores = outputs_dict['iou_predictions']  # [num_masks]
                if iou_scores.dim() == 1:
                    iou_scores = iou_scores.unsqueeze(0)  # [1, num_masks]
            else:
                # SAM1の場合（既存のコード）
                sparse_embeddings, dense_embeddings = \
                    self.segmentation_head.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None
                    )
                
                # マスクデコード
                masks, iou_scores = self.segmentation_head.mask_decoder(
                    image_embeddings=image_features,
                    image_pe=self.segmentation_head.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True
                )
        
        # 8. 結果の整理
        outputs = {
            'masks': masks,
            'text_logits': separated_outputs['text_logits'],
            'iou_scores': iou_scores,
            'visual_features': separated_outputs['visual_features'],
            'visual_pooled': separated_outputs['visual_pooled'],
            'qformer_outputs': qformer_outputs,
        }
        
        # 9. 損失計算（訓練時）
        if labels is not None:
            # CompositeLossQFormerSAM2の正しいシグネチャに合わせる
            # text_embedsはQ-Formerのテキスト出力を使用（存在する場合）
            text_embeds = None
            if 'text_outputs' in qformer_outputs:
                # Q-Formerのテキスト出力から埋め込みを取得
                text_embeds = qformer_outputs['text_outputs'].mean(dim=1)  # [B, hidden_size]
            
            loss_dict = self.loss_function(
                predicted_masks=masks,
                target_masks=labels,
                query_embeds=qformer_outputs.get('query_embeds'),
                text_embeds=text_embeds,
                sam_prompts=qformer_outputs.get('sam_prompts')
            )
            outputs['loss'] = loss_dict['total_loss']
            outputs['loss_dict'] = loss_dict
        
        # 常に辞書形式で返す（テストコードとの互換性のため）
        return outputs
    
    def generate_text(
        self,
        images: torch.Tensor,
        text_prompt: str,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        テキスト生成（推論用）
        """
        with torch.no_grad():
            # Forward処理
            outputs = self.forward(
                images=images,
                text_input=[text_prompt],
                mode='itg'
            )
            
            # テキスト生成
            text_logits = outputs['text_logits']
            
            # 生成処理（簡易版）
            generated_ids = []
            for _ in range(max_length):
                # 最後のトークンのロジット
                next_logits = text_logits[:, -1, :] / temperature
                next_probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(next_probs, 1)
                generated_ids.append(next_token)
                
                # 停止条件
                if next_token.item() == self.llama_tokenizer.eos_token_id:
                    break
            
            # デコード
            generated_ids = torch.cat(generated_ids, dim=1)
            generated_text = self.llama_tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            return generated_text