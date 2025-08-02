"""
SAM2.1 + Qwen2.5-VL統合モデル - Option A実装
Qwen ViT主体アーキテクチャ: Qwen2.5-VLのビジョンエンコーダを主に使用し、SAM2.1はマスクデコーダのみ活用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Dict, Any, List, Tuple
from PIL import Image
import warnings
import os
import math

# Transformers imports for Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# LoRA/QLoRA imports
from model.lora_config import LoRAConfigManager, create_lora_manager

# SAM2.1 imports
try:
    # プロジェクト内sam2フォルダをPATHに追加
    import sys
    project_root = os.path.join(os.path.dirname(__file__), '..')
    sam2_path = os.path.join(project_root, 'sam2')
    if os.path.exists(sam2_path) and sam2_path not in sys.path:
        sys.path.insert(0, sam2_path)
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    from sam2.modeling.sam.prompt_encoder import PromptEncoder
    SAM2_AVAILABLE = True
    print("✅ SAM2.1インポート成功")
except ImportError as e:
    print(f"⚠️ SAM2.1インポートエラー: {e}")
    warnings.warn("SAM2.1 not available. Please install: git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .")
    SAM2_AVAILABLE = False

# Type aliases
ImageType = Union[torch.Tensor, np.ndarray, Image.Image]


class SAMQwenModelOptionA(nn.Module):
    """
    SAM2.1 + Qwen2.5-VL統合モデル（Option A: Qwen ViT主体）
    
    アーキテクチャ:
    - Qwen2.5-VL: 画像エンコーダ（ViT）+ 言語モデル
    - SAM2.1: マスクデコーダのみ使用
    - 統合: Qwen ViTの特徴をSAMマスクデコーダに入力
    
    特徴:
    - Qwenの強力な視覚エンコーダを活用
    - SAMの高精度マスク生成機能を統合
    - 将来のFoodLMM拡張に最適化
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 sam_checkpoint: str = "sam2_hiera_large.pt",
                 sam_config: str = "sam2_hiera_l.yaml",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "auto",
                 lora_config: Optional[LoRAConfigManager] = None):
        """
        統合モデルの初期化
        
        Args:
            model_name: Qwen2.5-VLモデル名
            sam_checkpoint: SAM2.1チェックポイントファイル
            sam_config: SAM2.1設定ファイル
            torch_dtype: モデル精度
            device_map: デバイス配置戦略
            lora_config: LoRA/QLoRA設定マネージャー
        """
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2.1が必要です。インストールしてください: git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .")
        
        self.model_name = model_name
        self.sam_checkpoint = sam_checkpoint
        self.sam_config = sam_config
        self.torch_dtype = torch_dtype
        self.lora_config = lora_config
        
        print(f"🚀 SAM2.1 + Qwen2.5-VL統合モデル（Option A）初期化開始...")
        
        # Qwen2.5-VLの初期化
        self._init_qwen()
        
        # SAM2.1マスクデコーダの初期化  
        self._init_sam_decoder()
        
        # 特徴変換層の初期化
        self._init_feature_projector()
        
        # 統一トークン空間の初期化
        self._init_unified_token_space()
        
        # LoRA設定の適用
        if self.lora_config is not None:
            self._apply_lora()
        
        print(f"✅ 統合モデル初期化完了（Option A: Qwen ViT主体）")
        print(f"  - Qwen: {model_name}")
        print(f"  - SAM Decoder: {sam_checkpoint}")
        print(f"  - デバイス: {self.device}")
        print(f"  - アーキテクチャ: Qwen ViT → SAM Mask Decoder")
        if self.lora_config is not None:
            print(f"  - LoRA/QLoRA: 有効")
    
    def _init_qwen(self):
        """Qwen2.5-VLの初期化（最新API）"""
        print(f"🧠 Qwen2.5-VL初期化中: {self.model_name}")
        
        # モデル読み込み
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # プロセッサー読み込み
        self.qwen_processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Qwenの設定情報を取得
        self.qwen_config = self.qwen_model.config
        self.vision_config = self.qwen_config.vision_config
        self.text_config = self.qwen_config.text_config
        
        # ビジョンエンコーダの特徴次元を取得
        self.vision_hidden_size = self.vision_config.hidden_size  # 通常1280
        self.patch_size = self.vision_config.patch_size  # 通常14
        
        print(f"✅ Qwen2.5-VL初期化完了")
        print(f"  - Vision Hidden Size: {self.vision_hidden_size}")
        print(f"  - Patch Size: {self.patch_size}")
    
    def _init_sam_decoder(self):
        """SAM2.1マスクデコーダの初期化"""
        print(f"🎯 SAM2.1マスクデコーダ初期化中...")
        
        try:
            # SAM2.1の完全なモデルを一時的にロード（デコーダを取得するため）
            config_name = "sam2_hiera_l.yaml"
            project_root = os.path.join(os.path.dirname(__file__), '..')
            ckpt_path = os.path.join(project_root, 'checkpoints', self.sam_checkpoint)
            
            # SAM2.1モデル構築
            sam_model = build_sam2(config_name, ckpt_path, device=self.device)
            
            # マスクデコーダとプロンプトエンコーダを抽出
            self.sam_mask_decoder = sam_model.sam_mask_decoder
            self.sam_prompt_encoder = sam_model.sam_prompt_encoder
            
            # SAMの設定情報を保存
            self.sam_embed_dim = sam_model.hidden_dim  # 通常256
            self.sam_image_size = sam_model.image_size  # 通常1024
            
            # 元のSAMモデルは削除（メモリ節約）
            del sam_model
            
            print(f"✅ SAM2.1マスクデコーダ初期化完了")
            print(f"  - SAM Embed Dim: {self.sam_embed_dim}")
            print(f"  - SAM Image Size: {self.sam_image_size}")
            
        except Exception as e:
            print(f"❌ SAM2.1マスクデコーダ初期化エラー: {e}")
            raise RuntimeError(f"Failed to initialize SAM2.1 mask decoder: {str(e)}")
    
    def _init_feature_projector(self):
        """Qwen視覚特徴をSAMマスクデコーダに適合させる変換層"""
        print(f"🔄 特徴変換層初期化中...")
        
        # Qwen ViT (vision_hidden_size) → SAM Decoder (sam_embed_dim)の変換
        # Option A仕様書に従い、段階的な投影を実装
        self.vision_to_sam_projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.sam_embed_dim)
        )
        
        # 位置エンコーディング（SAMマスクデコーダ用）
        # SAMは通常64x64の特徴マップを期待
        self.sam_pe_layer = nn.Embedding(64 * 64, self.sam_embed_dim)
        
        print(f"✅ 特徴変換層初期化完了")
        print(f"  - 変換: {self.vision_hidden_size} → {self.sam_embed_dim}")
    
    def _init_unified_token_space(self):
        """LISA/GSVAスタイルの統一トークン空間初期化"""
        print("🔧 統一トークン空間初期化中...")
        
        # 特殊トークンの追加
        self.seg_token = "<SEG>"
        self.rej_token = "[REJ]"
        
        # トークナイザーに特殊トークンを追加
        special_tokens = [self.seg_token, self.rej_token]
        num_added_tokens = self.qwen_processor.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        
        if num_added_tokens > 0:
            # 埋め込み層のリサイズ
            self.qwen_model.resize_token_embeddings(len(self.qwen_processor.tokenizer))
            print(f"✅ 特殊トークン追加: {special_tokens}")
        
        # トークンIDを保存
        self.seg_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids(self.seg_token)
        self.rej_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids(self.rej_token)
        
        # <SEG>トークンから「なし」トークンから」マスククエリへの投影層
        # LISA準拠: LLM hidden → SAM mask query
        llm_hidden_size = self.text_config.hidden_size
        self.seg_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size // 2),
            nn.LayerNorm(llm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(llm_hidden_size // 2, llm_hidden_size // 4),
            nn.LayerNorm(llm_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(llm_hidden_size // 4, self.sam_embed_dim)
        )
        
        print("✅ 統一トークン空間初期化完了")
    
    def _apply_lora(self):
        """LoRA/QLoRAを適用"""
        print("🔧 LoRA/QLoRA設定を適用中...")
        
        if self.lora_config is None:
            return
            
        # QwenモデルにLoRA適用
        try:
            from peft import get_peft_model
            qwen_lora_config = self.lora_config.get_qwen_lora_config()
            self.qwen_model = get_peft_model(self.qwen_model, qwen_lora_config)
            print("✅ QwenモデルにLoRAを適用")
        except Exception as e:
            print(f"⚠️ QwenへのLoRA適用をスキップ: {e}")
        
        # 必要に応じてSAMデコーダにもLoRA適用可能
        # （Option Aでは初期段階では不要）
        
        # LoRA設定サマリーを表示
        self.lora_config.print_lora_summary()
    
    @property
    def device(self):
        """モデルのデバイスを取得"""
        return next(self.parameters()).device
    
    def extract_qwen_vision_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Qwenのビジョンエンコーダから特徴を抽出
        
        Args:
            inputs: プロセッサーからの入力辞書（pixel_values, input_ids等を含む）
            
        Returns:
            vision_features: ビジョン特徴 [B, num_patches, hidden_size]
        """
        # Qwenモデルの内部forwardを使って視覚特徴を取得
        # まず、モデルのembedding層を通す
        model_inputs = self.qwen_model.prepare_inputs_for_generation(**inputs)
        
        # Qwenモデルの内部処理を利用
        # inputs_embedsを取得（視覚情報が埋め込まれている）
        with torch.no_grad():
            # モデルのforward処理の最初の部分を実行
            outputs = self.qwen_model.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 視覚特徴を含む隠れ状態を取得
            # 注：実際の視覚特徴の抽出方法は、モデルの内部実装に依存
            # ここでは簡略化のため、最初の隠れ状態を使用
            hidden_states = outputs.hidden_states[0]  # [B, seq_len, hidden_size]
            
            # 画像トークンの位置を特定（簡略化：最初のN個のトークンを画像と仮定）
            # 実際には、画像トークンの位置を正確に特定する必要がある
            num_image_tokens = 256  # 仮の値（16x16パッチ）
            vision_features = hidden_states[:, :num_image_tokens, :]
            
        return vision_features
    
    def prepare_sam_features(self, vision_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Qwenのビジョン特徴をSAMマスクデコーダ用に変換
        
        Args:
            vision_features: Qwenビジョン特徴 [B, num_patches, vision_hidden_size]
            
        Returns:
            sam_features: SAM用特徴マップ [B, sam_embed_dim, H, W]
            sam_pe: 位置エンコーディング [B, sam_embed_dim, H, W]
        """
        B, N, C = vision_features.shape
        
        # パッチ数から空間サイズを計算
        H = W = int(math.sqrt(N))
        assert H * W == N, f"パッチ数 {N} が正方形グリッドではありません"
        
        # SAM用に次元変換
        sam_features_flat = self.vision_to_sam_projector(vision_features)  # [B, N, sam_embed_dim]
        
        # 空間形状にリシェイプ
        sam_features = sam_features_flat.permute(0, 2, 1).reshape(B, self.sam_embed_dim, H, W)
        
        # 必要に応じてSAMの期待する解像度にリサイズ（通常64x64）
        if H != 64:
            sam_features = F.interpolate(sam_features, size=(64, 64), mode='bilinear', align_corners=False)
        
        # 位置エンコーディングを生成
        positions = torch.arange(64 * 64, device=self.device)
        sam_pe = self.sam_pe_layer(positions).reshape(1, 64, 64, self.sam_embed_dim)
        sam_pe = sam_pe.permute(0, 3, 1, 2).expand(B, -1, -1, -1)
        
        return sam_features, sam_pe
    
    def forward_with_segmentation(self, 
                                 images: Union[torch.Tensor, List[Image.Image]], 
                                 messages: List[Dict],
                                 max_new_tokens: int = 512,
                                 return_dict: bool = True) -> Dict[str, Any]:
        """
        統合推論: テキスト生成とセグメンテーション
        Option A: Qwen ViTで一度だけ画像をエンコードし、LLMとSAMデコーダの両方で使用
        
        Args:
            images: 入力画像
            messages: チャット形式のメッセージリスト
            max_new_tokens: 生成する最大トークン数
            return_dict: 辞書形式で返すか
            
        Returns:
            results: 生成テキストとマスクを含む結果
        """
        # 1. 画像とテキストの前処理
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # process_vision_infoを使って画像を処理
        image_inputs, video_inputs = process_vision_info(messages)
        
        # プロセッサーで入力を準備
        inputs = self.qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # 2. Qwenのビジョンエンコーダで特徴抽出（一度だけ）
        with torch.no_grad():
            # pixel_valuesがあるか確認
            if inputs.get("pixel_values") is not None:
                # Qwenモデルから視覚特徴を抽出
                vision_features = self.extract_qwen_vision_features(inputs)
                
                # SAM用に変換
                sam_features, sam_pe = self.prepare_sam_features(vision_features)
                
                # 元の画像サイズを保存
                orig_h, orig_w = inputs["pixel_values"].shape[-2:]
            else:
                vision_features = None
                sam_features = None
                sam_pe = None
                orig_h = orig_w = None
        
        # 3. テキスト生成（Qwenの通常のgenerate）
        generated_ids = self.qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.qwen_processor.tokenizer.pad_token_id,
            eos_token_id=self.qwen_processor.tokenizer.eos_token_id,
        )
        
        # 生成されたテキストをデコード
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        generated_text = self.qwen_processor.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        # 4. <SEG>トークンがあればマスク生成
        masks = None
        has_seg_token = self.seg_token in generated_text
        
        if has_seg_token and sam_features is not None:
            # <SEG>トークンの位置を見つける
            seg_positions = (generated_ids[0] == self.seg_token_id).nonzero(as_tuple=True)[0]
            
            if len(seg_positions) > 0:
                # 最後の<SEG>トークンの隠れ状態を取得
                # （簡略化のため、ここではダミーのマスククエリを使用）
                # 実際の実装では、生成時の隠れ状態を保存して使用する必要がある
                mask_query = torch.randn(1, 1, self.sam_embed_dim, device=self.device)
                
                # SAMマスクデコーダでマスク生成
                with torch.no_grad():
                    # デコーダ入力を準備
                    sparse_embeddings = mask_query  # [1, 1, sam_embed_dim]
                    dense_embeddings = torch.zeros(1, self.sam_embed_dim, 64, 64, device=self.device)
                    
                    # マスクデコーダを実行
                    low_res_masks, iou_predictions = self.sam_mask_decoder(
                        image_embeddings=sam_features,
                        image_pe=sam_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                    )
                    
                    # 最も高いIoUスコアのマスクを選択
                    best_mask_idx = iou_predictions.argmax(dim=1)
                    mask = low_res_masks[0, best_mask_idx[0]]  # [256, 256]
                    
                    # 元の画像サイズにアップサンプル
                    if orig_h is not None and orig_w is not None:
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=(orig_h, orig_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    
                    # 二値化
                    masks = (mask > 0).float()
        
        # 5. 結果をまとめて返す
        results = {
            "generated_text": generated_text,
            "mask": masks,
            "has_seg_token": has_seg_token,
            "raw_generated_ids": generated_ids,
        }
        
        return results
    
    def forward_train(self, 
                     images: Union[torch.Tensor, List[Image.Image]], 
                     messages: List[Dict],
                     target_masks: Optional[torch.Tensor] = None,
                     max_new_tokens: int = 128) -> Dict[str, Any]:
        """
        訓練用forward（損失計算含む）
        """
        # TODO: 訓練用の実装（隠れ状態の保存と<SEG>トークンからのマスク生成）
        raise NotImplementedError("訓練用forwardは次のステップで実装予定")
    
    def configure_for_training(self, freeze_sam_decoder: bool = True, freeze_qwen_vision: bool = False):
        """
        訓練用のパラメータ設定
        
        Args:
            freeze_sam_decoder: SAMデコーダを凍結するか
            freeze_qwen_vision: Qwenのビジョンエンコーダを凍結するか
        """
        # SAMデコーダの凍結設定
        if freeze_sam_decoder:
            for param in self.sam_mask_decoder.parameters():
                param.requires_grad = False
            for param in self.sam_prompt_encoder.parameters():
                param.requires_grad = False
            print("✅ SAMデコーダを凍結")
        
        # Qwenビジョンエンコーダの凍結設定
        if freeze_qwen_vision:
            for param in self.qwen_model.model.visual.parameters():
                param.requires_grad = False
            print("✅ Qwenビジョンエンコーダを凍結")
        
        # 学習可能パラメータ数を表示
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 学習可能パラメータ: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


def create_sam_qwen_model_option_a(model_config: Dict[str, Any]) -> SAMQwenModelOptionA:
    """
    統合モデルを作成するヘルパー関数（Option A版）
    
    Args:
        model_config: モデル設定辞書
        
    Returns:
        初期化されたSAMQwenModelOptionA
    """
    return SAMQwenModelOptionA(**model_config)