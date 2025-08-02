"""
SAM2.1 + Qwen2.5-VL統合モデル - ハイブリッド実装
段階的なOption A移行のための中間実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Dict, Any, List, Tuple
from PIL import Image
import warnings
import os

# Transformers imports for Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# LoRA/QLoRA imports
from model.lora_config import LoRAConfigManager

# SAM2.1 imports
try:
    import sys
    project_root = os.path.join(os.path.dirname(__file__), '..')
    sam2_path = os.path.join(project_root, 'sam2')
    if os.path.exists(sam2_path) and sam2_path not in sys.path:
        sys.path.insert(0, sam2_path)
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("✅ SAM2.1インポート成功（ハイブリッド版）")
except ImportError as e:
    print(f"⚠️ SAM2.1インポートエラー: {e}")
    SAM2_AVAILABLE = False

# Type aliases
ImageType = Union[torch.Tensor, np.ndarray, Image.Image]


class SAMQwenModelHybrid(nn.Module):
    """
    SAM2.1 + Qwen2.5-VL統合モデル（ハイブリッド実装）
    
    アーキテクチャ:
    - SAM2.1: 画像エンコーダ + マスクデコーダ（セグメンテーション用）
    - Qwen2.5-VL: 言語生成専用（SAMの視覚特徴を言語空間に投影）
    - 統合: SAM視覚特徴 → Qwen言語モデル
    
    特徴:
    - Option Aへの段階的移行のための中間実装
    - SAMの高品質な視覚特徴を活用
    - Qwenの強力な言語生成能力を統合
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 sam_checkpoint: str = "sam2_hiera_large.pt",
                 sam_config: str = "sam2_hiera_l.yaml",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "auto",
                 lora_config: Optional[LoRAConfigManager] = None):
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2.1が必要です。")
        
        self.model_name = model_name
        self.sam_checkpoint = sam_checkpoint
        self.sam_config = sam_config
        self.torch_dtype = torch_dtype
        self.lora_config = lora_config
        
        print(f"🚀 SAM2.1 + Qwen2.5-VL統合モデル（ハイブリッド）初期化開始...")
        
        # Qwen2.5-VLの初期化
        self._init_qwen()
        
        # SAM2.1の初期化
        self._init_sam()
        
        # 特徴投影層の初期化
        self._init_projector()
        
        # 統一トークン空間の初期化
        self._init_unified_token_space()
        
        print(f"✅ 統合モデル初期化完了（ハイブリッド実装）")
        print(f"  - Qwen: {model_name}")
        print(f"  - SAM: {sam_checkpoint}")
        print(f"  - アーキテクチャ: SAM Vision → Projection → Qwen LLM")
    
    def _init_qwen(self):
        """Qwen2.5-VLの初期化"""
        print(f"🧠 Qwen2.5-VL初期化中: {self.model_name}")
        
        # モデル読み込み（言語生成部分のみ使用）
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
        
        # 設定情報を取得
        self.qwen_config = self.qwen_model.config
        self.text_config = self.qwen_config.text_config
        self.text_hidden_size = self.text_config.hidden_size
        
        print(f"✅ Qwen2.5-VL初期化完了")
        print(f"  - Text Hidden Size: {self.text_hidden_size}")
    
    def _init_sam(self):
        """SAM2.1の初期化"""
        print(f"🎯 SAM2.1初期化中: {self.sam_checkpoint}")
        
        try:
            config_name = "sam2_hiera_l.yaml"
            project_root = os.path.join(os.path.dirname(__file__), '..')
            ckpt_path = os.path.join(project_root, 'checkpoints', self.sam_checkpoint)
            
            # SAM2.1モデル構築
            self.sam_model = build_sam2(config_name, ckpt_path, device=self.device)
            
            # SAM2.1プレディクター初期化
            self.sam_predictor = SAM2ImagePredictor(self.sam_model)
            
            # SAMの設定情報
            self.sam_embed_dim = 256  # SAM2.1 FPN出力次元
            self.sam_image_size = 1024
            
            print(f"✅ SAM2.1初期化完了")
            
        except Exception as e:
            print(f"❌ SAM2.1初期化エラー: {e}")
            raise RuntimeError(f"Failed to initialize SAM2.1: {str(e)}")
    
    def _init_projector(self):
        """SAM特徴をQwen言語空間に投影する層"""
        print(f"🔄 特徴投影層初期化中...")
        
        # SAM特徴（256次元）をQwen埋め込み空間に投影
        # 複数の投影戦略を用意
        
        # 1. グローバルプーリング戦略（シンプル）
        self.global_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 256, H, W] → [B, 256, 1, 1]
            nn.Flatten(),  # [B, 256]
            nn.Linear(self.sam_embed_dim, self.text_hidden_size),
            nn.LayerNorm(self.text_hidden_size),
            nn.GELU()
        )
        
        # 2. パッチトークン戦略（詳細）
        # SAMの空間特徴を複数のトークンとして保持
        self.patch_projector = nn.Sequential(
            nn.Conv2d(self.sam_embed_dim, self.sam_embed_dim, kernel_size=2, stride=2),  # ダウンサンプル
            nn.GELU(),
            nn.Conv2d(self.sam_embed_dim, self.text_hidden_size, kernel_size=1),  # 次元変換
        )
        
        # デバイスに移動
        self.global_projector = self.global_projector.to(self.device)
        self.patch_projector = self.patch_projector.to(self.device)
        
        print(f"✅ 特徴投影層初期化完了")
    
    def _init_unified_token_space(self):
        """統一トークン空間の初期化"""
        print("🔧 統一トークン空間初期化中...")
        
        # 特殊トークンの追加
        self.seg_token = "<SEG>"
        self.rej_token = "[REJ]"
        self.img_token = "<IMG>"  # 画像特徴用の特殊トークン
        
        # トークナイザーに特殊トークンを追加
        special_tokens = [self.seg_token, self.rej_token, self.img_token]
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
        self.img_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids(self.img_token)
        
        # <SEG>トークンからマスククエリへの投影層
        self.seg_projector = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.text_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.text_hidden_size // 2, self.sam_embed_dim)
        ).to(self.device)
        
        print("✅ 統一トークン空間初期化完了")
    
    @property
    def device(self):
        """モデルのデバイスを取得"""
        return next(self.parameters()).device
    
    def encode_image_with_sam(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        SAM2.1で画像をエンコード
        
        Returns:
            - image_embed: 低解像度画像埋め込み [1, 256, 64, 64]
            - high_res_feats: 高解像度特徴のリスト
        """
        # PIL画像の場合はnumpy配列に変換
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 4:
                image_np = image_np[0]  # バッチ次元を除去
            image_np = image_np.transpose(1, 2, 0)  # CHW → HWC
        else:
            image_np = image
        
        # SAMで画像をセット（特徴抽出）
        self.sam_predictor.set_image(image_np)
        
        # 特徴を取得
        features = self.sam_predictor._features
        
        return features
    
    def project_sam_to_qwen(self, sam_features: Dict[str, torch.Tensor], use_patches: bool = False) -> torch.Tensor:
        """
        SAM特徴をQwen言語空間に投影
        
        Args:
            sam_features: SAMの画像特徴
            use_patches: パッチトークンを使用するか（False: グローバルプーリング）
            
        Returns:
            vision_embeddings: Qwen用の視覚埋め込み [B, num_tokens, hidden_size]
        """
        image_embed = sam_features['image_embed']  # [1, 256, 64, 64]
        
        if use_patches:
            # パッチトークン戦略
            projected = self.patch_projector(image_embed)  # [1, hidden_size, 32, 32]
            B, C, H, W = projected.shape
            vision_embeddings = projected.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [1, 1024, hidden_size]
        else:
            # グローバルプーリング戦略
            vision_embeddings = self.global_projector(image_embed).unsqueeze(1)  # [1, 1, hidden_size]
        
        return vision_embeddings
    
    def forward_with_segmentation(self, 
                                 images: Union[torch.Tensor, List[Image.Image]], 
                                 messages: List[Dict],
                                 max_new_tokens: int = 512,
                                 use_patch_tokens: bool = False) -> Dict[str, Any]:
        """
        統合推論: テキスト生成とセグメンテーション
        
        Args:
            images: 入力画像
            messages: チャット形式のメッセージリスト
            max_new_tokens: 生成する最大トークン数
            use_patch_tokens: パッチトークンを使用するか
            
        Returns:
            results: 生成テキストとマスクを含む結果
        """
        # 画像は1枚のみ対応（簡略化のため）
        if isinstance(images, list):
            image = images[0]
        else:
            image = images
        
        # 1. SAMで画像をエンコード
        sam_features = self.encode_image_with_sam(image)
        
        # 2. SAM特徴をQwen空間に投影
        vision_embeddings = self.project_sam_to_qwen(sam_features, use_patch_tokens)
        
        # 3. テキストプロンプトの準備（画像トークンを含む）
        # メッセージから画像部分を除去し、<IMG>トークンに置換
        modified_messages = []
        for msg in messages:
            if msg["role"] == "user":
                new_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        # テキストの先頭に<IMG>トークンを追加
                        new_content.append({
                            "type": "text", 
                            "text": f"{self.img_token} {item['text']}"
                        })
                modified_messages.append({
                    "role": msg["role"],
                    "content": new_content
                })
            else:
                modified_messages.append(msg)
        
        # 4. テキスト生成
        # チャットテンプレートを適用
        text = self.qwen_processor.apply_chat_template(
            modified_messages, tokenize=False, add_generation_prompt=True
        )
        
        # トークナイズ
        inputs = self.qwen_processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # デバッグ情報
        print(f"[DEBUG] Input text: {text}")
        print(f"[DEBUG] Input IDs shape: {inputs['input_ids'].shape}")
        print(f"[DEBUG] Vision embeddings shape: {vision_embeddings.shape}")
        
        # <IMG>トークンの位置を特定
        img_token_positions = (inputs["input_ids"] == self.img_token_id).nonzero(as_tuple=True)
        print(f"[DEBUG] IMG token positions: {img_token_positions}")
        
        # 入力埋め込みを取得し、<IMG>位置に視覚埋め込みを挿入
        with torch.no_grad():
            # 埋め込み層から入力埋め込みを取得
            embeddings = self.qwen_model.model.language_model.get_input_embeddings()
            inputs_embeds = embeddings(inputs["input_ids"])
            
            # <IMG>トークンの位置に視覚埋め込みを挿入
            if len(img_token_positions[0]) > 0:
                batch_idx = img_token_positions[0][0]
                seq_idx = img_token_positions[1][0]
                
                # 視覚埋め込みが複数トークンの場合の処理
                if vision_embeddings.shape[1] > 1:
                    # 複数トークンの場合は、<IMG>位置以降に挿入
                    # （簡略化のため、ここでは最初のトークンのみ使用）
                    inputs_embeds[batch_idx, seq_idx] = vision_embeddings[0, 0]
                else:
                    inputs_embeds[batch_idx, seq_idx] = vision_embeddings[0, 0]
            
            # inputs_embedsを使って生成
            outputs = self.qwen_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.qwen_processor.tokenizer.pad_token_id,
                eos_token_id=self.qwen_processor.tokenizer.eos_token_id,
            )
        
        # 生成されたテキストをデコード
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        print(f"[DEBUG] Generated IDs: {generated_ids}")
        print(f"[DEBUG] Generated IDs shape: {generated_ids.shape}")
        
        generated_text = self.qwen_processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        print(f"[DEBUG] Generated text: {generated_text}")
        
        # 5. <SEG>トークンがあればマスク生成
        masks = None
        has_seg_token = self.seg_token in generated_text
        
        if has_seg_token:
            # SAMでマスク生成（簡略化：センターポイントプロンプト）
            H, W = sam_features['image_embed'].shape[-2:]
            center_point = np.array([[W//2, H//2]])
            center_label = np.array([1])
            
            # マスク予測
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=True,
            )
            
            # 最高スコアのマスクを選択
            best_idx = scores.argmax()
            masks = masks[best_idx]
        
        # 6. 結果をまとめて返す
        results = {
            "generated_text": generated_text,
            "mask": masks,
            "has_seg_token": has_seg_token,
            "sam_features": sam_features,  # デバッグ用
        }
        
        return results


def create_sam_qwen_model_hybrid(model_config: Dict[str, Any]) -> SAMQwenModelHybrid:
    """統合モデルを作成するヘルパー関数（ハイブリッド版）"""
    return SAMQwenModelHybrid(**model_config)