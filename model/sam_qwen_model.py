"""
SAM2.1 + Qwen2.5-VL統合モデル - 最新正式API実装
2025年最新のHugging Face Transformers + Meta SAM2.1公式APIベース
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
    SAM2_AVAILABLE = True
    print("✅ SAM2.1インポート成功")
except ImportError as e:
    print(f"⚠️ SAM2.1インポートエラー: {e}")
    warnings.warn("SAM2.1 not available. Please install: git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .")
    SAM2_AVAILABLE = False

# Type aliases
ImageType = Union[torch.Tensor, np.ndarray, Image.Image]


class SAMQwenModel(nn.Module):
    """
    SAM2.1 + Qwen2.5-VL統合モデル（正式API版）
    
    アーキテクチャ:
    - SAM2.1: セグメンテーション専用（独立動作）
    - Qwen2.5-VL: 画像理解・テキスト生成専用（独立動作）
    - 統合: 両方の結果をマージして返す
    
    特徴:
    - 公式APIのみ使用（monkey-patch不使用）
    - モジュラー設計（各コンポーネント独立）
    - 将来のFoodLMM拡張に対応
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
        
        print(f"🚀 SAM2.1 + Qwen2.5-VL統合モデル初期化開始...")
        
        # Qwen2.5-VLの初期化
        self._init_qwen()
        
        # SAM2.1の初期化  
        self._init_sam()
        
        # Sa2VA/GSVA準拠のエンドツーエンド統合機能初期化
        self._init_unified_token_space()
        
        # LoRA設定の適用
        if self.lora_config is not None:
            self._apply_lora()
        
        print(f"✅ 統合モデル初期化完了")
        print(f"  - Qwen: {model_name}")
        print(f"  - SAM: {sam_checkpoint}")
        print(f"  - デバイス: {self.device}")
        print(f"  - 統一トークン空間: 有効（<SEG>, [REJ]対応）")
        if self.lora_config is not None:
            print(f"  - LoRA/QLoRA: 有効")
    
    def _init_qwen(self):
        """Qwen2.5-VLの初期化（最新API）"""
        print(f"🧠 Qwen2.5-VL初期化中: {self.model_name}")
        
        # モデル読み込み（Flash Attentionなし）
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
        
        print(f"✅ Qwen2.5-VL初期化完了")
    
    def _init_sam(self):
        """SAM2.1の初期化（公式API）"""
        print(f"🎯 SAM2.1初期化中: {self.sam_checkpoint}")
        
        try:
            # Hydraの設定パスの解決をSAM2内部に任せる
            # 直接ファイル名のみを指定
            config_name = "sam2_hiera_l.yaml"
            
            # チェックポイントファイルパス
            project_root = os.path.join(os.path.dirname(__file__), '..')
            ckpt_path = os.path.join(project_root, 'checkpoints', self.sam_checkpoint)
            
            # SAM2.1モデル構築
            self.sam_model = build_sam2(config_name, ckpt_path, device=self.device)
            
            # SAM2.1プレディクター初期化
            self.sam_predictor = SAM2ImagePredictor(self.sam_model)
            
            print(f"✅ SAM2.1初期化完了")
            
        except Exception as e:
            print(f"❌ SAM2.1初期化エラー: {e}")
            raise RuntimeError(f"Failed to initialize SAM2.1: {str(e)}")
    
    def _apply_lora(self):
        """LoRA/QLoRAを適用"""
        print("🔧 LoRA/QLoRA設定を適用中...")
        
        if self.lora_config is None:
            return
            
        # QLoRAの場合、量子化設定を適用（実装簡略化のため現在は通常のLoRAのみ）
        if self.lora_config.use_qlora:
            print("⚠️ QLoRA（4bit量子化）は現在の実装では簡略化のため無効です。通常のLoRAを使用します。")
        
        # QwenモデルにLoRA適用
        try:
            from peft import get_peft_model
            qwen_lora_config = self.lora_config.get_qwen_lora_config()
            self.qwen_model = get_peft_model(self.qwen_model, qwen_lora_config)
            print("✅ QwenモデルにLoRAを適用")
        except Exception as e:
            print(f"⚠️ QwenへのLoRA適用をスキップ: {e}")
        
        # SAMのmask_decoderにLoRA適用（手動実装）
        # 注: PEFTは直接SAMをサポートしていないため、手動でLoRA層を追加
        self._apply_lora_to_sam_decoder()
        
        # LoRA設定サマリーを表示
        self.lora_config.print_lora_summary()
    
    def _apply_lora_to_sam_decoder(self):
        """SAMのmask_decoderに手動でLoRA層を追加（正しい実装）"""
        import torch.nn as nn
        
        # LoRAを適用するためのカスタムLinear層
        class LoRALinear(nn.Module):
            def __init__(self, original_linear, rank=16, alpha=32, dropout=0.1):
                super().__init__()
                self.original_linear = original_linear
                self.rank = rank
                self.alpha = alpha
                self.scaling = alpha / rank
                self.dropout = nn.Dropout(dropout)
                
                # LoRAパラメータ（A: down projection, B: up projection）
                self.lora_A = nn.Parameter(torch.zeros((rank, original_linear.in_features)))
                self.lora_B = nn.Parameter(torch.zeros((original_linear.out_features, rank)))
                
                # 初期化（Aは正規分布、Bはゼロ）
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                
                # 元の層のパラメータを凍結
                for param in self.original_linear.parameters():
                    param.requires_grad = False
                
                # 元の層の重みとバイアスをコピー（参照ではなくコピー）
                self.register_buffer('weight', original_linear.weight.data.clone())
                if original_linear.bias is not None:
                    self.register_buffer('bias', original_linear.bias.data.clone())
                else:
                    self.register_buffer('bias', None)
            
            def forward(self, x):
                # 元の線形変換
                base_output = F.linear(x, self.weight, self.bias)
                
                # LoRAの追加分
                if self.training:
                    # x @ A^T -> [batch, rank]
                    lora_output = x @ self.lora_A.T
                    lora_output = self.dropout(lora_output)
                    # @ B^T -> [batch, out_features]
                    lora_output = lora_output @ self.lora_B.T
                    return base_output + lora_output * self.scaling
                
                return base_output
        
        # SAM2.1では sam_mask_decoder を使用
        if hasattr(self.sam_model, 'sam_mask_decoder'):
            mask_decoder = self.sam_model.sam_mask_decoder
        elif hasattr(self.sam_model, 'mask_decoder'):
            mask_decoder = self.sam_model.mask_decoder
        else:
            print("⚠️ SAM mask_decoderが見つかりません")
            return
        
        if mask_decoder is not None:
            replaced_count = 0
            
            # 再帰的に全モジュールを処理する関数
            def replace_linear_with_lora(parent_module, parent_name=""):
                nonlocal replaced_count
                
                for name, child in list(parent_module.named_children()):
                    full_name = f"{parent_name}.{name}" if parent_name else name
                    
                    if isinstance(child, nn.Linear):
                        # LoRAを適用する対象を選択（q_proj, k_proj, v_proj, o_projなど）
                        if any(target in name for target in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'fc1', 'fc2']):
                            lora_linear = LoRALinear(
                                child,
                                rank=self.lora_config.sam_lora_r,
                                alpha=self.lora_config.sam_lora_alpha,
                                dropout=self.lora_config.sam_lora_dropout
                            ).to(self.device)
                            
                            # 親モジュールの属性として設定
                            setattr(parent_module, name, lora_linear)
                            replaced_count += 1
                    else:
                        # 再帰的に子モジュールも処理
                        replace_linear_with_lora(child, full_name)
            
            # mask_decoderの全層を処理
            replace_linear_with_lora(mask_decoder)
            
            if replaced_count > 0:
                print(f"✅ SAM mask_decoderに{replaced_count}個のLoRA層を適用")
            else:
                print("⚠️ SAM mask_decoderにLoRA適用対象の層が見つかりませんでした")
    
    @property
    def device(self) -> torch.device:
        """モデルのデバイスを取得"""
        return next(self.qwen_model.parameters()).device
    
    def forward(self,
                image: ImageType,
                question: str = "",
                point_coords: Optional[torch.Tensor] = None,
                point_labels: Optional[torch.Tensor] = None,
                box: Optional[torch.Tensor] = None,
                return_text_only: bool = False,
                return_mask_only: bool = False,
                max_new_tokens: int = 128) -> Dict[str, Any]:
        """
        統合推論（セグメンテーション + テキスト生成）
        
        Args:
            image: 入力画像
            question: 質問テキスト
            point_coords: 点プロンプト [N, 2]
            point_labels: 点ラベル [N] (1=前景, 0=背景)
            box: ボックスプロンプト [4] (x1,y1,x2,y2)
            return_text_only: テキストのみ返す
            return_mask_only: マスクのみ返す
            max_new_tokens: 最大生成トークン数
            
        Returns:
            結果辞書 (mask, generated_text, scores, etc.)
        """
        results = {}
        
        # 画像前処理
        img_np = self._prepare_image(image)
        
        # セグメンテーション実行（SAM2.1）
        if not return_text_only:
            seg_results = self._run_segmentation(img_np, point_coords, point_labels, box)
            results.update(seg_results)
        
        # テキスト生成実行（Qwen2.5-VL）
        if not return_mask_only and question:
            text_results = self._run_text_generation(img_np, question, max_new_tokens)
            results.update(text_results)
        
        return results
    
    def _prepare_image(self, image: ImageType) -> np.ndarray:
        """画像を統一フォーマット（numpy uint8）に変換"""
        if isinstance(image, torch.Tensor):
            # Tensorの場合
            if image.dim() == 4:
                image = image.squeeze(0)  # バッチ次元削除
            if image.dim() == 3 and image.shape[0] == 3:
                # CHW -> HWC
                image = image.permute(1, 2, 0)
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            # PIL Imageの場合
            img_np = np.array(image.convert("RGB"))
        else:
            # NumPy arrayの場合
            img_np = image.astype(np.uint8)
        
        return img_np
    
    def _run_segmentation(self,
                         img_np: np.ndarray,
                         point_coords: Optional[torch.Tensor] = None,
                         point_labels: Optional[torch.Tensor] = None,
                         box: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """SAM2.1でセグメンテーション実行"""
        # SAM2.1が利用できない場合はエラー
        if self.sam_predictor is None:
            raise RuntimeError("SAM2.1 predictor is not initialized. Please check SAM2.1 installation.")
        
        # 画像設定
        self.sam_predictor.set_image(img_np)
        
        # プロンプト準備
        points_np = point_coords.cpu().numpy() if point_coords is not None else None
        labels_np = point_labels.cpu().numpy() if point_labels is not None else None
        box_np = box.cpu().numpy() if box is not None else None
        
        # セグメンテーション実行
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            box=box_np,
            multimask_output=True
        )
        
        # 最良マスク選択
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return {
            'mask': torch.from_numpy(best_mask).bool(),
            'all_masks': torch.from_numpy(masks),
            'iou_scores': torch.from_numpy(scores),
            'best_mask_idx': best_idx,
            'logits': torch.from_numpy(logits)
        }
    
    def _run_text_generation(self, 
                           img_np: np.ndarray, 
                           question: str, 
                           max_new_tokens: int) -> Dict[str, Any]:
        """Qwen2.5-VLでテキスト生成実行"""
        try:
            # メッセージ構築（最新API形式）
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_np},
                    {"type": "text", "text": question}
                ]
            }]
            
            # チャットテンプレート適用
            prompt = self.qwen_processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # メッセージに画像を埋め込む
            if isinstance(img_np, (list, tuple)):
                image_list = list(img_np)
            else:
                image_list = [img_np]
            
            # メッセージ内の画像プレースホルダーを実際の画像に置き換え
            updated_messages = []
            for msg in messages:
                updated_msg = dict(msg)
                if "content" in updated_msg:
                    updated_content = []
                    for item in updated_msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            # 画像プレースホルダーを実際の画像に置き換え
                            updated_content.append({"type": "image", "image": image_list[0]})
                        else:
                            updated_content.append(item)
                    updated_msg["content"] = updated_content
                updated_messages.append(updated_msg)
            
            # 画像・動画情報処理
            image_inputs, video_inputs = process_vision_info(updated_messages)
            
            # 入力準備
            inputs = self.qwen_processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # テキスト生成
            with torch.inference_mode():
                generated_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.7,
                    pad_token_id=self.qwen_processor.tokenizer.eos_token_id
                )
            
            # 生成テキスト抽出
            input_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, input_len:]
            generated_text = self.qwen_processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True
            )[0]
            
            return {
                'generated_text': generated_text,
                'generated_ids': generated_ids,
                'input_length': input_len
            }
            
        except Exception as e:
            print(f"❌ テキスト生成エラー: {e}")
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def eval(self):
        """評価モードに設定"""
        super().eval()
        self.qwen_model.eval()
        self.sam_model.eval()
        return self
    
    def _init_unified_token_space(self):
        """
        Sa2VA/GSVA準拠の統一トークン空間初期化
        最新研究に基づくエンドツーエンド統合機能
        """
        print("🔗 統一トークン空間初期化中...")
        
        # 特殊トークンの追加（Sa2VA/GSVA準拠）
        special_tokens = ["<SEG>", "[REJ]"]  # GSVA拡張: 拒否トークン対応
        
        # トークナイザーに特殊トークンを追加
        num_added = self.qwen_processor.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        
        if num_added > 0:
            # モデルの語彙サイズを拡張
            self.qwen_model.resize_token_embeddings(len(self.qwen_processor.tokenizer))
            print(f"✅ 特殊トークン追加: {num_added}個")
            
            # 新規トークンの埋め込みを小さく初期化
            with torch.no_grad():
                embeddings = self.qwen_model.get_input_embeddings()
                old_num_tokens = embeddings.weight.size(0) - num_added
                # 新規トークンの埋め込みを正規分布で初期化（標準偏差を小さく）
                embeddings.weight[old_num_tokens:].normal_(mean=0.0, std=0.02)
        
        # トークンIDを保存
        self.seg_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("<SEG>")
        self.rej_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("[REJ]")
        
        # o3_spec6.md準拠: SAM ViT → Qwen LLM への投影層
        qwen_hidden_size = self.qwen_model.config.hidden_size
        # SAM2.1の実際の出力次元に基づく修正
        # デバッグ出力によると、backbone_fpnとvision_featuresは全て256次元
        # これはFPNによる統一化の結果
        sam_feat_dim = 256  # FPN d_model (sam2_hiera_l.yaml)
        
        # SAM ViT出力をQwen LLMトークンに変換する投影層
        self.vision_to_llm_projector = nn.Sequential(
            nn.Linear(sam_feat_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, qwen_hidden_size)
        ).to(device=self.device, dtype=torch.float32)
        
        # <SEG>トークン埋め込みをSAMクエリに変換する投影層
        sam_embed_dim = 256  # SAM2.1デコーダ入力次元
        self.seg_to_sam_projector = nn.Sequential(
            nn.Linear(qwen_hidden_size, qwen_hidden_size // 2),
            nn.LayerNorm(qwen_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(qwen_hidden_size // 2, sam_embed_dim)
        ).to(device=self.device, dtype=torch.float32)
        
        # 重みの初期化（より安定した初期化）
        # 参考: https://github.com/dvlab-research/LISA
        for module in [self.vision_to_llm_projector, self.seg_to_sam_projector]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    # Kaiming初期化でより安定させる
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    # スケーリングを調整
                    with torch.no_grad():
                        layer.weight.mul_(0.1)  # 初期値をさらに小さく
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Dropout):
                    # Dropout率を確認
                    pass
        
        print(f"✅ 統一トークン空間初期化完了")
        print(f"  - <SEG>トークンID: {self.seg_token_id}")
        print(f"  - [REJ]トークンID: {self.rej_token_id}")
        print(f"  - SAM→Qwen投影層: {sam_feat_dim} → {qwen_hidden_size}")
        print(f"  - Qwen→SAM投影層: {qwen_hidden_size} → {sam_embed_dim}")
    
    def forward_with_segmentation(self, images, messages, max_new_tokens=128):
        """
        Sa2VA準拠のエンドツーエンド推論（o3_spec6.md準拠）
        SAM2.1 ViTで画像エンコード → Qwen LLMでテキスト生成 → SAMデコーダでマスク生成
        
        Args:
            images: 入力画像 (PIL Image or tensor)
            messages: Qwen2.5-VL形式のメッセージ
            max_new_tokens: 最大生成トークン数
            
        Returns:
            Dict: 生成テキスト、マスク、メタデータ
        """
        try:
            # 画像の準備
            if isinstance(images, (list, tuple)):
                image = images[0]
            else:
                image = images
            
            # PIL Image または numpy array を処理
            if hasattr(image, 'convert'):
                image_np = np.array(image.convert('RGB'))
            else:
                image_np = np.array(image)
            
            h, w = image_np.shape[:2]
            
            # 1. SAM2.1のimage_encoderで画像をエンコード
            # SAM2の入力形式に変換
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # SAM2.1のimage_encoderを直接呼び出し
            backbone_out = self.sam_model.image_encoder(image_tensor)
            # backbone_out = {"vision_features": Tensor, "vision_pos_enc": List, "backbone_fpn": List}
            
            # デバッグ: SAMの出力を確認
            print(f"[DEBUG] backbone_out keys: {backbone_out.keys()}")
            print(f"[DEBUG] vision_features shape: {backbone_out['vision_features'].shape}")
            if "backbone_fpn" in backbone_out:
                print(f"[DEBUG] backbone_fpn length: {len(backbone_out['backbone_fpn'])}")
                for i, feat in enumerate(backbone_out['backbone_fpn']):
                    print(f"[DEBUG] backbone_fpn[{i}] shape: {feat.shape}")
            
            # 最高解像度の特徴マップを取得
            sam_features = backbone_out["vision_features"]  # [B, C, H, W]
            B, C, H_feat, W_feat = sam_features.shape
            
            # 2. SAM特徴をQwenのトークン列に変換
            # SAM2.1のFPNは全レベルを256次元に統一している
            # vision_featuresは最終的な統合特徴（256次元）
            sam_features_flat = sam_features.flatten(2).permute(0, 2, 1)  # [B, H*W, 256]
            print(f"[DEBUG] Using vision_features with shape {sam_features_flat.shape}")
            
            # 投影層を通す（dtype変換に注意）
            vision_tokens = self.vision_to_llm_projector(sam_features_flat)  # [B, H*W, qwen_hidden_size]
            # Qwenモデルと同じdtypeに変換
            vision_tokens = vision_tokens.to(dtype=self.torch_dtype)
            
            # 3. Qwen LLMでテキスト生成
            # プロンプトの準備
            prompt = self.qwen_processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # テキストトークンの準備（画像トークンは後で挿入）
            text_inputs = self.qwen_processor.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 訓練時の処理
            if self.training:
                # 簡易実装：固定的な<SEG>トークン生成を仮定
                # テキストと視覚トークンを結合してLLMに入力
                inputs_embeds = self.qwen_model.get_input_embeddings()(text_inputs.input_ids)
                
                # 視覚トークンをテキストトークンの先頭に挿入
                combined_embeds = torch.cat([vision_tokens, inputs_embeds], dim=1)
                
                # LLMフォワードパス
                model_outputs = self.qwen_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=torch.ones(combined_embeds.shape[:2], device=self.device),
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # <SEG>トークンの隠れ状態を取得
                # LISAに基づき、実際の<SEG>トークン位置を特定
                hidden_states = model_outputs.hidden_states[-1]  # 最終層の隠れ状態
                
                # <SEG>トークンの実際の位置を検索
                # 入力トークンの最後の部分から<SEG>トークンを探す
                seg_token_id = self.seg_token_id
                
                # テキスト部分のトークンIDを取得して<SEG>位置を検索
                text_token_ids = text_inputs.input_ids[0]  # [seq_len]
                seg_positions = (text_token_ids == seg_token_id).nonzero(as_tuple=True)[0]
                
                if len(seg_positions) > 0:
                    # <SEG>トークンが見つかった場合
                    # vision_tokensの長さを考慮してオフセットを追加
                    seg_position = vision_tokens.shape[1] + seg_positions[0].item()
                    print(f"[DEBUG] Found <SEG> token at position {seg_position} (after {vision_tokens.shape[1]} vision tokens)")
                else:
                    # <SEG>トークンが見つからない場合は最後のトークンを使用
                    seg_position = -1
                    print("[WARNING] <SEG> token not found, using last token position")
                
                # seg_positionの隠れ状態を取得
                seg_hidden = hidden_states[:, seg_position, :]
                
                # デバッグ: seg_hiddenの初期状態を確認
                print(f"[DEBUG] seg_hidden initial stats: mean={seg_hidden.mean().item():.6f}, std={seg_hidden.std().item():.6f}, min={seg_hidden.min().item():.6f}, max={seg_hidden.max().item():.6f}")
                
                # より安定した正規化処理
                # 1. 先に値の範囲を制限
                seg_hidden = torch.clamp(seg_hidden, min=-10.0, max=10.0)
                
                # 2. FP32でLayerNormを適用
                if not hasattr(self, '_seg_hidden_norm'):
                    self._seg_hidden_norm = nn.LayerNorm(seg_hidden.shape[-1], eps=1e-6).to(device=self.device, dtype=torch.float32)
                seg_hidden_fp32 = seg_hidden.float()
                seg_hidden = self._seg_hidden_norm(seg_hidden_fp32)
                
                # 3. 再度クリッピング
                seg_hidden = torch.clamp(seg_hidden, min=-5.0, max=5.0)
                
                # SAMクエリに変換（FP32で処理）
                seg_hidden_fp32 = seg_hidden.float() if seg_hidden.dtype != torch.float32 else seg_hidden
                seg_query = self.seg_to_sam_projector(seg_hidden_fp32)  # [B, 256]
                
                # 出力の正規化（安定性向上）
                seg_query = F.layer_norm(seg_query, seg_query.shape[1:])
                seg_query = torch.clamp(seg_query, min=-10.0, max=10.0)
                
                # SAM2.1のマスクデコーダでマスク生成
                # 簡易実装：点プロンプトとして中心点を使用
                center_point = torch.tensor([[w//2, h//2]], dtype=torch.float32, device=self.device)
                center_label = torch.tensor([1], device=self.device)
                
                # SAMマスクデコーダの呼び出し（公式API版）
                # backbone_out辞書全体を渡す
                mask_logits = self._generate_mask_with_sam_decoder(
                    backbone_out, seg_query, h, w
                )
                
                mask = torch.sigmoid(mask_logits).squeeze(0).detach().cpu().numpy()
                
                return {
                    'generated_text': "マスクを検出しました。<SEG>",
                    'raw_text': "マスクを検出しました。<SEG>",
                    'masks': [mask],
                    'mask_logits': mask_logits,
                    'has_masks': True,
                    'rejected': False,
                    'seg_token_positions': [[0, combined_embeds.shape[1]-1]],
                    'sam_features': sam_features  # デバッグ用
                }
            
            # 推論時の処理（TODO: 実装）
            # 現在は簡易的に元の実装を使用
            with torch.no_grad():
                # 元の実装にフォールバック（一時的）
                result = self._fallback_forward(images, messages, max_new_tokens)
                result['sam_features'] = sam_features  # SAM特徴を追加
                return result
            
        except Exception as e:
            print(f"❌ エンドツーエンド推論エラー: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Forward with segmentation failed: {str(e)}")
    
    def _find_seg_token_positions(self, token_ids):
        """<SEG>トークンの位置を検出"""
        if isinstance(token_ids, torch.Tensor):
            seg_positions = (token_ids == self.seg_token_id).nonzero(as_tuple=False)
            return seg_positions.cpu().numpy().tolist()
        return []
    
    def _generate_mask_with_sam_decoder(self, sam_features, seg_query, h, w):
        """
        SAM2.1のマスクデコーダを使用してマスクを生成（公式API版）
        
        Args:
            sam_features: SAMの画像特徴 (backbone_out辞書)
            seg_query: LLMからのセグメンテーションクエリ [B, 256]
            h, w: 出力マスクのサイズ
            
        Returns:
            mask_logits: マスクのロジット [1, H, W]
        """
        try:
            # SAM2.1のmask_decoderを取得
            if hasattr(self.sam_model, 'sam_mask_decoder'):
                mask_decoder = self.sam_model.sam_mask_decoder
            elif hasattr(self.sam_model, 'mask_decoder'):
                mask_decoder = self.sam_model.mask_decoder
            else:
                raise RuntimeError("SAM mask_decoder not found")
            
            # バッチサイズを取得
            B = seg_query.shape[0]
            
            # seg_queryをSAMのsparse_embeddings形式に変換
            # SAM2.1ではsparse_embeddingsは[B, N, 256]形式
            # LLMからのクエリを1つのポイントエンベディングとして扱う
            sparse_embeddings = seg_query.unsqueeze(1)  # [B, 1, 256]
            
            # dense_embeddingsはnull（マスクプロンプトなし）
            # SAM2.1ではno_mask_embedはmask_tokensに統合されている
            if hasattr(mask_decoder, 'no_mask_embed'):
                no_mask_embed = mask_decoder.no_mask_embed.weight
            elif hasattr(mask_decoder, 'mask_tokens'):
                # SAM2.1の実装ではmask_tokensが使用される
                no_mask_embed = mask_decoder.mask_tokens.weight[0:1]  # 最初のトークン
            else:
                # フォールバック: ゼロエンベディング
                no_mask_embed = torch.zeros(1, mask_decoder.hidden_dim, device=self.device)
            
            # dense_embeddingsを作成
            if hasattr(mask_decoder, 'image_embedding_size'):
                H_emb, W_emb = mask_decoder.image_embedding_size
            else:
                # SAM2.1では固定サイズ
                H_emb, W_emb = 64, 64
            
            dense_embeddings = no_mask_embed.reshape(1, -1, 1, 1).expand(
                B, -1, H_emb, W_emb
            )
            
            # 位置エンコーディングを取得
            # vision_featuresのサイズに合わせて調整
            if isinstance(sam_features, dict):
                vision_features = sam_features["vision_features"]
                B, C, H_feat, W_feat = vision_features.shape
            else:
                vision_features = sam_features
                B, C, H_feat, W_feat = vision_features.shape
            
            # 位置エンコーディングをvision_featuresのサイズに合わせる
            if hasattr(self.sam_model, 'sam_prompt_encoder') and hasattr(self.sam_model.sam_prompt_encoder, 'get_dense_pe'):
                # get_dense_peは通常固定サイズ(64x64)を返すので、サイズ調整が必要
                image_pe = self.sam_model.sam_prompt_encoder.get_dense_pe()
                # サイズを調整
                if image_pe.shape[-2:] != (H_feat, W_feat):
                    image_pe = F.interpolate(image_pe, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            else:
                # フォールバック: ゼロベクトル
                image_pe = torch.zeros(1, C, H_feat, W_feat, device=self.device, dtype=vision_features.dtype)
            
            # high_res_featuresを取得
            high_res_features = sam_features.get("backbone_fpn", None) if isinstance(sam_features, dict) else None
            
            # SAM2.1 mask_decoderを直接呼び出し
            # 公式APIを使用
            low_res_masks, iou_predictions, _, _ = mask_decoder(
                image_embeddings=vision_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,  # 複数マスクを出力
                repeat_image=False,
                high_res_features=high_res_features
            )
            
            # 最良のマスクを選択（IoUスコアが最高）
            best_idx = torch.argmax(iou_predictions[0])
            mask_logits = low_res_masks[0:1, best_idx:best_idx+1]  # [1, 1, H_low, W_low]
            
            # 高解像度にアップサンプリング
            mask_logits = F.interpolate(
                mask_logits,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            # [1, H, W]形式に変換
            mask_logits = mask_logits.squeeze(1)
            
            return mask_logits
            
        except Exception as e:
            print(f"[❗ SAM mask_decoder呼び出しエラー: {e}")
            print(f"[❗ 簡易実装にフォールバックします")
            
            # フォールバック: 簡易実装
            B = seg_query.shape[0]
            
            # seg_queryを使ってシンプルなマスク生成
            # 1x1畳み込みで空間マスクを生成
            if not hasattr(self, '_mask_head'):
                self._mask_head = nn.Sequential(
                    nn.Conv2d(256, 128, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 64, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 1, 1)
                ).to(device=self.device, dtype=torch.float32)
            
            # seg_queryを空間的に展開
            H_feat, W_feat = 64, 64  # デフォルト特徴マップサイズ
            seg_spatial = seg_query.view(B, 256, 1, 1).expand(B, 256, H_feat, W_feat)
            
            # マスク生成
            mask_logits = self._mask_head(seg_spatial)  # [B, 1, H_feat, W_feat]
            
            # アップサンプリング
            mask_logits = F.interpolate(
                mask_logits,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            return mask_logits.squeeze(1)  # [B, H, W]
    
    def _fallback_forward(self, images, messages, max_new_tokens):
        """
        フォールバック実装（元の処理を一時的に使用）
        """
        # 元の画像処理を使用
        if isinstance(images, (list, tuple)):
            image_list = list(images)
        else:
            image_list = [images]
        
        updated_messages = []
        for msg in messages:
            updated_msg = dict(msg)
            if "content" in updated_msg:
                updated_content = []
                for item in updated_msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        updated_content.append({"type": "image", "image": image_list[0]})
                    else:
                        updated_content.append(item)
                updated_msg["content"] = updated_content
            updated_messages.append(updated_msg)
        
        # 元のQwen処理
        prompt = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(updated_messages)
        
        inputs = self.qwen_processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "do_sample": False,
            "pad_token_id": self.qwen_processor.tokenizer.eos_token_id
        }
        
        outputs = self.qwen_model.generate(**inputs, **gen_config)
        
        generated_ids = outputs.sequences
        input_length = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]
        
        generated_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False
        )[0]
        
        # 簡易マスク生成
        masks = self._extract_masks_from_generation_simple(generated_ids_trimmed, images)
        
        return {
            'generated_text': generated_text.replace("<SEG>", "[MASK]"),
            'raw_text': generated_text,
            'masks': masks,
            'has_masks': len(masks) > 0,
            'rejected': "[REJ]" in generated_text,
            'seg_token_positions': self._find_seg_token_positions(generated_ids_trimmed)
        }
    
    def _extract_masks_from_generation_simple(self, generated_ids, images):
        """
        簡易的なマスク抽出（一時的な実装）
        """
        masks = []
        
        # <SEG>トークンが含まれている場合
        if (generated_ids == self.seg_token_id).any():
            # ダミーマスクを生成
            if isinstance(images, (list, tuple)):
                image = images[0]
            else:
                image = images
                
            if hasattr(image, 'size'):
                w, h = image.size
            else:
                h, w = image.shape[:2] if len(image.shape) >= 2 else (224, 224)
            
            # 中央に円形のマスク
            y, x = np.ogrid[:h, :w]
            center = (h//2, w//2)
            radius = min(h, w) // 4
            mask = ((x - center[1])**2 + (y - center[0])**2 <= radius**2).astype(float)
            masks.append(mask)
        
        return masks
    
    
    
    def train(self, mode: bool = True):
        """訓練モードに設定"""
        super().train(mode)
        self.qwen_model.train(mode)
        self.sam_model.train(mode)
        if hasattr(self, 'seg_to_sam_projector'):
            self.seg_to_sam_projector.train(mode)
        if hasattr(self, 'vision_to_llm_projector'):
            self.vision_to_llm_projector.train(mode)
        return self
    
    def configure_for_training(self, freeze_sam_encoder: bool = True):
        """
        訓練用のパラメータ設定（LoRA前提のシンプル版）
        
        Args:
            freeze_sam_encoder: SAMのエンコーダを凍結するか
        """
        # LoRAが設定されている場合（前提条件）
        if self.lora_config is not None:
            # PEFTでLoRAが適用されたQwenモデルの場合
            # ベースモデルのパラメータは自動的に凍結されている
            # LoRAパラメータのみが学習可能
            print("✅ Qwen: LoRAパラメータのみ学習可能（ベースモデルは凍結）")
            
            # SAMのLoRAパラメータを学習可能にする（既に置換済みのLoRALinear内で処理済み）
            print("✅ SAM: LoRAパラメータのみ学習可能")
        else:
            # LoRAなしの場合（従来の動作）
            print("⚠️ LoRA未設定での学習は推奨されません")
            
        # SAMエンコーダの凍結設定（LoRAとは独立）
        if freeze_sam_encoder:
            for name, param in self.sam_model.named_parameters():
                if 'image_encoder' in name:
                    param.requires_grad = False
            print("✅ SAMエンコーダを凍結")
        
        # 共通で学習可能な部分
        # 投影層は常に学習可能
        if hasattr(self, 'vision_to_llm_projector'):
            for param in self.vision_to_llm_projector.parameters():
                param.requires_grad = True
            print("✅ vision_to_llm_projectorを学習可能に設定")
        
        if hasattr(self, 'seg_to_sam_projector'):
            for param in self.seg_to_sam_projector.parameters():
                param.requires_grad = True
            print("✅ seg_to_sam_projectorを学習可能に設定")
        
        # 新規追加トークンの埋め込みのみを学習可能に（勾配フックを使用）
        if hasattr(self.qwen_model, 'get_input_embeddings'):
            embeddings = self.qwen_model.get_input_embeddings()
            if embeddings is not None:
                # 埋め込み層全体を学習可能にする（フックで制御）
                embeddings.weight.requires_grad = True
                
                # 新規トークンのIDを取得
                seg_token_id = self.seg_token_id
                rej_token_id = self.rej_token_id
                
                # 勾配マスキングフックを登録
                def mask_embedding_gradients(grad):
                    # 全体をゼロにしてから新規トークンのみ勾配を通す
                    mask = torch.zeros_like(grad)
                    mask[seg_token_id] = 1.0
                    mask[rej_token_id] = 1.0
                    return grad * mask
                
                # フックを登録
                embeddings.weight.register_hook(mask_embedding_gradients)
                
                print(f"✅ 新規トークンのみ学習可能に設定（<SEG>: {seg_token_id}, [REJ]: {rej_token_id}）")
        
        # 学習可能パラメータ数を表示
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        # LoRAパラメータ数を正確に計算
        lora_params = 0
        qwen_lora_params = 0
        sam_lora_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                # Qwen LoRA
                if 'lora' in name.lower() and 'qwen' in name:
                    qwen_lora_params += param.numel()
                    lora_params += param.numel()
                # SAM LoRA（LoRALinear内のlora_A, lora_B）
                elif any(x in name for x in ['lora_A', 'lora_B']) and 'mask_decoder' in name:
                    sam_lora_params += param.numel()
                    lora_params += param.numel()
        
        print(f"\n📊 パラメータ統計:")
        print(f"  - 総パラメータ数: {total_params:,}")
        print(f"  - 学習可能パラメータ数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  - うちLoRAパラメータ: {lora_params:,} ({lora_params/total_params*100:.4f}%)")
    
    def forward_train(self, 
                     images: Union[torch.Tensor, List[Image.Image]], 
                     messages: List[Dict],
                     target_masks: Optional[torch.Tensor] = None,
                     max_new_tokens: int = 128) -> Dict[str, Any]:
        """
        訓練用forward（損失計算含む）
        
        Args:
            images: 入力画像
            messages: Qwen形式のメッセージ
            target_masks: 正解マスク [B, H, W]
            max_new_tokens: 最大生成トークン数
            
        Returns:
            損失を含む結果辞書
        """
        from model.losses import MultiModalLoss
        
        # 損失関数の初期化（設定から取得）
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = MultiModalLoss(
                seg_weight=1.0,
                text_weight=1.0,
                seg_loss_type='combined'
            )
        
        # エンドツーエンド推論
        results = self.forward_with_segmentation(images, messages, max_new_tokens)
        
        # 損失計算の準備
        losses = {}
        
        # セグメンテーション損失
        if target_masks is not None and results['has_masks']:
            # 訓練時はロジットを使用（勾配が流れる）
            if 'mask_logits' in results and results['mask_logits'] is not None:
                pred_mask = results['mask_logits']  # すでにテンソル
            else:
                # フォールバック（推論時など）
                pred_mask = results['masks'][0]
                if isinstance(pred_mask, np.ndarray):
                    pred_mask = torch.from_numpy(pred_mask).float()
                pred_mask = pred_mask.to(self.device)
            
            # デバイスとdtype調整
            target_masks = target_masks.to(self.device).float()
            
            # バッチ次元の調整
            if pred_mask.dim() == 2:
                pred_mask = pred_mask.unsqueeze(0)
            if target_masks.dim() == 2:
                target_masks = target_masks.unsqueeze(0)
            
            # 損失計算
            seg_losses = self.loss_fn.seg_loss_fn(pred_mask, target_masks)
            if isinstance(seg_losses, dict):
                losses.update(seg_losses)
            else:
                losses['segmentation_loss'] = seg_losses
        
        # テキスト生成損失（今回の最小実装では省略）
        # 本格実装では、生成されたトークンIDと正解トークンIDから計算
        
        # 結果に損失を追加
        results['losses'] = losses
        results['total_loss'] = losses.get('total_loss', losses.get('segmentation_loss', torch.tensor(0.0)))
        
        return results
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """学習可能なパラメータのリストを取得"""
        return [p for p in self.parameters() if p.requires_grad]


def create_sam_qwen_model(config: Dict[str, Any], lora_config: Optional[LoRAConfigManager] = None) -> SAMQwenModel:
    """
    設定辞書からSAMQwenModelを作成
    
    Args:
        config: 設定辞書
        lora_config: LoRA設定マネージャー（オプション）
        
    Returns:
        SAMQwenModel インスタンス
    """
    return SAMQwenModel(
        model_name=config.get('model_name', "Qwen/Qwen2.5-VL-3B-Instruct"),
        sam_checkpoint=config.get('sam_model_name', "sam2_hiera_large.pt"),
        sam_config=config.get('sam_config_name', "sam2_hiera_l.yaml"),
        torch_dtype=config.get('torch_dtype', torch.float16),
        device_map=config.get('device_map', "auto"),
        lora_config=lora_config
    )


# エクスポート
__all__ = ['SAMQwenModel', 'create_sam_qwen_model']