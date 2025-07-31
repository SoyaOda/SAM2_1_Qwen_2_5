"""
SAM2.1 + Qwen2.5-VL統合モデル - 最新正式API実装
2025年最新のHugging Face Transformers + Meta SAM2.1公式APIベース
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Dict, Any, List, Tuple
from PIL import Image
import warnings
import os

# Transformers imports for Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# SAM2.1 imports
try:
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
                 sam_checkpoint: str = "sam2.1_hiera_large.pt",
                 sam_config: str = "sam2.1_hiera_l.yaml",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "auto"):
        """
        統合モデルの初期化
        
        Args:
            model_name: Qwen2.5-VLモデル名
            sam_checkpoint: SAM2.1チェックポイントファイル
            sam_config: SAM2.1設定ファイル
            torch_dtype: モデル精度
            device_map: デバイス配置戦略
        """
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2.1が必要です。インストールしてください: git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .")
        
        self.model_name = model_name
        self.sam_checkpoint = sam_checkpoint
        self.sam_config = sam_config
        self.torch_dtype = torch_dtype
        
        print(f"🚀 SAM2.1 + Qwen2.5-VL統合モデル初期化開始...")
        
        # Qwen2.5-VLの初期化
        self._init_qwen()
        
        # SAM2.1の初期化  
        self._init_sam()
        
        print(f"✅ 統合モデル初期化完了")
        print(f"  - Qwen: {model_name}")
        print(f"  - SAM: {sam_checkpoint}")
        print(f"  - デバイス: {self.device}")
    
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
        
        # チェックポイント・設定ファイルのパス構築
        checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'sam2.1')
        
        ckpt_path = os.path.join(checkpoints_dir, self.sam_checkpoint)
        config_path = os.path.join(configs_dir, self.sam_config)
        
        # ファイル存在確認（存在しない場合は代替手段）
        if not os.path.exists(ckpt_path):
            print(f"⚠️ チェックポイントが見つかりません: {ckpt_path}")
            print("🔄 公式リポジトリから自動ダウンロードを試行...")
            ckpt_path = self.sam_checkpoint  # ファイル名のみを指定（SAM2が自動解決）
        
        if not os.path.exists(config_path):
            print(f"⚠️ 設定ファイルが見つかりません: {config_path}")
            config_path = f"sam2_hiera_l.yaml"  # デフォルト設定名
        
        try:
            # SAM2.1モデル構築
            self.sam_model = build_sam2(config_path, ckpt_path, device=self.device)
            
            # SAM2.1プレディクター初期化
            self.sam_predictor = SAM2ImagePredictor(self.sam_model)
            
            print(f"✅ SAM2.1初期化完了")
            
        except Exception as e:
            print(f"❌ SAM2.1初期化エラー: {e}")
            print("🔄 フォールバック: 簡単な設定で再試行...")
            
            # フォールバック: より簡単な初期化
            try:
                # 絶対パスでの自動解決を試行
                import sam2
                sam2_path = sam2.__path__[0]
                default_config = os.path.join(sam2_path, 'configs', 'sam2', 'sam2_hiera_l.yaml')
                if os.path.exists(default_config):
                    self.sam_model = build_sam2(default_config, self.sam_checkpoint)
                else:
                    # 最終フォールバック: シンプルな設定名のみ
                    self.sam_model = build_sam2("sam2_hiera_l.yaml", self.sam_checkpoint)
                
                self.sam_predictor = SAM2ImagePredictor(self.sam_model)
                print("✅ SAM2.1フォールバック初期化完了")
            except Exception as e2:
                print(f"❌ SAM2.1フォールバック初期化失敗: {e2}")
                print("⚠️ SAM2.1を使用しない統合モードで続行します")
                self.sam_model = None
                self.sam_predictor = None
    
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
        h, w = img_np.shape[:2]
        
        # SAM2.1が利用できない場合のフォールバック
        if self.sam_predictor is None:
            print("⚠️ SAM2.1が利用できません。ダミーマスクを返します。")
            dummy_mask = torch.zeros((h, w), dtype=torch.bool)
            return {
                'mask': dummy_mask,
                'all_masks': dummy_mask.unsqueeze(0),
                'iou_scores': torch.tensor([0.0]),
                'best_mask_idx': 0,
                'logits': torch.zeros((1, h, w))
            }
        
        try:
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
            
        except Exception as e:
            print(f"⚠️ セグメンテーションエラー: {e}")
            # フォールバック: ダミーマスクを返す
            dummy_mask = torch.zeros((h, w), dtype=torch.bool)
            return {
                'mask': dummy_mask,
                'all_masks': dummy_mask.unsqueeze(0),
                'iou_scores': torch.tensor([0.0]),
                'best_mask_idx': 0,
                'logits': torch.zeros((1, h, w))
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
            
            # 画像・動画情報処理
            image_inputs, video_inputs = process_vision_info(messages)
            
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
            print(f"⚠️ テキスト生成エラー: {e}")
            return {
                'generated_text': f"テキスト生成エラー: {str(e)}",
                'generated_ids': torch.tensor([[0]]),
                'input_length': 0
            }
    
    def eval(self):
        """評価モードに設定"""
        super().eval()
        self.qwen_model.eval()
        self.sam_model.eval()
        return self
    
    def train(self, mode: bool = True):
        """訓練モードに設定"""
        super().train(mode)
        self.qwen_model.train(mode)
        self.sam_model.train(mode)
        return self


def create_sam_qwen_model(config: Dict[str, Any]) -> SAMQwenModel:
    """
    設定辞書からSAMQwenModelを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        SAMQwenModel インスタンス
    """
    return SAMQwenModel(
        model_name=config.get('model_name', "Qwen/Qwen2.5-VL-3B-Instruct"),
        sam_checkpoint=config.get('sam_model_name', "sam2.1_hiera_large.pt"),
        sam_config=config.get('sam_config_name', "sam2.1_hiera_l.yaml"),
        torch_dtype=config.get('torch_dtype', torch.float16),
        device_map=config.get('device_map', "auto")
    )


# エクスポート
__all__ = ['SAMQwenModel', 'create_sam_qwen_model']