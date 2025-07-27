# model/sam2_integration.py
"""
SAM2統合モジュール: Segment Anything Model 2をQ-Formerと統合

SAM2の主な改善点:
- 6倍高速 + より高精度
- 画像・動画対応（統一アーキテクチャ）
- Transformer + streaming memoryアーキテクチャ
- リアルタイム処理対応（44 FPS）

参考:
- SAM2論文: https://arxiv.org/abs/2408.00714
- HuggingFace SAM2: https://huggingface.co/papers/2408.00714
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import sys
import os
import urllib.request
import hashlib
from pathlib import Path
import tempfile

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_linux

def download_sam2_checkpoint(model_name: str, save_path: str, download_url: str) -> bool:
    """
    SAM2チェックポイント自動ダウンロード (config_linux.py準拠)
    
    Args:
        model_name: モデル名 (例: "sam2-hiera-large")
        save_path: 保存先パス (config_linux.SAM2_CHECKPOINT_PATH)
        download_url: ダウンロードURL (config_linux.SAM2_DOWNLOAD_URL)
        
    Returns:
        bool: ダウンロード成功/既存ファイル確認済み
    """
    
    # 既存ファイル確認
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"  ✅ SAM2チェックポイント既存: {save_path}")
        print(f"    - ファイルサイズ: {file_size / (1024**3):.2f} GB")
        return True
    
    # ディレクトリ作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"  🔄 SAM2チェックポイントダウンロード中...")
    print(f"    - URL: {download_url}")
    print(f"    - 保存先: {save_path}")
    
    try:
        # プログレス表示付きダウンロード
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 100 == 0:  # 100ブロックごとに表示
                    print(f"    📥 ダウンロード進行: {percent}% ({block_num * block_size / (1024**2):.1f}MB)")
        
        urllib.request.urlretrieve(download_url, save_path, progress_hook)
        
        # ダウンロード完了確認
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"  ✅ SAM2チェックポイントダウンロード完了")
            print(f"    - ファイルサイズ: {file_size / (1024**3):.2f} GB")
            return True
        else:
            print(f"  ❌ ダウンロード失敗: ファイルが作成されませんでした")
            return False
            
    except Exception as e:
        print(f"  ❌ SAM2チェックポイントダウンロード失敗: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)  # 不完全ファイル削除
        return False

import subprocess
import sys

def check_python_and_sam2():
    """Python版本とSAM2インストール状況を確認"""
    python_version = sys.version_info
    
    if python_version < (3, 10):
        print(f"⚠️ Python {python_version.major}.{python_version.minor}では SAM2 は動作しません")
        print("Python 3.10以上が必要です")
        return False, "python_version_too_old"
    
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return True, "success"
    except ImportError as e:
        return False, str(e)

# Python 3.10環境でのSAM2インポート
sam2_available, sam2_status = check_python_and_sam2()

if sam2_available:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("✅ Meta公式SAM2利用可能 (Python 3.10+)")
else:
    SAM2_AVAILABLE = False
    print("❌ Meta公式SAM2が利用できません")
    if sam2_status == "python_version_too_old":
        print("解決策: pyenv local 3.10.12 でPython 3.10に切り替えてください")
    else:
        print("解決策: Python 3.10環境で pip install sam2 を実行してください")
        print(f"エラー詳細: {sam2_status}")


class SAM2Wrapper(nn.Module):
    """
    Meta公式SAM2ラッパー (HuggingFace Hub自動取得)
    
    重みの手動管理不要、Web自動取得:
    - facebook/sam2-hiera-large (推奨)
    - facebook/sam2-hiera-base
    - facebook/sam2-hiera-small
    - facebook/sam2-hiera-tiny
    """
    
    def __init__(
        self, 
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        target_dtype: torch.dtype = torch.bfloat16,
        debug_mode: bool = True,
        # 🔄 2025年SAM2最適化パラメータ (Web調査ベース)
        vos_optimized: bool = False,         # torch.compile VOS最適化
        compile_model: bool = False,         # モデル全体コンパイル
        memory_pathways: int = 1,            # メモリパス数（1-5, 3推奨）
        mixed_precision: bool = False,       # 混合精度学習
        **kwargs
    ):
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError(
                "Meta公式SAM2が利用できません。\n"
                "必須インストール: pip install segment-anything-2\n"
                "また、huggingface_hubも必要: pip install huggingface_hub"
            )
        
        # 🔄 2025年ベストプラクティス: 即座データ型統一設定
        self.device = "cuda"
        self.model_id = model_id
        self.debug_mode = debug_mode
        self.vos_optimized = vos_optimized
        self.compile_model = compile_model
        self.memory_pathways = memory_pathways
        self.mixed_precision = mixed_precision
        
        # GPU専用デバイス・データ型管理（Web調査ベース: 文字列→torch.dtype変換）
        self._target_device = self.device
        
        # 🔄 Web調査結果: 文字列dtype→torch.dtype変換
        if isinstance(target_dtype, str):
            if target_dtype == "bfloat16":
                self._target_dtype = torch.bfloat16
            elif target_dtype == "float16":
                self._target_dtype = torch.float16
            elif target_dtype == "float32":
                self._target_dtype = torch.float32
            else:
                # フォールバック: getattr使用
                self._target_dtype = getattr(torch, target_dtype, torch.bfloat16)
        else:
            self._target_dtype = target_dtype
        
        if debug_mode:
            print(f"  🔧 SAM2Wrapper設定:")
            print(f"    - target_dtype: {target_dtype}")
            print(f"    - debug_mode: {debug_mode}")
            print(f"    - vos_optimized: {vos_optimized}")
            print(f"    - compile_model: {compile_model}")
            print(f"    - memory_pathways: {memory_pathways}")
            print(f"    - mixed_precision: {mixed_precision}")
        
        print(f"🔄 Meta公式SAM2初期化中...")
        print(f"  - モデルID: {model_id}")
        print(f"  - デバイス: {self.device}")
        print(f"  - 自動取得: HuggingFace Hub")
        
        # 🔥 Lambda Cloud GPU環境での確実なCUDA初期化（訓練スクリプト対応）
        try:
            if self.device == "cuda":
                print("🔧 SAM2用CUDA初期化...")
                
                # Step 1: CUDA基本初期化
                torch.cuda.init()
                
                # Step 2: 実際のGPU操作でコンテキスト確立
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    current_device = torch.cuda.current_device()
                    
                    # 実際のテンソル操作でGPUコンテキストを作成
                    test_tensor = torch.randn(5, 5, device=self.device)
                    _ = test_tensor.sum()
                    del test_tensor
                    
                    torch.cuda.empty_cache()
                    
                    print(f"  ✅ SAM2用CUDA初期化完了: GPU {current_device}/{device_count}")
                else:
                    raise RuntimeError("CUDA利用不可")
                    
        except Exception as cuda_init_error:
            print(f"  ❌ SAM2用CUDA初期化失敗: {cuda_init_error}")
            print(f"  💡 訓練スクリプトにはGPU環境が必須 - CPUモードは非対応")
            
            # 訓練スクリプトではCPUフォールバックを無効化
            if debug_mode:
                print("  🔥 訓練スクリプト対応: GPU環境を強制要求")
                raise RuntimeError(f"GPU環境が必要です。CUDA初期化エラー: {cuda_init_error}")
            else:
                # テストモードのみCPUフォールバック
                print(f"  ⚠️ テストモード: CPUで実行")
                self.device = "cpu"
                self._target_device = "cpu"
        
        try:
            # 🔄 2025年最適化: VOS対応判定 (修正版: ImagePredictor使用)
            if vos_optimized or compile_model:
                print(f"  - 2025年最適化: HuggingFace優先・torch.compile対応")
                
                try:
                    # ✅ Web調査修正: HuggingFace優先使用 (API統一)
                    hf_model_id = config_linux.SAM2_HF_MODEL_ID
                    self.predictor = SAM2ImagePredictor.from_pretrained(
                        hf_model_id,
                        device=self.device
                    )
                    
                    # 🔄 torch.compile最適化 (ImagePredictorでも有効)
                    if compile_model:
                        try:
                            # Web調査結果: model属性でtorch.compile適用
                            if hasattr(self.predictor, 'model'):
                                self.predictor.model = torch.compile(
                                    self.predictor.model,
                                    mode="default",
                                    dynamic=True
                                )
                                print(f"  ✅ torch.compile適用完了")
                        except Exception as compile_error:
                            print(f"  ⚠️ torch.compile失敗 (動作継続): {compile_error}")
                    
                    print(f"✅ 2025年VOS最適化SAM2初期化成功")
                    print(f"  - HuggingFaceモデル: {hf_model_id}")
                    
                except Exception as hf_error:
                    # フォールバック: 元のmodel_id
                    print(f"  ⚠️ HuggingFace初期化失敗: {hf_error}")
                    print(f"  🔄 元model_idでフォールバック...")
                    
                    self.predictor = SAM2ImagePredictor.from_pretrained(
                        model_id,
                        device=self.device
                    )
                    
                    print(f"✅ SAM2初期化成功（フォールバック）")
                
            else:
                # 🔄 Web調査ベース: HuggingFace標準初期化 (最確実)
                print(f"  - 標準初期化: HuggingFace Hub使用")
                
                try:
                    # ✅ Web調査推奨: HuggingFaceモデルID優先使用
                    hf_model_id = config_linux.SAM2_HF_MODEL_ID
                    self.predictor = SAM2ImagePredictor.from_pretrained(
                        hf_model_id,
                        device=self.device
                    )
                    
                    print(f"✅ HuggingFace SAM2標準初期化成功")
                    print(f"  - HuggingFaceモデル: {hf_model_id}")
                    
                except Exception as std_error:
                    # フォールバック: 元のmodel_id
                    print(f"  ⚠️ HuggingFace標準初期化失敗: {std_error}")
                    print(f"  🔄 元model_idでフォールバック...")
                    
                    self.predictor = SAM2ImagePredictor.from_pretrained(
                        model_id,
                        device=self.device
                    )
                    
                    print(f"✅ SAM2標準初期化成功（フォールバック）")
            
            print("✅ 重みファイル自動取得完了")
            
        except Exception as e:
            error_msg = (
                f"❌ Meta公式SAM2初期化失敗: {e}\n"
                "必須要件:\n"
                "1. pip install segment-anything-2\n"
                "2. pip install huggingface_hub\n" 
                "3. インターネット接続\n"
                "4. HuggingFace Hub アクセス可能"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def set_image(self, image) -> None:
        """
        画像をSAM2に設定 (公式API + 2025年BFloat16対応)
        
        Args:
            image: numpy array (H, W, 3) RGB画像 [0, 255] またはPIL Image
        """
        # PyTorchテンソルをnumpy配列に変換（SAM2要求形式）
        # 2025年ベストプラクティス: BFloat16 → Float32 → uint8変換
        if isinstance(image, torch.Tensor):
            # Web調査結果: BFloat16のNumPy非対応のためFloat32へ先に変換
            if image.dtype == torch.bfloat16:
                image = image.to(torch.float32)
            
            # バッチ処理の対応
            if image.dim() == 4:
                # [B, C, H, W] -> 最初の画像のみ使用（SAM2は単一画像処理）
                if image.size(0) > 1:
                    print(f"⚠️ SAM2は単一画像処理のみ対応。バッチサイズ{image.size(0)}の最初の画像のみ使用")
                image = image[0]  # [C, H, W]
            
            # チャネルを最後に移動 [C, H, W] -> [H, W, C]
            if image.dim() == 3 and image.size(0) in [1, 3]:
                image = image.permute(1, 2, 0)
            
            # 値の範囲を[0, 255]に正規化
            if image.max() <= 1.0:
                image = image * 255.0
            
            image_np = image.detach().cpu().numpy().astype('uint8')
        else:
            image_np = image
            
        # 🔄 Meta公式SAM2 API
        self.predictor.set_image(image_np)
    
    def predict_with_prompts(
        self,
        prompt_embeddings: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Q-Formerプロンプトでセグメンテーション実行 (Meta公式API)
        
        Args:
            prompt_embeddings: (num_queries, embed_dim) Q-Formerからの埋め込み
            point_coords: (num_points, 2) ポイント座標（オプション）
            point_labels: (num_points,) ポイントラベル（オプション）
            boxes: (num_boxes, 4) バウンディングボックス（オプション）
            multimask_output: 複数マスク出力するかどうか
            
        Returns:
            Dict containing:
                - masks: (num_masks, H, W) 予測マスク
                - iou_predictions: (num_masks,) IoU予測値
                - low_res_logits: (num_masks, H//4, W//4) 低解像度ロジット
        """
        
        # SAM2では現在prompt_embeddingsを直接サポートしていないため、
        # 代替プロンプト形式を使用
        
        if point_coords is not None or boxes is not None:
            # 既存のプロンプト形式がある場合
            # 2025年ベストプラクティス: BFloat16 → Float32変換対応
            point_coords_np = None
            if point_coords is not None:
                coords_tensor = point_coords.to(torch.float32) if point_coords.dtype == torch.bfloat16 else point_coords
                point_coords_np = coords_tensor.detach().cpu().numpy()
            
            point_labels_np = None
            if point_labels is not None:
                labels_tensor = point_labels.to(torch.float32) if point_labels.dtype == torch.bfloat16 else point_labels
                point_labels_np = labels_tensor.detach().cpu().numpy()
            
            box_np = None
            if boxes is not None:
                box_tensor = boxes[0].to(torch.float32) if boxes[0].dtype == torch.bfloat16 else boxes[0]
                box_np = box_tensor.detach().cpu().numpy()
            
            masks, iou_predictions, low_res_logits = self.predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                box=box_np,
                multimask_output=multimask_output
            )
        else:
            # Q-Formerプロンプトから推定座標生成（簡易版）
            # 将来: prompt_embeddingsを座標に変換するネットワーク追加
            num_queries = prompt_embeddings.shape[0]
            
            # 複数クエリから複数ポイント生成
            grid_size = int(num_queries ** 0.5) or 2
            H, W = 1024, 1024  # SAM2解像度
            
            points = []
            labels = []
            
            for i in range(min(num_queries, 4)):  # 最大4ポイント
                x = (i % grid_size + 1) * W // (grid_size + 1)
                y = (i // grid_size + 1) * H // (grid_size + 1)
                points.append([x, y])
                labels.append(1)  # 前景
            
            point_coords_np = torch.tensor(points, dtype=torch.float32).numpy()
            point_labels_np = torch.tensor(labels, dtype=torch.int32).numpy()
            
            # 🔄 Meta公式SAM2 API
            masks, iou_predictions, low_res_logits = self.predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                multimask_output=multimask_output
            )
        
        # numpy配列をPyTorchテンソルに変換（GPU専用環境 + 2025年ベストプラクティス Web調査準拠）
        device = getattr(self, '_target_device', 'cuda')
        target_dtype = getattr(self, '_target_dtype', torch.bfloat16)
        
        # 🔄 Web調査結果: PyTorch公式推奨パターン（デバイス・データ型同時変換）
        try:
            # Web調査例: tensor.to(torch.bfloat16, device="cuda")
            masks = torch.from_numpy(masks).to(dtype=target_dtype, device=device)
            iou_predictions = torch.from_numpy(iou_predictions).to(dtype=target_dtype, device=device)
            low_res_logits = torch.from_numpy(low_res_logits).to(dtype=target_dtype, device=device)
        except Exception as to_error:
            # フォールバック: 段階的変換（Web調査ベース）
            print(f"  ⚠️ 同時変換失敗、段階的変換実行: {to_error}")
            masks = torch.from_numpy(masks).to(device).to(target_dtype)
            iou_predictions = torch.from_numpy(iou_predictions).to(device).to(target_dtype)
            low_res_logits = torch.from_numpy(low_res_logits).to(device).to(target_dtype)
        
        # デバッグ情報
        print(f"  📊 SAM2出力統計:")
        print(f"    - masks: {masks.shape}, {masks.dtype}, device: {masks.device}")
        print(f"    - iou_predictions: {iou_predictions.shape}, {iou_predictions.dtype}")
        print(f"    - 平均IoU: {iou_predictions.mean().item():.3f}")
        
        return {
            'masks': masks,
            'iou_predictions': iou_predictions, 
            'low_res_logits': low_res_logits
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """SAM2モデル情報取得"""
        try:
            # SAM2ImagePredictorから実際のモデルにアクセス
            actual_model = self.predictor.model if hasattr(self.predictor, 'model') else None
            
            if actual_model:
                parameters = sum(p.numel() for p in actual_model.parameters())
                image_encoder_type = type(actual_model.image_encoder).__name__ if hasattr(actual_model, 'image_encoder') else 'Unknown'
                mask_decoder_type = type(actual_model.mask_decoder).__name__ if hasattr(actual_model, 'mask_decoder') else 'Unknown'
            else:
                parameters = 0
                image_encoder_type = 'Unknown'
                mask_decoder_type = 'Unknown'
                
            return {
                'model_type': 'Meta SAM2',
                'model_id': self.model_id,
                'device': self.device,
                'parameters': parameters,
                'image_encoder': image_encoder_type,
                'mask_decoder': mask_decoder_type,
            }
        except Exception as e:
            return {
                'model_type': 'Meta SAM2',
                'model_id': self.model_id,
                'device': self.device,
                'parameters': 'Unknown',
                'error': str(e)
            }


# MockSAM2Wrapper削除 - Meta公式API一本化


def get_sam2_wrapper(
    model_id: str = "facebook/sam2-hiera-large", 
    target_dtype: torch.dtype = torch.bfloat16,
    debug_mode: bool = True,
    **kwargs
) -> SAM2Wrapper:
    """
    Meta公式SAM2ファクトリ関数 (フォールバック無し)
    
    Args:
        model_id: HuggingFace Hub上のSAM2モデルID
                 - facebook/sam2-hiera-large (推奨, 最高精度)
                 - facebook/sam2-hiera-base
                 - facebook/sam2-hiera-small  
                 - facebook/sam2-hiera-tiny
        target_dtype: 出力テンソルのデータ型 (デフォルト: BFloat16)
        **kwargs: その他のパラメータ
        
    Returns:
        SAM2Wrapper (Meta公式のみ)
        
    Raises:
        ImportError: SAM2が利用できない場合
        RuntimeError: SAM2初期化に失敗した場合
    """
    if not SAM2_AVAILABLE:
        raise ImportError(
            "Meta公式SAM2が利用できません。\n"
            "必須インストール:\n"
            "1. pip install segment-anything-2\n"
            "2. pip install huggingface_hub"
        )
    
    print(f"🔄 Meta公式SAM2を使用: {model_id}")
    print(f"  - データ型統一: {target_dtype}")
    print(f"  - デバッグモード: {debug_mode}")
    return SAM2Wrapper(model_id=model_id, target_dtype=target_dtype, debug_mode=debug_mode, **kwargs)


def test_sam2_integration():
    """Meta公式SAM2統合テスト (フォールバック無し)"""
    print("=== Meta公式SAM2統合テスト ===")
    
    try:
        # Meta公式SAM2初期化 (エラー時は例外発生)
        sam2 = get_sam2_wrapper()
        
        # モデル情報表示
        info = sam2.get_model_info()
        print(f"\nSAM2モデル情報:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # ダミー画像とプロンプト作成
        image = torch.randint(0, 255, (1024, 1024, 3), dtype=torch.uint8)
        prompt_embeddings = torch.randn(8, 256)  # Q-Formerからの8個のクエリ
        
        print(f"\nテストデータ:")
        print(f"  - 画像: {image.shape}")
        print(f"  - プロンプト埋め込み: {prompt_embeddings.shape}")
        
        # 画像設定
        sam2.set_image(image)
        
        # セグメンテーション実行
        with torch.no_grad():
            results = sam2.predict_with_prompts(prompt_embeddings)
        
        print(f"\n結果:")
        print(f"  - マスク: {results['masks'].shape}")
        print(f"  - IoU予測: {results['iou_predictions'].shape}")
        print(f"  - 低解像度ロジット: {results['low_res_logits'].shape}")
        
        # マスク品質確認
        masks = results['masks']
        print(f"\nマスク品質:")
        print(f"  - 平均IoU: {results['iou_predictions'].mean().item():.3f}")
        print(f"  - マスク平均値: {masks.mean().item():.3f}")
        print(f"  - マスク標準偏差: {masks.std().item():.3f}")
        
        print("\n✅ Meta公式SAM2統合テスト完了")
        
    except (ImportError, RuntimeError) as e:
        print(f"\n❌ Meta公式SAM2テスト失敗: {e}")
        print("Lambda Cloud環境での実行を推奨")


if __name__ == "__main__":
    test_sam2_integration()