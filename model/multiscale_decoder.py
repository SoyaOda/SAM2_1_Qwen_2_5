# model/multiscale_decoder.py
"""
マルチスケールセグメンテーションヘッド

o3-modification20250727.mdに基づく実装:
- SAM2のマスクデコーダーを拡張
- 複数解像度の特徴を活用
- 補助デコーダーによる高精度化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import math


class MultiScaleFeatureExtractor(nn.Module):
    """
    ViTからの多段特徴抽出
    
    SAM2のHiera画像エンコーダから複数解像度の特徴を抽出
    """
    
    def __init__(self, stages: List[int] = [3, 6, 9, 12]):
        super().__init__()
        self.stages = stages
        self.feature_maps = {}
        self.hooks = []
        
        print(f"✅ MultiScaleFeatureExtractor初期化")
        print(f"  - 抽出ステージ: {stages}")
    
    def register_hooks(self, model: nn.Module):
        """
        モデルの中間層にフックを登録
        
        Args:
            model: SAM2の画像エンコーダモデル
        """
        # 既存のフックをクリア
        self.remove_hooks()
        self.feature_maps.clear()
        
        # SAM2のHieraエンコーダーの構造を確認
        blocks = None
        stage_ends = None
        
        # SAM2のImageEncoderの場合（trunk.blocks構造）
        if hasattr(model, 'trunk') and hasattr(model.trunk, 'blocks'):
            trunk = model.trunk
            blocks = trunk.blocks
            
            # stage_endsがある場合は、ステージの境界を取得
            if hasattr(trunk, 'stage_ends'):
                stage_ends = trunk.stage_ends
                print(f"  - SAM2 Hiera検出: {len(blocks)}ブロック, ステージ境界: {stage_ends}")
        
        # 標準的なViT構造
        elif hasattr(model, 'blocks'):
            blocks = model.blocks
        
        # HuggingFace形式
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            blocks = model.encoder.layer
        
        if blocks is None or (isinstance(blocks, list) and len(blocks) == 0):
            # フォールバック: フック登録をスキップ
            print("⚠️ MultiScaleFeatureExtractor: 適切なブロック構造が見つかりません")
            print(f"  - モデルタイプ: {type(model)}")
            print(f"  - 利用可能な属性: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
            return
        
        # 各ステージにフックを登録
        if stage_ends is not None:
            # SAM2 Hieraの場合: stage_endsを使用してステージの最終ブロックにフック
            for stage_idx, block_idx in enumerate(stage_ends):
                if block_idx < len(blocks):
                    block = blocks[block_idx]
                    # ステージ番号は1から開始
                    stage_num = stage_idx + 1
                    hook = block.register_forward_hook(
                        lambda m, i, o, stage=stage_num: self._save_feature(o, f'stage_{stage}')
                    )
                    self.hooks.append(hook)
                    print(f"  - Stage {stage_num} (Block {block_idx})にフック登録")
        else:
            # 通常のViT: self.stagesで指定されたインデックスにフック
            for stage_idx in self.stages:
                if stage_idx <= len(blocks):
                    block = blocks[stage_idx - 1]
                    hook = block.register_forward_hook(
                        lambda m, i, o, stage=stage_idx: self._save_feature(o, f'stage_{stage}')
                    )
                    self.hooks.append(hook)
                    print(f"  - Stage {stage_idx}にフック登録")
    
    def _save_feature(self, output: torch.Tensor, name: str):
        """特徴マップを保存"""
        # outputがタプルの場合は最初の要素を取得
        if isinstance(output, tuple):
            output = output[0]
        self.feature_maps[name] = output
    
    def remove_hooks(self):
        """登録したフックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, x: torch.Tensor, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        マルチスケール特徴抽出
        
        Args:
            x: 入力画像
            model: SAM2画像エンコーダ
            
        Returns:
            各ステージの特徴マップの辞書
        """
        # フックが登録されていない場合は登録を試みる
        if not self.hooks:
            self.register_hooks(model)
        
        # 特徴マップをクリア
        self.feature_maps.clear()
        
        # フックが登録されている場合のみマルチスケール特徴抽出
        if self.hooks:
            # モデルを実行（フックが特徴を収集）
            with torch.no_grad():
                output = model(x)
            
            # SAM2の場合、出力が辞書形式
            if isinstance(output, dict) and 'vision_features' in output:
                final_features = output['vision_features']
            else:
                final_features = output
            
            # 最終出力も保存（Q-Former用）
            self.feature_maps['final'] = final_features
        else:
            # フックが登録できない場合は通常の出力のみ
            print("⚠️ マルチスケール特徴抽出が利用できません - 最終出力のみ使用")
            with torch.no_grad():
                output = model(x)
            
            # SAM2の場合、出力が辞書形式
            if isinstance(output, dict) and 'vision_features' in output:
                final_features = output['vision_features']
            else:
                final_features = output
                
            self.feature_maps['final'] = final_features
        
        # 収集した特徴マップを返す
        return self.feature_maps.copy()


class AuxiliaryDecoder(nn.Module):
    """
    補助デコーダー（高解像度特徴用）
    
    浅い層の高解像度特徴を処理し、細部のマスクを生成
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        out_channels: int = 256
    ):
        super().__init__()
        
        # 高解像度処理ブランチ
        self.high_res_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(32, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 中解像度処理ブランチ
        self.mid_res_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特徴統合
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        )
        
        # 粗マスクの精細化
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + 1, hidden_channels, 3, padding=1),  # +1 for coarse mask
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )
        
        print(f"✅ AuxiliaryDecoder初期化")
        print(f"  - 入力チャネル: {in_channels}")
        print(f"  - 隠れチャネル: {hidden_channels}")
    
    def forward(
        self,
        high_res_feat: torch.Tensor,
        mid_res_feat: torch.Tensor,
        coarse_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        高解像度特徴を用いたマスク精細化
        
        Args:
            high_res_feat: 高解像度特徴 [B, C, H1, W1]
            mid_res_feat: 中解像度特徴 [B, C, H2, W2]
            coarse_mask: 粗いマスク [B, 1, H, W]
            
        Returns:
            精細化されたマスク [B, 1, H, W]
        """
        # 解像度を合わせる
        target_size = coarse_mask.shape[-2:]
        
        # 各解像度の処理
        high_feat = self.high_res_branch(high_res_feat)
        high_feat = F.interpolate(high_feat, size=target_size, mode='bilinear', align_corners=False)
        
        mid_feat = self.mid_res_branch(mid_res_feat)
        mid_feat = F.interpolate(mid_feat, size=target_size, mode='bilinear', align_corners=False)
        
        # 特徴統合
        combined_feat = torch.cat([high_feat, mid_feat], dim=1)
        fused_feat = self.fusion(combined_feat)
        
        # 粗マスクと結合して精細化
        mask_and_feat = torch.cat([fused_feat, coarse_mask], dim=1)
        refined_mask = self.refine(mask_and_feat)
        
        return refined_mask


class EnhancedMaskDecoder(nn.Module):
    """
    マルチスケール対応MaskDecoder
    
    SAM2のMaskDecoderを拡張し、複数解像度の特徴を活用
    """
    
    def __init__(
        self,
        sam_mask_decoder: nn.Module,
        image_encoder_dim: int = 768,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 既存のSAM2 MaskDecoder
        self.main_decoder = sam_mask_decoder
        
        # 補助デコーダー
        self.aux_decoder = AuxiliaryDecoder(
            in_channels=image_encoder_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim
        )
        
        # マルチスケール特徴の投影
        self.feature_projectors = nn.ModuleDict({
            'high': nn.Conv2d(image_encoder_dim, image_encoder_dim, 1),
            'mid': nn.Conv2d(image_encoder_dim, image_encoder_dim, 1),
            'low': nn.Conv2d(image_encoder_dim, image_encoder_dim, 1)
        })
        
        # 視覚コンテキストの処理（LLMからの入力用）
        self.context_projector = nn.Sequential(
            nn.Linear(1024, hidden_dim),  # SAM2互換の次元
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # マスク融合
        self.mask_fusion = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3 masks: main, aux, context
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        print(f"✅ EnhancedMaskDecoder初期化")
        print(f"  - メインデコーダー: SAM2 MaskDecoder")
        print(f"  - 補助デコーダー: マルチスケール対応")
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multiscale_features: Optional[Dict[str, torch.Tensor]] = None,
        visual_context: Optional[torch.Tensor] = None,
        multimask_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        マルチスケール統合マスク生成
        
        Args:
            image_embeddings: 低解像度特徴（メイン）[B, C, H, W]
            image_pe: 位置エンコーディング
            sparse_prompt_embeddings: スパースプロンプト
            dense_prompt_embeddings: デンスプロンプト
            multiscale_features: マルチスケール特徴の辞書
            visual_context: LLMからの視覚コンテキスト [B, N, D]
            multimask_output: 複数マスク出力するか
            
        Returns:
            masks: 予測マスク [B, num_masks, H, W]
            iou_pred: IoU予測スコア [B, num_masks]
        """
        # メインデコーダー（低解像度特徴）
        masks_main, iou_pred = self.main_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output
        )
        
        # マルチスケール処理が不要な場合
        if multiscale_features is None or len(multiscale_features) == 0:
            return masks_main, iou_pred
        
        # 最良のマスクを選択（multimask_outputの場合）
        if multimask_output and masks_main.size(1) > 1:
            # IoUスコアが最高のマスクを選択
            best_idx = iou_pred.argmax(dim=1)
            coarse_mask = masks_main[torch.arange(masks_main.size(0)), best_idx].unsqueeze(1)
        else:
            coarse_mask = masks_main[:, 0:1]  # 最初のマスク
        
        # マルチスケール特徴の取得と投影
        high_res_feat = None
        mid_res_feat = None
        
        # 特徴マップの解像度に基づいて分類
        for name, feat in multiscale_features.items():
            if 'stage_1' in name or 'stage_2' in name:
                # 高解像度
                if high_res_feat is None:
                    high_res_feat = self.feature_projectors['high'](feat)
            elif 'stage_3' in name or 'stage_4' in name:
                # 中解像度
                if mid_res_feat is None:
                    mid_res_feat = self.feature_projectors['mid'](feat)
        
        # 補助デコーダーでの精細化
        if high_res_feat is not None and mid_res_feat is not None:
            # 特徴マップの形状を確認・調整
            if high_res_feat.dim() == 3:  # [B, N, C]の場合
                # 空間次元を復元
                B, N, C = high_res_feat.shape
                H = W = int(math.sqrt(N))
                high_res_feat = high_res_feat.transpose(1, 2).view(B, C, H, W)
            
            if mid_res_feat.dim() == 3:  # [B, N, C]の場合
                B, N, C = mid_res_feat.shape
                H = W = int(math.sqrt(N))
                mid_res_feat = mid_res_feat.transpose(1, 2).view(B, C, H, W)
            
            # 補助マスク生成
            aux_mask = self.aux_decoder(high_res_feat, mid_res_feat, coarse_mask)
            
            # 最終解像度にアップサンプル
            target_size = masks_main.shape[-2:]
            aux_mask = F.interpolate(aux_mask, size=target_size, mode='bilinear', align_corners=False)
        else:
            # マルチスケール特徴がない場合はコピー
            aux_mask = coarse_mask
        
        # 視覚コンテキストの処理
        if visual_context is not None:
            # コンテキストから空間的な重みマップを生成
            context_weights = self._compute_context_weights(visual_context, masks_main.shape)
        else:
            # コンテキストがない場合は均一な重み
            context_weights = torch.ones_like(coarse_mask)
        
        # マスクの融合
        # 3つのマスクを組み合わせ: メイン、補助、コンテキスト重み付き
        combined_masks = torch.cat([
            coarse_mask,
            aux_mask,
            coarse_mask * context_weights
        ], dim=1)
        
        final_mask = self.mask_fusion(combined_masks)
        
        # multimask_outputの場合は元の形式に合わせる
        if multimask_output:
            # 元のマスクと融合マスクを結合
            final_masks = torch.cat([final_mask, masks_main[:, 1:]], dim=1)
            # IoUスコアも調整（融合マスクのスコアを少し高く）
            final_iou = torch.cat([
                iou_pred[:, 0:1] * 1.1,  # 融合マスクのスコアを10%上げる
                iou_pred[:, 1:]
            ], dim=1)
        else:
            final_masks = final_mask
            final_iou = iou_pred
        
        return final_masks, final_iou
    
    def _compute_context_weights(
        self,
        visual_context: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """
        視覚コンテキストから空間的重みマップを生成
        """
        B, N, D = visual_context.shape
        H, W = target_shape[-2:]
        
        # コンテキストベクトルをプール
        context_pooled = visual_context.mean(dim=1)  # [B, D]
        
        # 空間的な重みに変換
        context_proj = self.context_projector(context_pooled)  # [B, hidden_dim]
        
        # 1x1の重みマップとして開始
        weight_map = context_proj.view(B, -1, 1, 1)
        
        # ターゲットサイズにアップサンプル
        weight_map = F.interpolate(
            weight_map[:, 0:1],  # 最初のチャネルのみ使用
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        # シグモイドで正規化
        weight_map = torch.sigmoid(weight_map)
        
        return weight_map


class MultiScaleSegmentationHead(nn.Module):
    """
    マルチスケールセグメンテーションヘッドの統合クラス
    """
    
    def __init__(
        self,
        sam_wrapper,
        stages: List[int] = [3, 6, 9, 12]
    ):
        super().__init__()
        
        # マルチスケール特徴抽出器
        self.feature_extractor = MultiScaleFeatureExtractor(stages)
        
        # 拡張マスクデコーダー
        # SAM2の場合はpredictor.modelからsam_mask_decoderを取得
        if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
            # SAM2の場合
            actual_model = sam_wrapper.predictor.model
            if hasattr(actual_model, 'sam_mask_decoder'):
                mask_decoder = actual_model.sam_mask_decoder
            else:
                raise AttributeError("SAM2 modelにsam_mask_decoderが見つかりません")
        elif hasattr(sam_wrapper, 'mask_decoder'):
            # SAM1の場合
            mask_decoder = sam_wrapper.mask_decoder
        else:
            raise AttributeError("sam_mask_decoderが見つかりません")
            
        self.enhanced_decoder = EnhancedMaskDecoder(
            sam_mask_decoder=mask_decoder,
            image_encoder_dim=768,  # ViT-L
            hidden_dim=256
        )
        
        # SAMの他のコンポーネント
        # SAM2の場合はpredictor.modelから取得
        if hasattr(sam_wrapper, 'predictor') and hasattr(sam_wrapper.predictor, 'model'):
            # SAM2の場合
            actual_model = sam_wrapper.predictor.model
            self.image_encoder = actual_model.image_encoder if hasattr(actual_model, 'image_encoder') else None
            # SAM2ではsam_prompt_encoderが正しい属性名
            self.prompt_encoder = actual_model.sam_prompt_encoder if hasattr(actual_model, 'sam_prompt_encoder') else None
            
            # SAM2Wrapperも保持（predict_with_promptsメソッド用）
            self.sam_wrapper = sam_wrapper
        else:
            # SAM1の場合
            self.image_encoder = sam_wrapper.image_encoder
            self.prompt_encoder = sam_wrapper.prompt_encoder
            self.sam_wrapper = sam_wrapper
        
        print(f"✅ MultiScaleSegmentationHead初期化完了")
    
    def forward(
        self,
        images: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        visual_context: Optional[torch.Tensor] = None,
        multimask_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        マルチスケールセグメンテーション
        """
        # image_encoderの確認
        if self.image_encoder is None:
            raise RuntimeError(
                "MultiScaleSegmentationHeadのimage_encoderがNoneです。"
                "SAM2の初期化でimage_encoderが正しく設定されませんでした。"
            )
        
        # マルチスケール特徴抽出
        multiscale_features = self.feature_extractor(images, self.image_encoder)
        
        # メイン特徴（最終層またはfinal）
        if 'stage_4' in multiscale_features:
            image_embeddings = multiscale_features['stage_4']
        elif 'final' in multiscale_features:
            image_embeddings = multiscale_features['final']
        else:
            # 全ての特徴マップから最も適切なものを選択
            if multiscale_features:
                # 最後のステージの特徴を使用
                sorted_keys = sorted(multiscale_features.keys())
                image_embeddings = multiscale_features[sorted_keys[-1]]
            else:
                # フォールバック: 直接エンコード
                encoded_output = self.image_encoder(images)
                # SAM2は辞書形式で返すことがある
                if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
                    image_embeddings = encoded_output['vision_features']
                else:
                    image_embeddings = encoded_output
        
        # prompt_encoderの確認
        if self.prompt_encoder is None:
            raise RuntimeError(
                "MultiScaleSegmentationHeadのprompt_encoderがNoneです。"
                "SAM2の初期化でprompt_encoderが正しく設定されませんでした。"
            )
        
        # プロンプトエンコーディング
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks
        )
        
        # 位置エンコーディング
        image_pe = self.prompt_encoder.get_dense_pe()
        
        # マルチスケールマスク生成
        masks, iou_predictions = self.enhanced_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multiscale_features=multiscale_features,
            visual_context=visual_context,
            multimask_output=multimask_output
        )
        
        return masks, iou_predictions, image_embeddings