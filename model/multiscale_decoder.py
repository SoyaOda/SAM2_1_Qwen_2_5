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
        モデルの中間層にフックを登録（Webリサーチに基づくSAM2 Hiera対応版）
        
        Args:
            model: SAM2の画像エンコーダモデル
        """
        # 既存のフックをクリア
        self.remove_hooks()
        self.feature_maps.clear()
        
        # SAM2のHieraエンコーダーの構造を確認
        blocks = None
        stage_ends = None
        
        # デバッグ: モデル構造を詳細に調査
        print(f"🔍 SAM2 Hieraモデル構造調査（Webリサーチベース）:")
        print(f"  - モデルタイプ: {type(model).__name__}")
        print(f"  - 利用可能な主要属性:")
        for attr in ['trunk', 'neck', 'fpn', 'patch_embed', 'blocks', 'encoder']:
            if hasattr(model, attr):
                attr_obj = getattr(model, attr)
                print(f"    - {attr}: {type(attr_obj).__name__}")
                # さらに深く調査
                if hasattr(attr_obj, 'blocks'):
                    blocks_obj = attr_obj.blocks
                    print(f"      - {attr}.blocks: {type(blocks_obj).__name__} (長さ: {len(blocks_obj) if hasattr(blocks_obj, '__len__') else 'N/A'})")
                    # Hieraの特有属性をチェック
                    if hasattr(attr_obj, 'stage_ends'):
                        print(f"      - {attr}.stage_ends: {getattr(attr_obj, 'stage_ends')}")
        
        # SAM2のImageEncoderの場合（trunk.blocks構造）
        if hasattr(model, 'trunk') and hasattr(model.trunk, 'blocks'):
            trunk = model.trunk
            blocks = trunk.blocks
            
            # Webリサーチ結果: SAM2 Hieraは48ブロック、4つのステージに分割
            # Stage 1 (stride 4): 高解像度特徴 - blocks 0-11
            # Stage 2 (stride 8): 中解像度特徴 - blocks 12-23  
            # Stage 3 (stride 16): 低解像度特徴（FPN用） - blocks 24-35
            # Stage 4 (stride 32): 最低解像度特徴（FPN用） - blocks 36-47
            
            # stage_endsがある場合は、ステージの境界を取得
            if hasattr(trunk, 'stage_ends'):
                stage_ends = trunk.stage_ends
                print(f"  - SAM2 Hiera検出: {len(blocks)}ブロック, ステージ境界: {stage_ends}")
            else:
                # Webリサーチによる標準的なHieraステージ分割（48ブロックの場合）
                if len(blocks) == 48:
                    stage_ends = [11, 23, 35, 47]  # 各ステージの最終ブロック
                    print(f"  - SAM2 Hiera標準構成推定: 48ブロック -> ステージ境界: {stage_ends}")
                else:
                    # 動的計算: ブロック数を4等分
                    stage_size = len(blocks) // 4
                    stage_ends = [stage_size-1, stage_size*2-1, stage_size*3-1, len(blocks)-1]
                    print(f"  - SAM2 Hiera動的構成推定: {len(blocks)}ブロック -> ステージ境界: {stage_ends}")
            
            # Hieraの詳細構造を確認
            print(f"  - Hieraブロック詳細:")
            for i, block in enumerate(blocks[:3]):  # 最初の3ブロックのみ
                print(f"    - Block {i}: {type(block).__name__}")
        
        # 標準的なViT構造
        elif hasattr(model, 'blocks'):
            blocks = model.blocks
            print(f"  - 標準ViT構造検出: {len(blocks)}ブロック")
        
        # HuggingFace形式
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            blocks = model.encoder.layer
            print(f"  - HuggingFace形式検出: {len(blocks)}層")
        
        if blocks is None or (isinstance(blocks, list) and len(blocks) == 0):
            # フォールバック: フック登録をスキップ
            print("⚠️ MultiScaleFeatureExtractor: 適切なブロック構造が見つかりません")
            print(f"  - モデルタイプ: {type(model)}")
            print(f"  - 利用可能な属性: {[attr for attr in dir(model) if not attr.startswith('_')][:15]}")
            return
        
        # 各ステージにフックを登録（Webリサーチベース）
        if stage_ends is not None:
            # SAM2 Hieraの場合: 4つのストライドレベルに対応するフック
            stride_levels = [4, 8, 16, 32]  # Webリサーチ結果
            
            for stage_idx, block_idx in enumerate(stage_ends):
                if block_idx < len(blocks):
                    block = blocks[block_idx]
                    # ステージ番号は1から開始、strideレベル情報も含める
                    stage_num = stage_idx + 1
                    stride = stride_levels[stage_idx] if stage_idx < len(stride_levels) else 32
                    
                    hook = block.register_forward_hook(
                        lambda m, i, o, stage=stage_num, stride_val=stride: 
                        self._save_feature(o, f'stage_{stage}_stride_{stride_val}')
                    )
                    self.hooks.append(hook)
                    print(f"  - Stage {stage_num} (Block {block_idx}, Stride {stride})にフック登録")
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
        # 特徴マップをクリア
        self.feature_maps.clear()
        
        # SAM2のFPN neck出力を直接使用する（より安定的）
        if hasattr(model, 'neck') and hasattr(model, 'trunk'):
            print("🔍 SAM2 FPN neck経由の特徴抽出を実行...")
            with torch.no_grad():
                # トランクで階層的特徴を抽出
                trunk_output = model.trunk(x)
                # FPN neckで特徴を融合
                features, pos = model.neck(trunk_output)
                
                # FPN出力は通常リスト形式（複数レベルの特徴）
                if isinstance(features, (list, tuple)):
                    # 最高解像度の特徴を使用（通常は最初の要素）
                    main_features = features[0] if len(features) > 0 else features
                    self.feature_maps['fpn_features'] = main_features
                    self.feature_maps['final'] = main_features
                    print(f"✅ FPN特徴抽出成功:")
                    print(f"  - FPN特徴数: {len(features)}")
                    print(f"  - メイン特徴: {main_features.shape}")
                else:
                    self.feature_maps['fpn_features'] = features
                    self.feature_maps['final'] = features
                    print(f"✅ FPN特徴抽出成功:")
                    print(f"  - FPN特徴: {features.shape}")
                
                if pos is not None:
                    self.feature_maps['fpn_pos_encoding'] = pos
                    if isinstance(pos, (list, tuple)):
                        print(f"  - 位置エンコーディング数: {len(pos)}")
                    else:
                        print(f"  - 位置エンコーディング: {pos.shape}")
                
            return self.feature_maps
        
        # フォールバック: フックベースの抽出
        # フックが登録されていない場合は登録を試みる
        if not self.hooks:
            self.register_hooks(model)
        
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
            
            # デバッグ: 収集された特徴マップの形状を確認
            print(f"🔍 マルチスケール特徴抽出結果:")
            for name, feat in self.feature_maps.items():
                if isinstance(feat, torch.Tensor):
                    print(f"  - {name}: {feat.shape} (device: {feat.device}, dtype: {feat.dtype})")
                else:
                    print(f"  - {name}: {type(feat)}")
            
            # 特徴マップが期待される形状でない場合の修正
            # SAM2 Hieraは[B, num_patches, hidden_dim]形式で出力することがある
            for name, feat in list(self.feature_maps.items()):
                if isinstance(feat, torch.Tensor) and feat.dim() == 3:
                    B, N, C = feat.shape
                    # トークン数が少なすぎる場合はスキップ
                    if N < 100:  # 10x10以下の解像度は無視
                        print(f"  ⚠️ {name}のトークン数が少なすぎます ({N}トークン) - スキップ")
                        del self.feature_maps[name]
                        continue
                    
                    # 空間次元を推定して4D形状に変換
                    H = W = int(math.sqrt(N))
                    if H * W == N:
                        # 正方形の場合
                        feat_4d = feat.transpose(1, 2).reshape(B, C, H, W)
                        self.feature_maps[name] = feat_4d
                        print(f"  ✅ {name}を4D形状に変換: {feat.shape} -> {feat_4d.shape}")
                    else:
                        # 正方形でない場合は最も近い矩形を推定
                        # SAM2は通常正方形なのでこのケースは稀
                        print(f"  ⚠️ {name}の空間次元が推定できません (N={N}) - そのまま保持")
                        
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
        # SAM2標準次元一致確認（デバッグ修正ルール対応）
        # Webリサーチ結果: image_embeddings, dense_prompt_embeddingsの空間次元は一致が必要
        if image_embeddings.dim() == 4 and dense_prompt_embeddings.dim() == 4:
            img_h, img_w = image_embeddings.shape[-2:]
            dense_h, dense_w = dense_prompt_embeddings.shape[-2:]
            
            if (img_h, img_w) != (dense_h, dense_w):
                print(f"  🔧 次元不整合修正: image_embeddings{(img_h, img_w)} vs dense_prompt_embeddings{(dense_h, dense_w)}")
                
                # dense_prompt_embeddingsをimage_embeddingsの解像度に合わせる
                import torch.nn.functional as F
                dense_prompt_embeddings = F.interpolate(
                    dense_prompt_embeddings, 
                    size=(img_h, img_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                print(f"  ✅ dense_prompt_embeddings調整完了: -> {(img_h, img_w)}")
        
        # 高解像度特徴の準備（Webリサーチ結果に基づくSAM2 Hiera対応版）
        # SAM2は[stride4_features, stride8_features]のリストを期待
        # stride 4と8の特徴はマスクデコーダのアップサンプリング層に追加される
        high_res_features = None
        if multiscale_features is not None and len(multiscale_features) > 1:
            stride4_feat = None  # Stage 1: stride 4, 高解像度特徴
            stride8_feat = None  # Stage 2: stride 8, 中解像度特徴
            
            # Webリサーチベース: stride値によるダイレクト選択
            for name, feat in multiscale_features.items():
                if isinstance(feat, torch.Tensor):
                    # 新しい命名規則に対応: stage_X_stride_Y
                    if 'stride_4' in name:
                        stride4_feat = feat
                        print(f"  🎯 Stride 4特徴直接選択: {name} {feat.shape}")
                    elif 'stride_8' in name:
                        stride8_feat = feat
                        print(f"  🎯 Stride 8特徴直接選択: {name} {feat.shape}")
                    # 従来の命名規則にも対応
                    elif 'stage_1' in name and stride4_feat is None:
                        stride4_feat = feat
                        print(f"  🔧 Stage 1をStride 4特徴として選択: {name} {feat.shape}")
                    elif 'stage_2' in name and stride8_feat is None:
                        stride8_feat = feat
                        print(f"  🔧 Stage 2をStride 8特徴として選択: {name} {feat.shape}")
            
            # フォールバック: 解像度ベースの選択
            if stride4_feat is None or stride8_feat is None:
                print(f"  🔄 フォールバック: 解像度ベースの特徴選択")
                for name, feat in multiscale_features.items():
                    if isinstance(feat, torch.Tensor) and feat.dim() == 4:
                        _, _, h, w = feat.shape
                        # Webリサーチ結果: SAM2の標準解像度に基づく分類
                        if h >= 224 and w >= 224 and stride4_feat is None:  # 高解像度 (stride 4相当)
                            stride4_feat = feat
                            print(f"    📐 高解像度特徴をStride 4として選択: {name} {feat.shape}")
                        elif h >= 112 and w >= 112 and stride8_feat is None:  # 中解像度 (stride 8相当)
                            stride8_feat = feat
                            print(f"    📐 中解像度特徴をStride 8として選択: {name} {feat.shape}")
            
            # 両方の解像度が利用可能な場合のみhigh_res_featuresを設定
            if stride4_feat is not None and stride8_feat is not None:
                # Webリサーチ結果: SAM2期待解像度への調整
                import torch.nn.functional as F
                
                # SAM2の標準解像度（image_embeddingsは64x64が基準）
                base_h, base_w = image_embeddings.shape[-2:]  # 通常64x64
                
                # Webリサーチ結果: stride 4特徴は4倍解像度（256x256）
                expected_stride4_size = (base_h * 4, base_w * 4)  # 256x256
                if stride4_feat.dim() == 3:
                    # [B, N, C] -> [B, C, H, W]への変換
                    B, N, C = stride4_feat.shape
                    H = W = int(math.sqrt(N))
                    stride4_feat = stride4_feat.transpose(1, 2).reshape(B, C, H, W)
                
                if stride4_feat.shape[-2:] != expected_stride4_size:
                    stride4_feat = F.interpolate(
                        stride4_feat, size=expected_stride4_size, 
                        mode='bilinear', align_corners=False
                    )
                    print(f"    📐 Stride 4特徴リサイズ: -> {expected_stride4_size}")
                
                # Webリサーチ結果: stride 8特徴は2倍解像度（128x128）
                expected_stride8_size = (base_h * 2, base_w * 2)  # 128x128
                if stride8_feat.dim() == 3:
                    # [B, N, C] -> [B, C, H, W]への変換
                    B, N, C = stride8_feat.shape
                    H = W = int(math.sqrt(N))
                    stride8_feat = stride8_feat.transpose(1, 2).reshape(B, C, H, W)
                
                if stride8_feat.shape[-2:] != expected_stride8_size:
                    stride8_feat = F.interpolate(
                        stride8_feat, size=expected_stride8_size,
                        mode='bilinear', align_corners=False
                    )
                    print(f"    📐 Stride 8特徴リサイズ: -> {expected_stride8_size}")
                
                # SAM2期待フォーマット: [stride4_features, stride8_features]
                high_res_features = [stride4_feat, stride8_feat]
                print(f"  ✅ 高解像度特徴準備完了（Webリサーチベース）: [{stride4_feat.shape}, {stride8_feat.shape}]")
            else:
                print(f"  ⚠️ 高解像度特徴不足: stride4={stride4_feat is not None}, stride8={stride8_feat is not None}")
                if stride4_feat is not None:
                    print(f"    - 利用可能なStride 4特徴: {stride4_feat.shape}")
                if stride8_feat is not None:
                    print(f"    - 利用可能なStride 8特徴: {stride8_feat.shape}")
        
        # メインデコーダー（低解像度特徴）
        # SAM2 MaskDecoderにはrepeat_imageとhigh_res_featuresパラメータが必要
        masks_main, iou_pred = self.main_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # バッチ処理済みの場合
            high_res_features=high_res_features  # SAM2高解像度特徴
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
        
        # マルチスケール特徴の取得と投影（Webリサーチベース）
        high_res_feat = None
        mid_res_feat = None
        
        # Webリサーチ結果に基づく特徴分類
        for name, feat in multiscale_features.items():
            if isinstance(feat, torch.Tensor):
                # stride値による直接分類
                if 'stride_4' in name:
                    # Stage 1: 高解像度特徴（アップサンプリング用）
                    high_res_feat = self.feature_projectors['high'](feat)
                    print(f"    📊 高解像度特徴投影: {name} -> {high_res_feat.shape}")
                elif 'stride_8' in name:
                    # Stage 2: 中解像度特徴（アップサンプリング用）
                    mid_res_feat = self.feature_projectors['mid'](feat)
                    print(f"    📊 中解像度特徴投影: {name} -> {mid_res_feat.shape}")
                # 従来の命名規則による分類
                elif ('stage_1' in name or 'stage_2' in name) and high_res_feat is None:
                    # 高解像度（stride 4, 8相当）
                    high_res_feat = self.feature_projectors['high'](feat)
                    print(f"    📊 高解像度特徴投影（従来命名）: {name} -> {high_res_feat.shape}")
                elif ('stage_3' in name or 'stage_4' in name) and mid_res_feat is None:
                    # 中解像度（stride 16, 32相当、FPN用）
                    mid_res_feat = self.feature_projectors['mid'](feat)
                    print(f"    📊 中解像度特徴投影（従来命名）: {name} -> {mid_res_feat.shape}")
        
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
        
        # デバッグ修正ルール: 段階的修正でマルチスケール特徴抽出
        print(f"🔍 MultiScaleSegmentationHead.forward開始")
        print(f"  - 入力画像: {images.shape}")
        
        # まず基本的な画像エンコードで動作確認
        try:
            # 直接エンコード
            encoded_output = self.image_encoder(images)
            # SAM2は辞書形式で返すことがある
            if isinstance(encoded_output, dict) and 'vision_features' in encoded_output:
                image_embeddings = encoded_output['vision_features']
            else:
                image_embeddings = encoded_output
            
            print(f"  ✅ 基本エンコード成功: {image_embeddings.shape}")
            
            # マルチスケール特徴抽出を試行
            try:
                multiscale_features = self.feature_extractor(images, self.image_encoder)
                
                # マルチスケール特徴の利用可能性を確認
                valid_features = {}
                for name, feat in multiscale_features.items():
                    if isinstance(feat, torch.Tensor) and feat.numel() > 0:
                        valid_features[name] = feat
                
                if len(valid_features) > 1:
                    print(f"  ✅ マルチスケール特徴利用可能: {list(valid_features.keys())}")
                    
                    # finalがあればそれを優先使用
                    if 'final' in valid_features:
                        image_embeddings = valid_features['final']
                    # stage_4があれば使用
                    elif 'stage_4' in valid_features:
                        image_embeddings = valid_features['stage_4']
                    # それ以外は最後のステージを使用
                    else:
                        sorted_keys = sorted(valid_features.keys())
                        image_embeddings = valid_features[sorted_keys[-1]]
                else:
                    print(f"  ⚠️ 有効なマルチスケール特徴が不足: {len(valid_features)}個")
                    multiscale_features = None
                    
            except Exception as ms_error:
                print(f"  ❌ マルチスケール特徴抽出エラー: {str(ms_error)}")
                multiscale_features = None
                
        except Exception as enc_error:
            print(f"  ❌ 基本エンコードエラー: {str(enc_error)}")
            raise RuntimeError(f"画像エンコードに失敗しました: {str(enc_error)}")
        
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
        
        # マルチスケール対応: SAM2標準解像度（64x64）に統一
        # Webリサーチ結果: SAM2 MaskDecoderは64x64解像度で動作
        # image embeddings: 1x256x64x64, dense_prompt_embeddingsも同じ解像度が必要
        if multiscale_features is not None and len(multiscale_features) > 0:
            import torch.nn.functional as F
            
            # SAM2標準解像度（64x64）に統一
            sam2_standard_size = (64, 64)
            
            # dense_embeddingsをSAM2標準解像度に調整
            if dense_embeddings.shape[-2:] != sam2_standard_size:
                dense_embeddings = F.interpolate(
                    dense_embeddings, 
                    size=sam2_standard_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                print(f"  🔧 dense_embeddings SAM2標準解像度調整: -> {sam2_standard_size}")
            
            # image_embeddingsもSAM2標準解像度に調整（必要に応じて）
            if image_embeddings.dim() == 4 and image_embeddings.shape[-2:] != sam2_standard_size:
                print(f"  ⚠️ image_embeddings解像度調整: {image_embeddings.shape[-2:]} -> {sam2_standard_size}")
                image_embeddings = F.interpolate(
                    image_embeddings, 
                    size=sam2_standard_size, 
                    mode='bilinear', 
                    align_corners=False
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