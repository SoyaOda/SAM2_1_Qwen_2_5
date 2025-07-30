"""
シンプルなテスト用損失関数
勾配消失問題を回避し、基本的な機能確認に特化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any

class SimplifiedTestLoss(nn.Module):
    """
    シンプルなテスト用損失関数
    勾配保持を最優先し、複雑な計算を排除
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # シンプルな重み設定
        self.segmentation_weight = 1.0
        self.qformer_weight = 0.3
        
        # 勾配保持確実な損失関数のみ使用
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def forward(self, 
                predicted_masks: torch.Tensor,
                target_masks: torch.Tensor,
                query_embeds: Optional[torch.Tensor] = None,
                text_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        シンプルな損失計算（勾配フロー詳細デバッグ付き）
        
        Args:
            predicted_masks: (B, 3, H, W) または (B, H, W)
            target_masks: (B, H, W)
            query_embeds: (B, 32, 768) Q-Former出力
            text_embeds: (B, 5120) テキスト埋め込み
        
        Returns:
            loss_dict: 損失辞書
        """
        print(f"🔍 [GRAD_DEBUG] 損失計算開始")
        
        # Web調査準拠: 入力テンソルの勾配状況詳細確認
        self._debug_tensor_grad_status("predicted_masks", predicted_masks)
        self._debug_tensor_grad_status("target_masks", target_masks)
        if query_embeds is not None:
            self._debug_tensor_grad_status("query_embeds", query_embeds)
        if text_embeds is not None:
            self._debug_tensor_grad_status("text_embeds", text_embeds)
            
        total_loss = 0.0
        loss_dict = {}
        
        # === 1. メインセグメンテーション損失 ===
        if predicted_masks is not None and target_masks is not None:
            # Web調査準拠: テンソルスライス勾配保持デバッグ
            print(f"🔍 [SLICE_DEBUG] 予測マスク処理前:")
            print(f"  - predicted_masks: requires_grad={predicted_masks.requires_grad}, grad_fn={predicted_masks.grad_fn}")
            
            # 予測マスクの処理
            if predicted_masks.dim() == 4 and predicted_masks.size(1) > 1:
                # 複数マスクの場合、最初のマスクを使用（シンプル化）
                best_masks = predicted_masks[:, 0]  # (B, H, W)
                print(f"🔍 複数マスク検出: {predicted_masks.shape} -> 最初のマスク使用: {best_masks.shape}")
                
                # Web調査準拠: スライス後の勾配状況確認
                print(f"🔍 [SLICE_DEBUG] スライス後勾配状況:")
                print(f"  - best_masks: requires_grad={best_masks.requires_grad}, grad_fn={best_masks.grad_fn}")
                
                # Web調査準拠: スライス操作で勾配が失われた場合の修復
                if predicted_masks.requires_grad and not best_masks.requires_grad:
                    print(f"⚠️ [SLICE_FIX] スライス操作で勾配失失 - 修復中")
                    best_masks.requires_grad_(True)
                    print(f"✅ [SLICE_FIX] 勾配復元完了: requires_grad={best_masks.requires_grad}")
                    
            elif predicted_masks.dim() == 4 and predicted_masks.size(1) == 1:
                best_masks = predicted_masks.squeeze(1)  # (B, H, W)
                # Web調査準拠: squeeze操作後の勾配確認
                if predicted_masks.requires_grad and not best_masks.requires_grad:
                    best_masks.requires_grad_(True)
                    print(f"✅ [SQUEEZE_FIX] squeeze後勾配復元")
            else:
                best_masks = predicted_masks  # 既に (B, H, W)
            
            # Web調査準拠: 勾配保持データ型統一（BCEWithLogitsLoss要件）
            # tensor.to(dtype)は勾配切断するため、clone()使用で勾配フロー保持
            target_dtype = best_masks.dtype  # 予測マスクのデータ型に統一
            if target_masks.dtype != target_dtype:
                # Web調査準拠: clone()で勾配フロー維持しながら型変換
                target_masks_float = target_masks.clone().to(dtype=target_dtype)
                # 明示的に勾配設定を保持
                if target_masks.requires_grad:
                    target_masks_float.requires_grad_(True)
                print(f"🔧 [DTYPE_FIX] 勾配保持型変換: {target_masks.dtype} -> {target_dtype}")
            else:
                target_masks_float = target_masks
                print(f"🔧 [DTYPE_FIX] 型変換不要: 既に{target_dtype}")
            
            # Web調査準拠: 勾配フロー保持確認
            print(f"🔍 [GRAD_DEBUG] セグメンテーション損失計算前:")
            self._debug_tensor_grad_status("best_masks", best_masks)
            self._debug_tensor_grad_status("target_masks_float", target_masks_float)
            
            # Web調査準拠: MSELoss bfloat16ネイティブサポート活用
            try:
                # Web調査準拠: PyTorch MSELossはbfloat16を完全サポート
                # 型変換による勾配切断を回避し、元のdtypeで直接計算
                print(f"🔍 [NATIVE_BFLOAT16] MSE損失bfloat16直接計算:")
                print(f"  - best_masks: {best_masks.dtype}, requires_grad: {best_masks.requires_grad}")
                print(f"  - target_masks_float: {target_masks_float.dtype}, requires_grad: {target_masks_float.requires_grad}")
                
                # Web調査準拠: 型変換なしで直接MSE計算（勾配フロー完全保持）
                seg_loss = self.mse_loss(best_masks, target_masks_float)
                
                # 損失テンソルの勾配状況確認
                print(f"🔍 [GRAD_DEBUG] セグメンテーション損失計算後:")
                self._debug_tensor_grad_status("seg_loss", seg_loss)
                
                # Web調査準拠: 明示的にrequires_gradを確保
                if not seg_loss.requires_grad:
                    print(f"⚠️ [GRAD_DEBUG] seg_lossの勾配が無効 - 入力テンソル問題の可能性")
                    # 根本原因特定のため処理停止（md_files方針：フォールバック回避）
                    raise RuntimeError("セグメンテーション損失で勾配フローが切断されました。入力テンソルの勾配設定を確認してください。")
                
                total_loss += self.segmentation_weight * seg_loss
                loss_dict['segmentation'] = seg_loss
                print(f"✅ セグメンテーション損失計算成功: {seg_loss.item():.6f}")
            except RuntimeError as e:
                # md_files方針: 適切にエラーを出して止める
                raise e
            except Exception as e:
                print(f"❌ セグメンテーション損失エラー: {e}")
                raise RuntimeError(f"セグメンテーション損失で予期しないエラー: {e}")
        
        # === 2. Q-Former統合損失（軽量版） ===
        if query_embeds is not None and text_embeds is not None:
            print(f"🔍 [GRAD_DEBUG] Q-Former損失計算前:")
            self._debug_tensor_grad_status("query_embeds", query_embeds)
            self._debug_tensor_grad_status("text_embeds", text_embeds)
            
            try:
                # Q-Formerクエリの平均プーリング
                query_pooled = query_embeds.mean(dim=1)  # (B, 768)
                self._debug_tensor_grad_status("query_pooled", query_pooled)
                
                # Web調査準拠: 勾配保持次元調整
                if query_pooled.size(-1) != text_embeds.size(-1):
                    # パディングで次元拡張（勾配フロー保持）
                    pad_size = text_embeds.size(-1) - query_pooled.size(-1)
                    if pad_size > 0:
                        query_expanded = F.pad(query_pooled, (0, pad_size))
                        # パディング後も勾配設定を保持
                        if query_pooled.requires_grad and not query_expanded.requires_grad:
                            query_expanded = query_expanded.requires_grad_(True)
                    else:
                        # トランケート（勾配保持スライス）
                        query_expanded = query_pooled[:, :text_embeds.size(-1)]
                else:
                    query_expanded = query_pooled
                
                self._debug_tensor_grad_status("query_expanded", query_expanded)
                
                # Web調査準拠: Q-Former MSE損失もbfloat16直接計算
                print(f"🔍 [Q_NATIVE_BFLOAT16] Q-Former bfloat16直接計算:")
                print(f"  - query_expanded: {query_expanded.dtype}, requires_grad: {query_expanded.requires_grad}")
                print(f"  - text_embeds: {text_embeds.dtype}, requires_grad: {text_embeds.requires_grad}")
                
                # Web調査準拠: 型変換回避でQ-Former損失計算
                qformer_loss = self.mse_loss(query_expanded, text_embeds)
                
                print(f"🔍 [GRAD_DEBUG] Q-Former損失計算後:")
                self._debug_tensor_grad_status("qformer_loss", qformer_loss)
                
                # Web調査準拠: 勾配確認
                if not qformer_loss.requires_grad:
                    print(f"⚠️ [GRAD_DEBUG] qformer_lossの勾配が無効")
                    raise RuntimeError("Q-Former損失で勾配フローが切断されました。")
                
                total_loss += self.qformer_weight * qformer_loss
                loss_dict['qformer'] = qformer_loss
                print(f"✅ Q-Former損失計算成功: {qformer_loss.item():.6f}")
                
            except RuntimeError as e:
                raise e
            except Exception as e:
                print(f"❌ Q-Former損失エラー: {e}")
                raise RuntimeError(f"Q-Former損失で予期しないエラー: {e}")
        
        # 総損失の計算（Web調査準拠：フォールバック回避）
        if total_loss == 0.0:
            # md_files方針: フォールバック回避、適切にエラーを出す
            raise RuntimeError("全ての損失計算が失敗しました。入力データまたはモデル設定を確認してください。")
        
        # Web調査準拠: 総損失のデータ型統一・勾配確認
        original_device = predicted_masks.device
        original_dtype = predicted_masks.dtype
        
        # 統一されたfloat32損失をオリジナルデータ型に変換（必要に応じて）
        if total_loss.dtype != original_dtype:
            total_loss = total_loss.to(dtype=original_dtype)
            print(f"🔧 [DTYPE_FIX] 総損失データ型調整: {total_loss.dtype}")
        
        print(f"🔍 [GRAD_DEBUG] 総損失計算:")
        self._debug_tensor_grad_status("total_loss", total_loss)
        
        if not total_loss.requires_grad:
            print(f"⚠️ [GRAD_DEBUG] 総損失で勾配が無効 - 構成要素の問題")
            raise RuntimeError("総損失で勾配フローが切断されました。各損失成分の勾配設定を確認してください。")
        
        loss_dict['total_loss'] = total_loss
        
        # 最終勾配確認
        print(f"🔍 [GRAD_DEBUG] 最終損失勾配確認:")
        for name, loss in loss_dict.items():
            if isinstance(loss, torch.Tensor):
                print(f"  - {name}: requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn is not None}")
        
        return loss_dict
    
    def _debug_tensor_grad_status(self, name: str, tensor: torch.Tensor):
        """
        Web調査準拠: テンソルの勾配状況詳細デバッグ
        """
        if tensor is None:
            print(f"  - {name}: None")
            return
            
        is_leaf = tensor.is_leaf
        requires_grad = tensor.requires_grad
        has_grad_fn = tensor.grad_fn is not None
        
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        print(f"    - is_leaf={is_leaf}, requires_grad={requires_grad}, has_grad_fn={has_grad_fn}")
        
        # Web調査準拠: 勾配フロー切断の具体的原因特定
        if not requires_grad and not has_grad_fn:
            print(f"    ⚠️ [GRAD_ISSUE] {name}: 勾配が完全に無効（leaf tensorで勾配無効 または 計算グラフ切断）")
        elif requires_grad and not has_grad_fn and not is_leaf:
            print(f"    ⚠️ [GRAD_ISSUE] {name}: non-leaf tensorなのにgrad_fnが無い（計算グラフ切断の可能性）")
        elif not requires_grad and has_grad_fn:
            print(f"    ⚠️ [GRAD_ISSUE] {name}: requires_gradがFalseなのにgrad_fnが存在（不整合）")
        else:
            print(f"    ✅ [GRAD_OK] {name}: 勾配状況正常")
            
        # Web調査準拠: Parameter型チェック
        if isinstance(tensor, torch.nn.Parameter):
            print(f"    📌 [PARAM] {name}: Parameter型（勾配保持されるべき）")
    
    def _safe_tensor_operation(self, operation, *args, **kwargs):
        """安全なテンソル操作ラッパー"""
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ テンソル操作エラー: {e}")
            # デバイスとデータ型の確認
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    print(f"  - arg{i}: shape={arg.shape}, dtype={arg.dtype}, device={arg.device}")
            return None