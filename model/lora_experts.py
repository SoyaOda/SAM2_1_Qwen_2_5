# model/lora_experts.py
"""
LoRAエキスパート統合によるマルチモーダル適応

o3-modification20250727.mdに基づく実装:
- 低ランク適応（LoRA）モジュール
- MoEルーターによる動的エキスパート選択
- マルチモーダル対応の効率的適応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any, Union
import math
from collections import OrderedDict


class LoRALayer(nn.Module):
    """
    低ランク適応（LoRA）レイヤー
    
    W' = W + ΔW = W + BA^T の形式で重み行列を適応
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights
        self.merged = False
        
        # スケーリング係数（PEFT準拠）
        self.scaling = self.alpha / self.rank
        
        # 低ランク分解行列
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # ドロップアウト
        if dropout > 0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        # 重みの初期化
        self.reset_parameters()
    
    def reset_parameters(self):
        """パラメータの初期化（Kaiming初期化）"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, base_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        LoRA適応の適用
        
        Args:
            x: 入力テンソル
            base_output: ベースレイヤーの出力（提供される場合）
        
        Returns:
            適応された出力
        """
        # デバイス統一：lora_A, lora_Bを入力と同じデバイスに移動
        input_device = x.device
        if self.lora_A.device != input_device:
            self.lora_A.data = self.lora_A.data.to(input_device)
        if self.lora_B.device != input_device:
            self.lora_B.data = self.lora_B.data.to(input_device)
        
        # LoRA補正の計算: x @ A^T @ B^T * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        if base_output is not None:
            return base_output + lora_output
        else:
            return lora_output
    
    def merge(self):
        """LoRA重みをベース重みにマージ（推論高速化用）"""
        if self.merge_weights and not self.merged:
            # W' = W + BA * scaling
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge(self):
        """マージした重みを元に戻す"""
        if self.merge_weights and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


class LoRALinear(nn.Linear):
    """
    LoRA適応付きLinearレイヤー
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False
        
        # LoRAレイヤーの追加
        self.lora = LoRALayer(
            in_features,
            out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights
        )
        
        # fan_in_fan_outの場合は重み行列を転置
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA適応付きforward
        """
        # デバイス統一：LoRAレイヤーのパラメータを入力と同じデバイスに移動
        input_device = x.device
        if hasattr(self.lora, 'lora_A') and self.lora.lora_A.device != input_device:
            self.lora.lora_A.data = self.lora.lora_A.data.to(input_device)
        if hasattr(self.lora, 'lora_B') and self.lora.lora_B.device != input_device:
            self.lora.lora_B.data = self.lora.lora_B.data.to(input_device)
        
        # ベースのLinear計算
        if self.fan_in_fan_out:
            base_output = F.linear(x, self.weight.T, self.bias)
        else:
            base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA補正を追加
        return self.lora(x, base_output)
    
    def merge_weights(self):
        """推論用に重みをマージ"""
        if self.merge_weights and not self.merged:
            if self.fan_in_fan_out:
                self.weight.data += (self.lora.lora_B @ self.lora.lora_A).T * self.lora.scaling
            else:
                self.weight.data += (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.merged = True
    
    def unmerge_weights(self):
        """マージを元に戻す"""
        if self.merge_weights and self.merged:
            if self.fan_in_fan_out:
                self.weight.data -= (self.lora.lora_B @ self.lora.lora_A).T * self.lora.scaling
            else:
                self.weight.data -= (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.merged = False


class MultiModalityRouter(nn.Module):
    """
    マルチモーダルLoRAルーター
    
    入力モダリティに応じて適切なLoRAエキスパートを選択・組み合わせ
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        top_k: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = min(top_k, num_experts)
        
        # モダリティ判定ネットワーク
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # ノイズ生成（学習時のロバスト性向上）
        self.noise_std = 0.1
        
        # ミュート: MultiModalityRouter初期化ログ
        # print(f"✅ MultiModalityRouter初期化")
        # print(f"  - エキスパート数: {num_experts}")
        # print(f"  - Top-K: {top_k}")
    
    def forward(
        self,
        x: torch.Tensor,
        modality_hint: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        ルーティング重みの計算
        
        Args:
            x: 入力特徴 [B, seq_len, dim] or [B, dim]
            modality_hint: モダリティヒント（オプション）
            return_weights: 重みも返すか
        
        Returns:
            routing_weights: ルーティング重み [B, num_experts]
            (optional) routing_logits: ルーティングロジット
        """
        # 入力特徴の集約（シーケンスの場合）
        if x.dim() == 3:
            # [B, seq_len, dim] -> [B, dim]
            routing_input = x.mean(dim=1)
        else:
            routing_input = x
        
        # モダリティヒントがある場合は結合
        if modality_hint is not None:
            routing_input = torch.cat([routing_input, modality_hint], dim=-1)
        
        # ルーティングロジットの計算
        routing_logits = self.router(routing_input)
        
        # 学習時はノイズを追加（Mixture of Expertsの標準的手法）
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(routing_logits) * self.noise_std
            routing_logits = routing_logits + noise
        
        # Top-K選択
        if self.top_k < self.num_experts:
            # Top-Kの値とインデックスを取得
            topk_values, topk_indices = torch.topk(routing_logits, self.top_k, dim=-1)
            
            # Top-K以外をマスク
            routing_weights = torch.zeros_like(routing_logits)
            routing_weights.scatter_(-1, topk_indices, 
                                   F.softmax(topk_values / self.temperature, dim=-1))
        else:
            # 全エキスパート使用
            routing_weights = F.softmax(routing_logits / self.temperature, dim=-1)
        
        if return_weights:
            return routing_weights, routing_logits
        else:
            return routing_weights


class LoRAExpertMoE(nn.Module):
    """
    LoRAエキスパートのMixture of Experts
    
    複数のLoRAエキスパートを管理し、ルーターの判断に基づいて
    動的に組み合わせて適用
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        num_experts: int = 4,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        router_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.num_experts = num_experts
        self.rank = rank
        
        # 入出力次元の取得
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            raise ValueError("現在はLinearレイヤーのみサポート")
        
        # エキスパートLoRA群
        self.experts = nn.ModuleList([
            LoRALayer(
                self.in_features,
                self.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            ) for _ in range(num_experts)
        ])
        
        # ルーター
        router_config = router_config or {}
        self.router = MultiModalityRouter(
            input_dim=self.in_features,
            num_experts=num_experts,
            **router_config
        )
        
        # ルーターをベースレイヤーと同じデバイスに移動
        if hasattr(base_layer, 'weight') and base_layer.weight is not None:
            base_device = base_layer.weight.device
            base_dtype = base_layer.weight.dtype
            self.router = self.router.to(device=base_device, dtype=base_dtype)
            # ミュート: ルーターデバイス移動ログ
            # print(f"  ✅ ルーターをデバイス移動: {base_device}, dtype: {base_dtype}")
        
        # エキスパート名（デバッグ用）
        self.expert_names = [f"expert_{i}" for i in range(num_experts)]
        
        # ミュート: LoRAExpertMoE初期化ログ
        # print(f"✅ LoRAExpertMoE初期化")
        # print(f"  - ベースレイヤー: {base_layer.__class__.__name__}")
        # print(f"  - エキスパート数: {num_experts}")
        # print(f"  - ランク: {rank}")
    
    def forward(
        self,
        x: torch.Tensor,
        modality_hint: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        MoE LoRAのforward処理
        
        Args:
            x: 入力テンソル
            modality_hint: モダリティヒント
        
        Returns:
            適応された出力
        """
        # ベースレイヤーの出力
        base_output = self.base_layer(x)
        
        # ルーターのデバイス統一を確保
        input_device = x.device
        if next(self.router.parameters()).device != input_device:
            self.router = self.router.to(input_device)
        
        # ルーティング重みの計算
        routing_weights = self.router(x, modality_hint)  # [B, num_experts]
        
        # 各エキスパートの出力を計算（デバイス統一付き）
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # エキスパートのLoRAパラメータをデバイス統一
            if expert.lora_A.device != input_device:
                expert.lora_A.data = expert.lora_A.data.to(input_device)
            if expert.lora_B.device != input_device:
                expert.lora_B.data = expert.lora_B.data.to(input_device)
            
            expert_output = expert(x)  # LoRA補正のみ
            expert_outputs.append(expert_output)
        
        # エキスパート出力をスタック
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, B, ...]
        
        # 重み付き結合
        # routing_weightsを適切な形状に拡張
        weight_shape = [self.num_experts] + [1] * (expert_outputs.dim() - 1)
        routing_weights = routing_weights.T.view(weight_shape)  # [num_experts, 1, ...]
        
        # 重み付き平均
        combined_lora = (expert_outputs * routing_weights).sum(dim=0)
        
        # ベース出力とLoRA補正を結合
        return base_output + combined_lora
    
    def set_expert_names(self, names: List[str]):
        """エキスパート名の設定（例: ["RGB", "Depth", "IR", "General"]）"""
        if len(names) != self.num_experts:
            raise ValueError(f"名前の数({len(names)})がエキスパート数({self.num_experts})と一致しません")
        self.expert_names = names


def inject_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    use_moe: bool = False,
    num_experts: int = 4,
    router_config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    モデルの指定モジュールにLoRAを注入（デバッグ修正ルール対応）
    
    Args:
        model: 対象モデル
        target_modules: LoRAを適用するモジュール名のリスト
        rank: LoRAランク
        alpha: LoRAアルファ
        dropout: ドロップアウト率
        use_moe: MoEを使用するか
        num_experts: エキスパート数（MoE使用時）
        router_config: ルーター設定（MoE使用時）
    
    Returns:
        LoRA適用済みモデル
    """
    print(f"🔍 LoRA注入開始デバッグ:")
    print(f"  - 対象モデル: {type(model).__name__}")
    print(f"  - ターゲットモジュール: {target_modules}")
    print(f"  - ランク: {rank}, アルファ: {alpha}")
    print(f"  - MoE使用: {use_moe}, エキスパート数: {num_experts}")
    
    # GPU メモリ状況確認
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  - GPU メモリ (注入前): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    # モジュール名とモジュールのマッピングを作成
    modules_to_replace = []
    all_linear_modules = []
    
    for name, module in model.named_modules():
        # 全Linearモジュールを記録（デバッグ用）
        if isinstance(module, nn.Linear):
            all_linear_modules.append(name)
        
        # ターゲットモジュールに一致するか確認
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))
                # ミュート: ターゲット発見ログ
                # print(f"  ✅ ターゲット発見: {name} -> {module}")
                break
    
    print(f"  - 全Linearモジュール数: {len(all_linear_modules)}")
    print(f"  - LoRA注入対象数: {len(modules_to_replace)}")
    
    if len(modules_to_replace) == 0:
        print(f"  ⚠️ LoRA注入対象が見つかりません")
        print(f"  - 利用可能なLinearモジュール（最初の10個）:")
        for name in all_linear_modules[:10]:
            print(f"    - {name}")
        if len(all_linear_modules) > 10:
            print(f"    - ... 他{len(all_linear_modules)-10}個")
        return model
    
    # LoRAの注入（メモリ安全処理付き）
    injection_count = 0
    for i, (name, module) in enumerate(modules_to_replace):
        print(f"  🔧 LoRA注入中 ({i+1}/{len(modules_to_replace)}): {name}")
        
        try:
            # GPU メモリ状況を定期チェック
            if torch.cuda.is_available() and i % 5 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                if allocated > 70:  # 70GB以上の場合は警告
                    print(f"    ⚠️ 高メモリ使用量: {allocated:.2f}GB")
                    torch.cuda.empty_cache()  # キャッシュクリア
            
            # モジュールパスを分解
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            
            # 親モジュールを取得
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # LoRAモジュールで置換
            if use_moe:
                # MoE LoRA
                lora_module = LoRAExpertMoE(
                    base_layer=module,
                    num_experts=num_experts,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    router_config=router_config
                )
            else:
                # 単一LoRA
                lora_module = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=module.bias is not None
                )
                
                # 元の重みをコピー（デバイス同期）
                target_device = next(module.parameters()).device
                lora_module = lora_module.to(target_device)
                
                with torch.no_grad():
                    lora_module.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        lora_module.bias.data.copy_(module.bias.data)
            
            # モジュールを置換
            setattr(parent, child_name, lora_module)
            injection_count += 1
            print(f"    ✅ 注入完了: {name}")
            
        except Exception as e:
            print(f"    ❌ LoRA注入エラー ({name}): {str(e)}")
            print(f"      - エラータイプ: {type(e).__name__}")
            # エラーが発生しても続行（デバッグ修正ルール）
            continue
    
    # 最終メモリ状況
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  - GPU メモリ (注入後): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    print(f"✅ LoRA注入完了: {injection_count}/{len(modules_to_replace)}個のモジュール")
    return model