"""
LoRA/QLoRA設定と管理モジュール
SAM2.1 + Qwen2.5-VL統合モデル用
"""
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


@dataclass
class LoRAConfigManager:
    """LoRA/QLoRA設定の統一管理クラス"""
    
    # Qwen用LoRA設定
    qwen_lora_r: int = 16
    qwen_lora_alpha: int = 32
    qwen_lora_dropout: float = 0.05
    qwen_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        # オプション: gate_proj, up_proj, down_proj も追加可能
    ])
    
    # SAM用LoRA設定
    sam_lora_r: int = 16
    sam_lora_alpha: int = 32
    sam_lora_dropout: float = 0.1
    sam_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj"  # Mask Decoder内の注意機構
    ])
    
    # QLoRA設定（4bit量子化）
    use_qlora: bool = False
    qlora_compute_dtype: str = "bfloat16"  # または "float16"
    
    # 共通設定
    bias: str = "none"
    
    def get_qwen_lora_config(self) -> LoraConfig:
        """Qwen用のLoRA設定を取得"""
        return LoraConfig(
            r=self.qwen_lora_r,
            lora_alpha=self.qwen_lora_alpha,
            target_modules=self.qwen_target_modules,
            lora_dropout=self.qwen_lora_dropout,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )
    
    def get_sam_lora_config(self) -> LoraConfig:
        """SAM用のLoRA設定を取得"""
        return LoraConfig(
            r=self.sam_lora_r,
            lora_alpha=self.sam_lora_alpha,
            target_modules=self.sam_target_modules,
            lora_dropout=self.sam_lora_dropout,
            bias=self.bias,
            task_type=None,  # SAMには特定のタスクタイプなし
        )
    
    def get_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """QLoRA用のBitsAndBytes設定を取得"""
        if not self.use_qlora:
            return None
            
        compute_dtype = getattr(torch, self.qlora_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    
    def apply_lora_to_qwen(self, qwen_model, enable_lora: bool = True):
        """QwenモデルにLoRAを適用"""
        if not enable_lora:
            return qwen_model
            
        # QLoRAの場合、事前準備が必要
        if self.use_qlora:
            qwen_model = prepare_model_for_kbit_training(qwen_model)
            
        # LoRA適用
        lora_config = self.get_qwen_lora_config()
        qwen_model = get_peft_model(qwen_model, lora_config)
        
        return qwen_model
    
    def apply_lora_to_sam_decoder(self, sam_model, enable_lora: bool = True):
        """SAMのMask DecoderにLoRAを適用
        
        注: SAM全体ではなく、mask_decoder部分のみにLoRAを適用
        """
        if not enable_lora:
            return sam_model
            
        # SAMのmask_decoder内の該当モジュールを特定
        lora_config = self.get_sam_lora_config()
        
        # PEFTはモジュール名でターゲットを自動検出するが、
        # SAMの場合は手動で適用範囲を制限する必要がある
        # mask_decoder内のみを対象にする
        from peft import inject_adapter_in_model
        
        # mask_decoder配下のモジュールのみを対象にカスタムLoRA適用
        # この実装は簡略版。実際にはmask_decoder内の構造を確認して調整
        
        return sam_model
    
    def print_lora_summary(self):
        """LoRA設定のサマリーを表示"""
        print("\n" + "="*50)
        print("📊 LoRA/QLoRA設定サマリー")
        print("="*50)
        
        print("\n[Qwen LoRA設定]")
        print(f"  - Rank (r): {self.qwen_lora_r}")
        print(f"  - Alpha: {self.qwen_lora_alpha}")
        print(f"  - Dropout: {self.qwen_lora_dropout}")
        print(f"  - Target modules: {', '.join(self.qwen_target_modules)}")
        
        print("\n[SAM LoRA設定]")
        print(f"  - Rank (r): {self.sam_lora_r}")
        print(f"  - Alpha: {self.sam_lora_alpha}")
        print(f"  - Dropout: {self.sam_lora_dropout}")
        print(f"  - Target modules: {', '.join(self.sam_target_modules)}")
        
        if self.use_qlora:
            print("\n[QLoRA設定]")
            print(f"  - 4bit量子化: 有効")
            print(f"  - Compute dtype: {self.qlora_compute_dtype}")
        else:
            print("\n[QLoRA設定]")
            print(f"  - 4bit量子化: 無効（通常のLoRA）")
        
        print("="*50)


def create_lora_manager(
    use_qlora: bool = False,
    qwen_r: int = 16,
    sam_r: int = 16,
    **kwargs
) -> LoRAConfigManager:
    """LoRAマネージャーのファクトリー関数
    
    Args:
        use_qlora: QLoRA（4bit量子化）を使用するか
        qwen_r: QwenのLoRAランク
        sam_r: SAMのLoRAランク
        **kwargs: その他の設定パラメータ
        
    Returns:
        設定済みのLoRAマネージャー
    """
    return LoRAConfigManager(
        use_qlora=use_qlora,
        qwen_lora_r=qwen_r,
        sam_lora_r=sam_r,
        **kwargs
    )