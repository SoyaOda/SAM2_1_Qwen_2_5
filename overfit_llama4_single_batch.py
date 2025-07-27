#!/usr/bin/env python3
"""
LISA-Llama4 プロジェクト: 単一バッチ過学習検証スクリプト
Web調査に基づく実証済み分散方法を採用 (accelerateライブラリを回避)

実装方針:
- HuggingFace公式推奨のdevice_map="auto"分散方法
- accelerateのCPUオフロードを回避
- 固定データでの過学習による学習能力検証
"""

import sys
import os
import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話モードに設定
import matplotlib.pyplot as plt
import numpy as np

# PEFT関連
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor, 
    Llama4ForConditionalGeneration,
    BitsAndBytesConfig
)

# プロジェクト固有のインポート
sys.path.append('.')
from utils.utils import (
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OverfitConfig:
    """過学習テスト用設定"""
    model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    max_length: int = 256  # 過学習テスト用に短縮
    num_epochs: int = 10
    learning_rate: float = 1e-4
    
    # LoRA設定 (Webリサーチに基づく実証済み設定)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # GPU分散設定 (HuggingFace公式推奨方法)
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"  # Web調査で推奨されている設定
    use_4bit: bool = True
    
    # 過学習判定設定
    target_loss: float = 0.1  # この値以下で過学習成功とみなす
    
    # 出力設定
    output_dir: str = "./llama4_overfit_results"
    
    def __post_init__(self):
        """デフォルト設定の初期化"""
        if self.lora_target_modules is None:
            # Llama4で実証済みのターゲットモジュール
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

class TensorJSONEncoder(json.JSONEncoder):
    """Tensorオブジェクト用のJSONエンコーダー"""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {
                "__tensor__": True,
                "data": obj.detach().cpu().tolist() if obj.numel() <= 100 else f"<Tensor shape={obj.shape}>",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device)
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)

class Llama4OverfitTest:
    """Llama4過学習テストクラス - Web調査結果に基づく実装"""
    
    def __init__(self, config: OverfitConfig):
        self.config = config
        self.results = {
            "training_logs": [],
            "loss_history": [],
            "success_metrics": {},
            "config": config.__dict__
        }
        
        # 出力ディレクトリ作成
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 固定データ
        self.fixed_data = None
        
    def setup_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Web調査に基づく最適化されたモデル・トークナイザー初期化"""
        logger.info("=== Llama4モデルとトークナイザーの初期化 ===")
        
        try:
            # 動的コンパイルを無効化してGPU分散エラーを回避
            torch.compiler.disable()
            logger.info("動的コンパイル無効化: GPU分散エラー回避のため")
            
            # 1. プロセッサー初期化 (HuggingFace公式方法)
            logger.info(f"プロセッサー初期化: {self.config.model_name}")
            processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # 2. 量子化設定 (Web調査の実証済み設定)
            quantization_config = None
            if self.config.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.torch_dtype),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("4bit量子化設定を適用")
            
            # 3. モデル初期化 (HuggingFace公式推奨方法)
            logger.info("Llama4モデル初期化開始...")
            model = Llama4ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                attn_implementation=self.config.attn_implementation,  # eager設定使用
                device_map="auto",  # HuggingFace公式推奨の分散方法
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"✓ モデル初期化完了")
            logger.info(f"  - パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"  - デバイス分散: {model.hf_device_map}")
            
            return model, processor
            
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            raise
    
    def apply_lora_config(self, model) -> Any:
        """Web調査に基づくLoRA設定適用"""
        logger.info("=== LoRA設定適用 ===")
        
        try:
            # LoRA設定作成 (実証済みパラメータ)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                use_rslora=False  # Web調査で安定性が確認された設定
            )
            
            logger.info(f"LoRA設定:")
            logger.info(f"  - rank: {self.config.lora_r}")
            logger.info(f"  - alpha: {self.config.lora_alpha}")
            logger.info(f"  - dropout: {self.config.lora_dropout}")
            logger.info(f"  - target_modules: {self.config.lora_target_modules}")
            
            # LoRA適用
            model = get_peft_model(model, lora_config)
            
            # 学習可能パラメータ統計
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info(f"✓ LoRA適用完了")
            logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
            logger.info(f"  - 全パラメータ: {total_params:,}")
            logger.info(f"  - 学習可能割合: {100 * trainable_params / total_params:.3f}%")
            
            return model
            
        except Exception as e:
            logger.error(f"LoRA適用エラー: {e}")
            raise
    
    def prepare_fixed_data(self, processor) -> Dict[str, Any]:
        """固定データの準備"""
        logger.info("=== 固定データ準備 ===")
        
        # 過学習テスト用の固定メッセージ (Llama4マルチモーダル対応フォーマット)
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "What is the capital of France?"}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": "The capital of France is Paris."}
                ]
            }
        ]
        
        try:
            # Web調査に基づくLlama4プロセッサー正しい使用方法
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,   # 生成プロンプトを追加
                tokenize=True,               # 直接トークン化
                return_dict=True,            # 辞書形式で返す
                return_tensors="pt"          # PyTorchテンソルとして返す
            )
            
            logger.info(f"✓ 固定データ準備完了")
            logger.info(f"  - 入力長: {inputs['input_ids'].shape[-1]}")
            logger.info(f"  - メッセージ: {messages}")
            
            self.fixed_data = {
                "inputs": inputs,
                "messages": messages
            }
            
            return self.fixed_data
            
        except Exception as e:
            logger.error(f"固定データ準備エラー: {e}")
            raise
    
    def setup_optimizer(self, model) -> torch.optim.Optimizer:
        """オプティマイザー設定"""
        # 学習可能パラメータのみを対象
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"✓ オプティマイザー設定完了")
        logger.info(f"  - 学習率: {self.config.learning_rate}")
        logger.info(f"  - 学習可能パラメータ数: {len(trainable_params)}")
        
        return optimizer
    
    def run_training_epoch(self, model, optimizer, epoch: int) -> float:
        """単一エポックの学習実行"""
        model.train()
        optimizer.zero_grad()
        
        # 固定データを適切なデバイスに移動
        inputs = self.fixed_data["inputs"]
        first_device = next(iter(model.hf_device_map.values()))
        inputs = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # 順伝播
        outputs = model(**inputs)
        
        # Llama4では常に手動で損失を計算（outputs.lossが正しくない）
        # logitsを取得
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'loss') and isinstance(outputs.loss, dict) and 'logits' in outputs.loss:
            logits = outputs.loss['logits']
        else:
            raise ValueError("Could not find logits in outputs")
        
        # 言語モデリング損失を手動計算
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # 逆伝播
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        optimizer.step()
        
        # 損失値を安全に取得
        if isinstance(loss, dict):
            # 辞書形式の場合、'loss'キーを探す
            loss_value = loss.get('loss', loss.get('total_loss', list(loss.values())[0] if loss else 0))
            if hasattr(loss_value, 'item'):
                loss_value = loss_value.item()
            else:
                loss_value = float(loss_value)
        else:
            # テンソル形式の場合
            loss_value = loss.item()
        
        # ログ記録
        log_entry = {
            "epoch": epoch,
            "loss": loss_value,
            "learning_rate": self.config.learning_rate
        }
        
        self.results["training_logs"].append(log_entry)
        self.results["loss_history"].append(loss_value)
        
        return loss_value
    
    def run_overfit_test(self, model, processor) -> Dict[str, Any]:
        """過学習テスト実行"""
        logger.info("=== 過学習テスト開始 ===")
        
        # オプティマイザー設定
        optimizer = self.setup_optimizer(model)
        
        # 学習ループ
        initial_loss = None
        final_loss = None
        
        for epoch in range(self.config.num_epochs):
            loss = self.run_training_epoch(model, optimizer, epoch + 1)
            
            if epoch == 0:
                initial_loss = loss
            final_loss = loss
            
            logger.info(f"エポック {epoch + 1}/{self.config.num_epochs}: 損失 = {loss:.6f}")
            
            # 早期停止判定
            if loss < self.config.target_loss:
                logger.info(f"✓ 目標損失{self.config.target_loss}を達成！エポック{epoch + 1}で早期停止")
                break
            
            # メモリクリーンアップ
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # 過学習成功判定
        loss_reduction = initial_loss - final_loss if initial_loss else 0
        loss_reduction_ratio = loss_reduction / initial_loss if initial_loss else 0
        
        overfit_success = (
            final_loss < self.config.target_loss or
            loss_reduction_ratio > 0.5  # 50%以上損失が減少
        )
        
        success_metrics = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_reduction,
            "loss_reduction_ratio": loss_reduction_ratio,
            "target_loss_achieved": final_loss < self.config.target_loss,
            "significant_improvement": loss_reduction_ratio > 0.5,
            "overfit_success": overfit_success
        }
        
        self.results["success_metrics"] = success_metrics
        
        logger.info("=== 過学習テスト完了 ===")
        logger.info(f"  - 初期損失: {initial_loss:.6f}")
        logger.info(f"  - 最終損失: {final_loss:.6f}")
        logger.info(f"  - 損失減少: {loss_reduction:.6f} ({loss_reduction_ratio:.1%})")
        logger.info(f"  - 過学習成功: {'✓' if overfit_success else '✗'}")
        
        return success_metrics
    
    def create_loss_plot(self) -> str:
        """損失曲線プロット作成"""
        if not self.results["loss_history"]:
            return ""
        
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.results["loss_history"]) + 1), 
                    self.results["loss_history"], 
                    'b-', linewidth=2, label='Training Loss')
            
            # 目標損失線
            plt.axhline(y=self.config.target_loss, color='r', linestyle='--', 
                       label=f'Target Loss ({self.config.target_loss})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('LISA-Llama4 Overfit Test: Loss Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 最小値にマーク
            min_loss_epoch = np.argmin(self.results["loss_history"]) + 1
            min_loss_value = min(self.results["loss_history"])
            plt.plot(min_loss_epoch, min_loss_value, 'ro', markersize=8, 
                    label=f'Min Loss: {min_loss_value:.4f}')
            plt.legend()
            
            plot_file = Path(self.config.output_dir) / "loss_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 損失曲線プロット保存: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"プロット作成エラー: {e}")
            return ""
    
    def save_results(self) -> str:
        """結果保存"""
        output_file = Path(self.config.output_dir) / "llama4_overfit_test_results.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, cls=TensorJSONEncoder)
            
            logger.info(f"✓ テスト結果を保存: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
            return ""
    
    def run_test(self) -> Dict[str, Any]:
        """テスト実行"""
        logger.info("🚀 LISA-Llama4 過学習テスト開始")
        logger.info(f"設定: {self.config}")
        
        try:
            # 1. モデル・トークナイザー初期化
            model, processor = self.setup_model_and_tokenizer()
            
            # 2. LoRA適用
            model = self.apply_lora_config(model)
            
            # 3. 固定データ準備
            self.prepare_fixed_data(processor)
            
            # 4. 過学習テスト実行
            success_metrics = self.run_overfit_test(model, processor)
            
            # 5. 損失曲線プロット作成
            plot_file = self.create_loss_plot()
            if plot_file:
                self.results["plot_file"] = plot_file
            
            # 6. 結果保存
            results_file = self.save_results()
            
            logger.info("✅ 過学習テスト完了!")
            return self.results
            
        except Exception as e:
            logger.error(f"テスト実行エラー: {e}")
            self.results["error"] = str(e)
            return self.results

def main():
    """メイン実行関数"""
    config = OverfitConfig()
    tester = Llama4OverfitTest(config)
    
    results = tester.run_test()
    
    # 結果サマリー表示
    if "error" not in results:
        print("\n" + "="*50)
        print("🎉 LISA-Llama4 過学習テスト 完了!")
        print("="*50)
        
        if "success_metrics" in results:
            metrics = results["success_metrics"]
            print(f"📊 テスト結果:")
            print(f"  - 初期損失: {metrics.get('initial_loss', 'N/A'):.6f}")
            print(f"  - 最終損失: {metrics.get('final_loss', 'N/A'):.6f}")
            print(f"  - 損失減少率: {metrics.get('loss_reduction_ratio', 0):.1%}")
            print(f"  - 目標損失達成: {'✓' if metrics.get('target_loss_achieved', False) else '✗'}")
            print(f"  - 過学習成功: {'✓' if metrics.get('overfit_success', False) else '✗'}")
        
        if "plot_file" in results:
            print(f"\n📈 損失曲線: {results['plot_file']}")
            
        # 総合判定
        overfit_success = results.get("success_metrics", {}).get("overfit_success", False)
        if overfit_success:
            print("\n🎯 結論: モデルは正常に学習能力を示しました！")
            print("   次は実際のデータセットでの学習に進むことができます。")
        else:
            print("\n⚠️  結論: 過学習が確認できませんでした。")
            print("   設定やデータを見直す必要があります。")
    else:
        print(f"❌ テスト失敗: {results['error']}")

if __name__ == "__main__":
    main() 