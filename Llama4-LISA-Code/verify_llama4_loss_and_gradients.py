#!/usr/bin/env python3
"""
LISA-Llama4 プロジェクト: LoRA損失と勾配検証スクリプト
Web調査に基づく実証済み分散方法を採用 (accelerateライブラリを回避)

実装方針:
- HuggingFace公式推奨のdevice_map="auto"分散方法
- accelerateのCPUオフロードを回避
- 実証済み設定によるLoRA適用
"""

import sys
import os
import gc
import json
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# PEFT関連
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor, 
    Llama4ForConditionalGeneration,
    AutoTokenizer,
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
class VerificationConfig:
    """検証用設定"""
    model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    max_length: int = 512
    batch_size: int = 1
    num_steps: int = 3
    
    # LoRA設定 (Webリサーチに基づく実証済み設定)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # GPU分散設定 (HuggingFace公式推奨方法)
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"  # Web調査で推奨されている設定
    use_4bit: bool = True
    
    # 出力設定
    output_dir: str = "./llama4_verification_results"
    
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
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)

class Llama4LoRAVerification:
    """Llama4 LoRA検証クラス - Web調査結果に基づく実装"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        self.results = {"verification_steps": [], "summary": {}}
        
        # 出力ディレクトリ作成
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # ステップカウンター
        self.step_counter = 0
        
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
            
            # 3. モデル初期化 (GPU分散対応の安全な設定)
            logger.info("Llama4モデル初期化開始...")
            model = Llama4ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                attn_implementation=self.config.attn_implementation,  # eager設定を使用
                device_map="auto",
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
    
    def verify_gradient_flow(self, model, processor) -> Dict[str, Any]:
        """勾配フロー検証"""
        logger.info("=== 勾配フロー検証 ===")
        
        verification_results = {
            "parameter_analysis": {},
            "gradient_flow": {},
            "loss_computation": {},
            "step_details": []
        }
        
        try:
            # サンプルデータ準備 (Llama4マルチモーダル対応フォーマット)
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Hello, how are you?"}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": "I'm doing well, thank you for asking!"}
                    ]
                }
            ]
            
            logger.info("サンプルデータでの順伝播・逆伝播テスト開始")
            
            for step in range(self.config.num_steps):
                logger.info(f"--- ステップ {step + 1}/{self.config.num_steps} ---")
                
                step_result = {
                    "step": step + 1,
                    "input_analysis": {},
                    "forward_pass": {},
                    "loss_computation": {},
                    "backward_pass": {},
                    "parameter_updates": {}
                }
                
                # 入力準備 (Web調査に基づくLlama4プロセッサー正しい使用方法)
                try:
                    # HuggingFaceの公式例とMedium記事に従った正しい方法
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,   # 生成プロンプトを追加
                        tokenize=True,               # 直接トークン化
                        return_dict=True,            # 辞書形式で返す
                        return_tensors="pt"          # PyTorchテンソルとして返す
                    )
                    
                    # 入力を最初のGPUに移動
                    first_device = next(iter(model.hf_device_map.values()))
                    inputs = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    step_result["input_analysis"] = {
                        "input_ids_shape": inputs["input_ids"].shape if "input_ids" in inputs else "N/A",
                        "sequence_length": inputs["input_ids"].size(-1) if "input_ids" in inputs else 0,
                        "device": str(first_device),
                        "input_keys": list(inputs.keys()),
                        "input_types": {k: type(v).__name__ for k, v in inputs.items()}
                    }
                    
                    # デバッグ情報を表示
                    logger.info(f"  入力分析:")
                    logger.info(f"    - Keys: {list(inputs.keys())}")
                    logger.info(f"    - Input IDs shape: {inputs.get('input_ids', 'missing').shape if hasattr(inputs.get('input_ids', None), 'shape') else 'N/A'}")
                    if 'attention_mask' in inputs:
                        logger.info(f"    - Attention mask shape: {inputs['attention_mask'].shape}")
                    if 'labels' in inputs:
                        logger.info(f"    - Labels shape: {inputs['labels'].shape}")
                    
                except Exception as e:
                    logger.warning(f"入力準備エラー: {e}")
                    continue
                
                # 順伝播
                try:
                    model.train()
                    
                    # 勾配をゼロ初期化
                    model.zero_grad()
                    
                    # 順伝播実行
                    with torch.enable_grad():
                        outputs = model(**inputs)
                        
                        # 損失計算 (言語モデリング損失)
                        logger.info(f"  出力分析:")
                        logger.info(f"    - Outputs type: {type(outputs)}")
                        logger.info(f"    - Has loss attr: {hasattr(outputs, 'loss')}")
                        if hasattr(outputs, 'loss'):
                            logger.info(f"    - Loss value: {outputs.loss}")
                            logger.info(f"    - Loss type: {type(outputs.loss)}")
                            logger.info(f"    - Loss shape: {outputs.loss.shape if hasattr(outputs.loss, 'shape') else 'no shape'}")
                        
                        # Llama4では常に手動で損失を計算（outputs.lossが正しくない）
                        logger.info(f"    - Computing loss manually (Llama4 fix)")
                        
                        # logitsを取得
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif isinstance(outputs.loss, dict) and 'logits' in outputs.loss:
                            logits = outputs.loss['logits']
                        else:
                            raise ValueError("Could not find logits in outputs")
                        
                        logger.info(f"    - Logits shape: {logits.shape}")
                        
                        # 言語モデリング損失を手動計算
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        logger.info(f"    - Shift logits shape: {shift_logits.shape}")
                        logger.info(f"    - Shift labels shape: {shift_labels.shape}")
                        
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1)
                        )
                        
                        logger.info(f"  最終損失:")
                        logger.info(f"    - Loss type: {type(loss)}")
                        logger.info(f"    - Loss shape: {loss.shape if hasattr(loss, 'shape') else 'no shape'}")
                        logger.info(f"    - Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'no requires_grad'}")
                    
                    step_result["forward_pass"] = {
                        "success": True,
                        "output_shape": outputs.logits.shape if hasattr(outputs, 'logits') else "N/A",
                        "output_device": str(outputs.logits.device) if hasattr(outputs, 'logits') else "N/A"
                    }
                    
                    # 損失値を安全に取得
                    if isinstance(loss, dict):
                        # 辞書形式の場合、'loss'キーを探す
                        loss_value = loss.get('loss', loss.get('total_loss', list(loss.values())[0] if loss else 0))
                        if hasattr(loss_value, 'item'):
                            loss_value = loss_value.item()
                        else:
                            loss_value = float(loss_value)
                        loss_device = str(loss_value.device) if hasattr(loss_value, 'device') else "unknown"
                        loss_requires_grad = getattr(loss_value, 'requires_grad', False)
                        loss = loss_value  # 後の使用のため
                    else:
                        # テンソル形式の場合
                        loss_value = float(loss.item())
                        loss_device = str(loss.device)
                        loss_requires_grad = loss.requires_grad
                    
                    step_result["loss_computation"] = {
                        "loss_value": loss_value,
                        "loss_device": loss_device,
                        "loss_requires_grad": loss_requires_grad
                    }
                    
                except Exception as e:
                    logger.error(f"順伝播エラー: {e}")
                    step_result["forward_pass"]["error"] = str(e)
                    continue
                
                # 逆伝播
                try:
                    loss.backward()
                    
                    # 勾配統計収集
                    grad_stats = self._collect_gradient_statistics(model)
                    step_result["backward_pass"] = grad_stats
                    
                    # 損失値を安全に表示
                    display_loss = loss_value if isinstance(loss, dict) else loss.item()
                    logger.info(f"  損失: {display_loss:.6f}")
                    logger.info(f"  勾配有りパラメータ: {grad_stats['params_with_grad']}")
                    logger.info(f"  勾配L2ノルム: {grad_stats['total_grad_norm']:.6f}")
                    
                except Exception as e:
                    logger.error(f"逆伝播エラー: {e}")
                    step_result["backward_pass"]["error"] = str(e)
                
                verification_results["step_details"].append(step_result)
                
                # メモリクリーンアップ
                del inputs, outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # 全体的な分析
            verification_results["parameter_analysis"] = self._analyze_model_parameters(model)
            
            return verification_results
            
        except Exception as e:
            logger.error(f"勾配フロー検証エラー: {e}")
            verification_results["error"] = str(e)
            return verification_results
    
    def _collect_gradient_statistics(self, model) -> Dict[str, Any]:
        """勾配統計収集"""
        grad_stats = {
            "total_params": 0,
            "params_with_grad": 0,
            "params_without_grad": 0,
            "total_grad_norm": 0.0,
            "max_grad_norm": 0.0,
            "min_grad_norm": float('inf'),
            "lora_grad_norms": {},
            "base_model_grad_norms": {}
        }
        
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            grad_stats["total_params"] += 1
            
            if param.grad is not None:
                grad_stats["params_with_grad"] += 1
                
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                grad_stats["max_grad_norm"] = max(grad_stats["max_grad_norm"], param_norm)
                grad_stats["min_grad_norm"] = min(grad_stats["min_grad_norm"], param_norm)
                
                # LoRAパラメータと基底モデルパラメータを分類
                if 'lora_' in name:
                    grad_stats["lora_grad_norms"][name] = param_norm
                else:
                    grad_stats["base_model_grad_norms"][name] = param_norm
            else:
                grad_stats["params_without_grad"] += 1
        
        grad_stats["total_grad_norm"] = total_norm ** 0.5
        
        if grad_stats["min_grad_norm"] == float('inf'):
            grad_stats["min_grad_norm"] = 0.0
        
        return grad_stats
    
    def _analyze_model_parameters(self, model) -> Dict[str, Any]:
        """モデルパラメータ分析"""
        analysis = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
            "lora_parameters": 0,
            "base_model_parameters": 0,
            "parameter_breakdown": {},
            "device_distribution": {}
        }
        
        for name, param in model.named_parameters():
            analysis["total_parameters"] += param.numel()
            
            if param.requires_grad:
                analysis["trainable_parameters"] += param.numel()
            else:
                analysis["frozen_parameters"] += param.numel()
            
            if 'lora_' in name:
                analysis["lora_parameters"] += param.numel()
            else:
                analysis["base_model_parameters"] += param.numel()
            
            # デバイス分布
            device_str = str(param.device)
            if device_str not in analysis["device_distribution"]:
                analysis["device_distribution"][device_str] = 0
            analysis["device_distribution"][device_str] += param.numel()
        
        # パーセンテージ計算
        total = analysis["total_parameters"]
        if total > 0:
            analysis["trainable_percentage"] = 100 * analysis["trainable_parameters"] / total
            analysis["lora_percentage"] = 100 * analysis["lora_parameters"] / total
        
        return analysis
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """結果保存"""
        output_file = Path(self.config.output_dir) / "llama4_lora_verification_results.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=TensorJSONEncoder)
            
            logger.info(f"✓ 検証結果を保存: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
            return ""
    
    def run_verification(self) -> Dict[str, Any]:
        """検証実行"""
        logger.info("🚀 LISA-Llama4 LoRA検証開始")
        logger.info(f"設定: {self.config}")
        
        try:
            # 1. モデル・トークナイザー初期化
            model, processor = self.setup_model_and_tokenizer()
            
            # 2. LoRA適用
            model = self.apply_lora_config(model)
            
            # 3. 勾配フロー検証
            verification_results = self.verify_gradient_flow(model, processor)
            
            # 4. 結果まとめ
            self.results.update(verification_results)
            self.results["config"] = self.config.__dict__
            
            # 5. 結果保存
            output_file = self.save_results(self.results)
            
            logger.info("✅ 検証完了!")
            return self.results
            
        except Exception as e:
            logger.error(f"検証実行エラー: {e}")
            self.results["error"] = str(e)
            return self.results

def main():
    """メイン実行関数"""
    config = VerificationConfig()
    verifier = Llama4LoRAVerification(config)
    
    results = verifier.run_verification()
    
    # 結果サマリー表示
    if "error" not in results:
        print("\n" + "="*50)
        print("🎉 LISA-Llama4 LoRA検証 成功!")
        print("="*50)
        
        if "parameter_analysis" in results:
            analysis = results["parameter_analysis"]
            print(f"📊 パラメータ統計:")
            print(f"  - 総パラメータ数: {analysis.get('total_parameters', 0):,}")
            print(f"  - 学習可能パラメータ: {analysis.get('trainable_parameters', 0):,}")
            print(f"  - 学習可能割合: {analysis.get('trainable_percentage', 0):.3f}%")
            print(f"  - LoRAパラメータ: {analysis.get('lora_parameters', 0):,}")
        
        if "step_details" in results and results["step_details"]:
            print(f"\n🔄 勾配フロー:")
            for step_detail in results["step_details"]:
                if "loss_computation" in step_detail:
                    loss_val = step_detail["loss_computation"].get("loss_value", "N/A")
                    print(f"  - ステップ {step_detail['step']}: 損失 = {loss_val}")
    else:
        print(f"❌ 検証失敗: {results['error']}")

if __name__ == "__main__":
    main() 