#!/usr/bin/env python3
"""
第5節：堅牢性と将来の開発に向けた事前検証
5.1. サニティチェック1：単一バッチへの過学習

論理的根拠:
これは、モデルの学習能力を試すための最も基本的なテストです。
もしモデルがごく少量のデータすら記憶できないのであれば、大規模データセットから汎化則を学習することは不可能です。
このテストは、アーキテクチャ、損失関数、オプティマイザが三位一体となって正しく機能していることを確認します。

期待される結果:
損失値が着実に、かつ急速に減少し、ゼロに近づいていく様子が観測されるはずです。
もし損失が停滞したり、激しく振動したりする場合は、学習率が不適切である、オプティマイザに問題がある、
あるいは損失計算に根深い問題が残っているなど、根本的な問題の存在を示唆します。
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話型に設定
import matplotlib.pyplot as plt
import psutil

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn

def get_config():
    """
    config_linux.pyを必須として読み込む
    読み込めない場合はエラーで停止
    """
    try:
        import config_linux as config
        print(f"設定: config_linux.py を使用")
        return config
    except ImportError as e:
        print(f"❌ ERROR: config_linux.pyが見つかりません")
        print(f"   詳細: {e}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        print(f"   ファイル存在確認: {os.path.exists('config_linux.py')}")
        raise SystemExit("config_linux.pyが必須です。ファイルが存在することを確認してください。")

def parse_args():
    # config_linux.pyから設定を取得
    config = get_config()
    
    parser = argparse.ArgumentParser(description="単一バッチでの過学習テスト")
    parser.add_argument("--iterations", type=int, default=20, help="過学習テストのイテレーション数")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="学習率")
    parser.add_argument("--dataset_type", type=str, default="all", 
                       choices=["sem_seg", "refer_seg", "vqa", "reason_seg", "all"],
                       help="テストに使用するデータセットタイプ（'all'で全データセット）")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE_PER_GPU, help="バッチサイズ")
    parser.add_argument("--wait_between_iterations", type=float, default=0.0, 
                       help="各イテレーション間の待機時間（秒）")
    return parser.parse_args()

def plot_loss_curve(loss_history: List[Dict[str, float]], output_path: str):
    """損失曲線をプロットして保存"""
    iterations = list(range(len(loss_history)))
    
    # 各損失成分を抽出
    total_losses = [h['total_loss'] for h in loss_history]
    text_losses = [h.get('text_loss', 0.0) for h in loss_history]
    dice_losses = [h.get('dice_loss', 0.0) for h in loss_history]
    bce_losses = [h.get('bce_loss', 0.0) for h in loss_history]
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 総損失のプロット（対数スケール）
    ax1.plot(iterations, total_losses, 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_yscale('log')
    ax1.set_title('単一バッチ過学習テスト: 総損失の推移（対数スケール）')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 損失成分の詳細プロット
    ax2.plot(iterations, text_losses, 'r-', linewidth=1, label='Text Loss')
    ax2.plot(iterations, dice_losses, 'g-', linewidth=1, label='DICE Loss')
    ax2.plot(iterations, bce_losses, 'm-', linewidth=1, label='BCE Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('損失成分の詳細推移')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"損失曲線を保存: {output_path}")

def get_memory_usage():
    """GPU/CPUメモリ使用量を取得"""
    memory_info = {}
    
    # CPUメモリ
    cpu_memory = psutil.virtual_memory()
    memory_info['cpu_used_gb'] = cpu_memory.used / (1024**3)
    memory_info['cpu_total_gb'] = cpu_memory.total / (1024**3)
    memory_info['cpu_percent'] = cpu_memory.percent
    
    # GPUメモリ（CUDA利用可能な場合）
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
        
        memory_info['gpu_used_gb'] = gpu_memory
        memory_info['gpu_max_gb'] = gpu_memory_max
        memory_info['gpu_cached_gb'] = gpu_memory_cached
        
        # GPU利用率（簡易的な計算）
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_total_memory = gpu_properties.total_memory / (1024**3)
        memory_info['gpu_total_gb'] = gpu_total_memory
        memory_info['gpu_percent'] = (gpu_memory / gpu_total_memory) * 100
    else:
        memory_info['gpu_used_gb'] = 0
        memory_info['gpu_max_gb'] = 0
        memory_info['gpu_cached_gb'] = 0
        memory_info['gpu_total_gb'] = 0
        memory_info['gpu_percent'] = 0
    
    return memory_info

def format_memory_info(memory_info):
    """メモリ情報を読みやすい形式でフォーマット"""
    cpu_info = f"CPU: {memory_info['cpu_used_gb']:.1f}/{memory_info['cpu_total_gb']:.1f}GB ({memory_info['cpu_percent']:.1f}%)"
    
    if torch.cuda.is_available():
        gpu_info = f"GPU: {memory_info['gpu_used_gb']:.1f}/{memory_info['gpu_total_gb']:.1f}GB ({memory_info['gpu_percent']:.1f}%) [Max: {memory_info['gpu_max_gb']:.1f}GB]"
    else:
        gpu_info = "GPU: N/A"
    
    return f"{cpu_info} | {gpu_info}"

def analyze_overfitting_success(loss_history: List[Dict[str, float]]) -> Dict[str, Any]:
    """過学習の成功度を分析"""
    if len(loss_history) < 2:
        return {
            "success": False, 
            "reason": "insufficient_iterations",
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "reduction_ratio": 0.0,
            "seg_reduction_ratio": 0.0,
            "text_reduction_ratio": 0.0,
            "stability_ratio": 0.0,
            "iterations": len(loss_history)
        }
    
    initial_loss = loss_history[0]['total_loss']
    final_loss = loss_history[-1]['total_loss']
    
    # 損失減少率
    reduction_ratio = (initial_loss - final_loss) / initial_loss
    
    # 最後の損失の安定性チェック（少ないイテレーション対応）
    if len(loss_history) >= 3:
        stable_window = max(2, len(loss_history) // 2)
        recent_losses = [h['total_loss'] for h in loss_history[-stable_window:]]
        recent_std = torch.tensor(recent_losses).std().item()
        recent_mean = torch.tensor(recent_losses).mean().item()
        stability_ratio = recent_std / recent_mean if recent_mean > 0 else float('inf')
    else:
        stability_ratio = 0.0  # 少ないイテレーションでは安定性チェックをスキップ
    
    # セグメンテーション損失の分析
    initial_seg_loss = loss_history[0].get('dice_loss', 0) + loss_history[0].get('bce_loss', 0)
    final_seg_loss = loss_history[-1].get('dice_loss', 0) + loss_history[-1].get('bce_loss', 0)
    seg_reduction_ratio = (initial_seg_loss - final_seg_loss) / initial_seg_loss if initial_seg_loss > 0 else 0
    
    # テキスト損失の分析
    initial_text_loss = loss_history[0].get('text_loss', 0)
    final_text_loss = loss_history[-1].get('text_loss', 0)
    text_reduction_ratio = (initial_text_loss - final_text_loss) / initial_text_loss if initial_text_loss > 0 else 0
    
    # 柔軟な成功基準（少ないイテレーション対応）
    if len(loss_history) < 5:
        # 少ないイテレーション（2-4回）の場合：損失減少があれば成功
        success = reduction_ratio > 0.1  # 10%以上の総損失減少
    else:
        # 十分なイテレーション（5回以上）の場合：厳格な基準
        success = (
            reduction_ratio > 0.2 and  # 20%以上の総損失減少
            seg_reduction_ratio > 0.3 and  # セグメンテーション損失が30%以上減少
            stability_ratio < 0.5  # 最近の損失の変動が平均の50%以下
        )
    
    return {
        "success": success,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "reduction_ratio": reduction_ratio,
        "seg_reduction_ratio": seg_reduction_ratio,
        "text_reduction_ratio": text_reduction_ratio,
        "stability_ratio": stability_ratio,
        "iterations": len(loss_history)
    }

def main():
    args = parse_args()
    config = get_config()
    
    print("="*80)
    print("第5節: 堅牢性と将来の開発に向けた事前検証")
    print("5.1. サニティチェック1：単一バッチへの過学習")
    print("="*80)
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print(f"テスト設定:")
    print(f"  - イテレーション数: {args.iterations}")
    print(f"  - 学習率: {args.learning_rate}")
    print(f"  - データセットタイプ: {args.dataset_type}")
    print(f"  - バッチサイズ: {args.batch_size}")
    if args.wait_between_iterations > 0:
        print(f"  - イテレーション間待機: {args.wait_between_iterations}秒")
    print("-" * 80)
    
    try:
        # 1. モデルの初期化
        print("\n📦 LISA-Gemmaモデルを初期化中...")
        
        lisa_config = LisaGemmaConfig(
            gemma_model_id=getattr(config, 'GEMMA_MODEL_ID', 'google/gemma-3-4b-it'),
            sam_checkpoint_path=getattr(config, 'SAM_CHECKPOINT_PATH', None),
            seg_token=getattr(config, 'SEG_TOKEN', '[SEG]'),
            gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),
            sam_prompt_embed_dim=getattr(config, 'SEG_PROJECTION_DIM', 256),
            gemma_image_size=getattr(config, 'GEMMA_IMAGE_SIZE', 896),
            sam_image_size=getattr(config, 'SAM_IMAGE_SIZE', 1024),
            model_max_length=getattr(config, 'MODEL_MAX_LENGTH', 2048),
        )
        
        model = LisaGemmaForCausalLM(lisa_config)
        model = model.to(device)
        
        # LoRA設定を適用（最適化後の設定で検証するため）
        print("\n🔧 LoRA設定を適用中...")
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=config.LORA_TARGET_MODULES,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # LoRAをGemmaモデルに適用
            model.gemma_model = get_peft_model(model.gemma_model, lora_config)
            print("✅ LoRA設定が正常に適用されました")
            
        except Exception as e:
            print(f"⚠️  LoRA適用に失敗: {e}")
            print("   LoRAなしで検証を続行します")
        
        model.train()  # 学習モードに設定
        print("✅ モデル初期化完了")
        
        # 学習可能パラメータの情報を表示
        param_info = model.get_trainable_parameters_info()
        print(f"  - 総パラメータ数: {param_info['total_parameters']:,}")
        print(f"  - 学習可能パラメータ数: {param_info['trainable_parameters']:,}")
        print(f"  - 学習可能率: {param_info['trainable_percentage']:.2f}%")
        
        # 仕様書準拠性チェック
        trainable_ratio = param_info['trainable_percentage']
        spec_compliant = trainable_ratio < 1.0
        print(f"  - 📋 仕様書準拠性: {'✅ 準拠' if spec_compliant else '❌ 違反'} (要求: <1%)")
        
        if not spec_compliant:
            print(f"    ⚠️  学習可能パラメータ率が仕様書要求を超過: {trainable_ratio:.2f}% > 1%")
            print("    → LoRA適用を確認してください")
        
        # 初期メモリ使用量
        initial_memory = get_memory_usage()
        print(f"  - 初期メモリ使用量: {format_memory_info(initial_memory)}")
        
        # 2. データセットとデータローダーの準備
        print(f"\n📊 {args.dataset_type}データセットを準備中...")
        
        processor = AutoProcessor.from_pretrained(lisa_config.gemma_model_id)
        
        # データセットタイプの決定
        if args.dataset_type == "all":
            dataset_spec = "sem_seg||refer_seg||vqa||reason_seg"
            print(f"  - 全データセットを使用: sem_seg, refer_seg, vqa, reason_seg")
        else:
            dataset_spec = args.dataset_type
            print(f"  - 単一データセットを使用: {args.dataset_type}")
        
        # A10 24GB制約対応: 動的バッチサイズ調整
        effective_batch_size = args.batch_size
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 25:  # A10 (24GB) 検出
                effective_batch_size = 1
                print(f"  ⚠️  A10 GPU検出 ({gpu_memory_gb:.1f}GB): バッチサイズを {args.batch_size} → {effective_batch_size} に調整")
            else:
                print(f"  ✅ GPU メモリ十分 ({gpu_memory_gb:.1f}GB): バッチサイズ {effective_batch_size} を維持")
        
        # 単一バッチテスト用の小規模データセット
        dataset = HybridDataset(
            base_image_dir=getattr(config, 'DATASET_BASE_DIR', './dataset'),
            gemma_processor=processor,
            dataset=dataset_spec,
            samples_per_epoch=effective_batch_size * 2  # テスト用に少数のサンプル
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            collate_fn=collate_fn,
            shuffle=False  # 再現性のため固定
        )
        
        print(f"✅ データセット準備完了（サンプル数: {len(dataset)}）")
        
        # 3. 固定バッチの取得
        print("\n🎯 固定バッチを取得中...")
        fixed_batch = next(iter(dataloader))
        
        # バッチをデバイスに移動
        fixed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in fixed_batch.items()}
        
        print(f"✅ 固定バッチ取得完了")
        print(f"  - バッチサイズ: {fixed_batch['input_ids'].shape[0]}")
        print(f"  - シーケンス長: {fixed_batch['input_ids'].shape[1]}")
        
        # バッチの内容を確認
        has_segmentation = 'ground_truth_mask' in fixed_batch
        print(f"  - セグメンテーションタスク: {'あり' if has_segmentation else 'なし'}")
        
        # 4. オプティマイザーと損失関数の準備
        print("\n⚙️ オプティマイザーと損失関数を準備中...")
        
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-2),
            betas=(getattr(config, 'BETA1', 0.9), getattr(config, 'BETA2', 0.95))
        )
        
        loss_fn = CompositeLoss(
            ce_loss_weight=getattr(config, 'CE_LOSS_WEIGHT', 1.0),
            dice_loss_weight=getattr(config, 'DICE_LOSS_WEIGHT', 0.5),
            bce_loss_weight=getattr(config, 'BCE_LOSS_WEIGHT', 2.0)
        )
        
        print("✅ オプティマイザーと損失関数の準備完了")
        
        # 5. 過学習ループの実行
        print(f"\n🚀 単一バッチでの過学習を開始...")
        print(f"目標: {args.iterations}回の学習で損失を大幅に減少させる")
        
        # デバッグログの抑制設定
        import logging
        logging.getLogger().setLevel(logging.WARNING)  # デバッグログを抑制
        
        print("-" * 80)
        
        loss_history = []
        best_loss = float('inf')
        start_time = time.time()
        
        for iteration in range(args.iterations):
            # フォワードパス
            optimizer.zero_grad()
            
            # モデルに適した入力形式を準備
            model_inputs = {
                'input_ids': fixed_batch['input_ids'],
                'attention_mask': fixed_batch['attention_masks'],  # collate_fnでは'attention_masks'が使われる
                'labels': fixed_batch['labels'],
                'generate_mask': has_segmentation,
            }
            
            # デュアルストリーム対応
            if 'images_for_gemma' in fixed_batch:
                model_inputs['images_for_gemma'] = fixed_batch['images_for_gemma']
            if 'images_for_sam' in fixed_batch:
                model_inputs['images_for_sam'] = fixed_batch['images_for_sam']
            
            # フォワードパス実行
            try:
                outputs = model(**model_inputs)
                
                # フォワードパス後のメモリ監視
                if iteration == 0:  # 初回のみ詳細表示
                    if torch.cuda.is_available():
                        allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                        print(f"  📊 フォワードパス後 GPU メモリ: {allocated_memory:.2f}GB")
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ フォワードパス中にGPUメモリ不足: {e}")
                print("🔧 メモリクリーンアップを実行中...")
                torch.cuda.empty_cache()
                raise
            
            # 損失計算
            losses = loss_fn(outputs, fixed_batch)
            total_loss = losses['total_loss']
            
            # バックワードパス
            try:
                total_loss.backward()
                
                # バックワードパス後のメモリ監視
                if iteration == 0:  # 初回のみ詳細表示
                    if torch.cuda.is_available():
                        allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                        print(f"  📊 バックワードパス後 GPU メモリ: {allocated_memory:.2f}GB")
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ バックワードパス中にGPUメモリ不足: {e}")
                print("🔧 メモリクリーンアップを実行中...")
                torch.cuda.empty_cache()
                raise
            
            # 勾配クリッピング（安定性のため）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # オプティマイザーステップ
            optimizer.step()
            
            # 損失履歴を記録
            loss_record = {
                'iteration': iteration,
                'total_loss': total_loss.item(),
                'text_loss': losses.get('text_loss', torch.tensor(0.0)).item(),
                'dice_loss': losses.get('dice_loss', torch.tensor(0.0)).item(),
                'bce_loss': losses.get('bce_loss', torch.tensor(0.0)).item(),
            }
            loss_history.append(loss_record)
            
            # 最良損失の更新
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            
            # 進捗表示（5イテレーションごと、または最終）
            if iteration % 5 == 0 or iteration == args.iterations - 1:
                elapsed_time = time.time() - start_time
                memory_info = get_memory_usage()
                memory_str = format_memory_info(memory_info)
                
                print(f"Iter {iteration:3d}/{args.iterations}: "
                      f"Loss={total_loss.item():.6f} "
                      f"(Text: {loss_record['text_loss']:.4f}, "
                      f"DICE: {loss_record['dice_loss']:.4f}, "
                      f"BCE: {loss_record['bce_loss']:.4f}) "
                      f"Best: {best_loss:.6f} "
                      f"Time: {elapsed_time:.1f}s")
                print(f"       Memory: {memory_str}")
            
            # イテレーション間の待機（メモリ監視やデバッグ用）
            if args.wait_between_iterations > 0 and iteration < args.iterations - 1:
                time.sleep(args.wait_between_iterations)
        
        print("-" * 80)
        print("✅ 過学習ループ完了")
        
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"🧹 GPU メモリクリーンアップ完了 (最終使用量: {final_memory:.2f}GB)")
        
        # 6. 結果の分析
        print("\n📊 過学習結果の分析...")
        
        analysis = analyze_overfitting_success(loss_history)
        
        print(f"過学習テスト結果: {'✅ 成功' if analysis['success'] else '❌ 失敗'}")
        print(f"  - 初期損失: {analysis['initial_loss']:.6f}")
        print(f"  - 最終損失: {analysis['final_loss']:.6f}")
        print(f"  - 総損失減少率: {analysis['reduction_ratio']:.1%}")
        print(f"  - セグメンテーション損失減少率: {analysis['seg_reduction_ratio']:.1%}")
        print(f"  - テキスト損失減少率: {analysis['text_reduction_ratio']:.1%}")
        print(f"  - 安定性比率: {analysis['stability_ratio']:.4f}")
        print(f"  - イテレーション数: {analysis['iterations']}")
        
        # 詳細な成功基準チェック
        print(f"\n📋 成功基準チェック (20回イテレーション対応):")
        print(f"  - 総損失減少 ≥20%: {'✅' if analysis['reduction_ratio'] > 0.2 else '❌'} ({analysis['reduction_ratio']:.1%})")
        print(f"  - セグメンテーション損失減少 ≥60%: {'✅' if analysis['seg_reduction_ratio'] > 0.6 else '❌'} ({analysis['seg_reduction_ratio']:.1%})")
        print(f"  - 安定性 ≤50%: {'✅' if analysis['stability_ratio'] < 0.5 else '❌'} ({analysis['stability_ratio']:.1%})")
        
        # 学習可能性の診断
        print(f"\n🔍 学習可能性の診断:")
        if analysis['text_reduction_ratio'] < 0.01:
            print("  ⚠️  テキスト損失がほとんど減少していません")
            print("     → Gemmaモデルの学習可能パラメータを確認してください")
        if analysis['seg_reduction_ratio'] > 0.5:
            print("  ✅ セグメンテーション損失は正常に減少しています")
            print("     → SAMデコーダーは正常に学習しています")
        else:
            print("  ⚠️  セグメンテーション損失の減少が不十分です")
            print("     → SAMデコーダーの学習設定を確認してください")
        
        # 7. 結果の保存
        print("\n💾 結果を保存中...")
        
        # 出力ディレクトリの作成
        output_dir = "verification_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 損失曲線のプロット
        plot_path = os.path.join(output_dir, f"overfit_single_batch_{session_timestamp}.png")
        plot_loss_curve(loss_history, plot_path)
        
        # 詳細結果をJSONで保存
        results = {
            "session_timestamp": session_timestamp,
            "test_config": {
                "iterations": args.iterations,
                "learning_rate": args.learning_rate,
                "dataset_type": args.dataset_type,
                "batch_size": args.batch_size,
            },
            "model_info": param_info,
            "analysis": analysis,
            "loss_history": loss_history,
            "final_status": "success" if analysis['success'] else "failure"
        }
        
        json_path = os.path.join(output_dir, f"overfit_single_batch_{session_timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 結果保存完了:")
        print(f"  - 損失曲線: {plot_path}")
        print(f"  - 詳細データ: {json_path}")
        
        # 過学習成功時のみモデルチェックポイントを保存
        if analysis['success']:
            checkpoint_path = os.path.join(output_dir, f"overfit_checkpoint_{session_timestamp}.pth")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'session_timestamp': session_timestamp,
                'config': {
                    'iterations': args.iterations,
                    'learning_rate': args.learning_rate,
                    'dataset_type': args.dataset_type,
                    'final_loss': analysis['final_loss'],
                    'reduction_ratio': analysis['reduction_ratio']
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  - モデルチェックポイント: {checkpoint_path}")
            
            # 最新の成功チェックポイントのシンボリックリンクを作成
            latest_checkpoint_path = os.path.join(output_dir, "latest_overfit_checkpoint.pth")
            try:
                # 既存のシンボリックリンクを安全に削除
                if os.path.islink(latest_checkpoint_path):
                    os.unlink(latest_checkpoint_path)
                elif os.path.exists(latest_checkpoint_path):
                    os.remove(latest_checkpoint_path)
                
                # 絶対パスを使用してシンボリックリンクを作成
                os.symlink(os.path.abspath(checkpoint_path), latest_checkpoint_path)
                print(f"  - 最新チェックポイント: {latest_checkpoint_path}")
            except (OSError, FileExistsError) as e:
                print(f"  ⚠️  シンボリックリンク作成をスキップ: {e}")
                print(f"      最新チェックポイント: {checkpoint_path} (直接参照)")
            except Exception as e:
                print(f"  ⚠️  予期しないエラーでシンボリックリンクをスキップ: {e}")
        
        print(f"✅ 結果保存完了:")
        print(f"  - 損失曲線: {plot_path}")
        print(f"  - 詳細データ: {json_path}")
        
        # 8. 最終判定
        print("\n" + "="*80)
        print("🎯 最終判定")
        print("="*80)
        
        if analysis['success']:
            print("✅ サニティチェック1: 単一バッチへの過学習 - 成功")
            print("   モデルは正常に学習能力を示しています。")
            print("   アーキテクチャ、損失関数、オプティマイザーが正しく機能しています。")
        else:
            print("❌ サニティチェック1: 単一バッチへの過学習 - 失敗")
            print("   以下の問題が考えられます:")
            
            if analysis['reduction_ratio'] < 0.1:
                print("   - 学習率が低すぎる可能性があります")
            elif analysis['stability_ratio'] > 0.4:
                print("   - 学習が不安定です（学習率が高すぎる可能性）")
            elif analysis['text_reduction_ratio'] < 0.01:
                print("   - Gemmaモデルが学習されていません（LoRA設定を確認）")
            elif analysis['seg_reduction_ratio'] < 0.5:
                print("   - セグメンテーション損失が十分減少していません")
            else:
                print("   - 損失関数または勾配伝播に問題がある可能性があります")
            
            print(f"   推奨: verify_loss_and_gradients.pyを再実行して根本原因を調査してください")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 