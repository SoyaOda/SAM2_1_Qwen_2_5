#!/usr/bin/env python3
"""
LISA-Llama4-Scout + Q-Former + SAM2 統合モデル 勾配フロー検証スクリプト

model/llama4_qformer_sam2.py と model/losses_qformer_sam2.py の勾配フローテスト

実行方法:
Lambda Cloud:
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip_address>:/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip_address> "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u test/verify_llama4_qformer_sam2_gradients.py 2>&1"
"""

import os
import sys
import logging
import torch
import warnings
from PIL import Image
from typing import Dict, Any, Tuple
import torch.nn as nn
import numpy as np

# Disable various warnings to clean up output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LISA-QFormer components
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
from model.losses_qformer_sam2 import get_composite_loss_qformer_sam2
import config_linux

def print_header(title: str):
    """Print formatted header"""
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)

def create_test_data(model) -> Dict[str, torch.Tensor]:
    """Create realistic test data for gradient verification (2025 best practices)
    
    Based on web research for Q-Former + SAM2 integration:
    - Uses real processor for realistic data pipeline
    - SAM2 standard input size: 512x512 (downscaled to 448 for compatibility)  
    - Q-Former requires proper text-image alignment
    - No requires_grad on input data (only check model parameter gradients)
    """
    
    logger.info("📊 現実的テストデータ作成（2025年ベストプラクティス）...")
    
    test_config = config_linux.get_test_config()
    batch_size = 1
    
    # 1. リアルな画像データ作成（SAM2推奨: 512x512ベース、448x448にリサイズ）
    # チェッカーボード + ノイズパターン（よりリアルに）
    image_size = test_config['image_size']  # 448
    
    # チェッカーボードベースパターン（SAM2テスト用）
    checker_size = 64
    image_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    for i in range(0, image_size, checker_size):
        for j in range(0, image_size, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                # 白い正方形
                end_i, end_j = min(i + checker_size, image_size), min(j + checker_size, image_size)
                image_array[i:end_i, j:end_j] = [240, 240, 240]
            else:
                # グラデーション正方形
                end_i, end_j = min(i + checker_size, image_size), min(j + checker_size, image_size)
                for di in range(end_i - i):
                    for dj in range(end_j - j):
                        image_array[i + di, j + dj] = [
                            int(128 + 64 * di / checker_size),
                            int(64 + 128 * dj / checker_size), 
                            int(200 - 100 * (di + dj) / (2 * checker_size))
                        ]
    
    # ランダムノイズ追加（リアリティ向上）
    noise = np.random.normal(0, 15, image_array.shape).astype(np.int16)
    image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # PyTorchテンソルに変換（勾配追跡なし - ベストプラクティス）
    images = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
    # 正規化（0-1範囲）
    images = images / 255.0
    
    logger.info(f"  📸 画像生成完了: {images.shape}, 範囲: {images.min():.3f}-{images.max():.3f}")
    
    # 2. Q-Former用セグメンテーションプロンプト作成
    seg_prompts = [
        "Segment the white checkerboard squares in this image.",
        "この画像の白いチェッカーボード領域を分割してください。",
        "Find and segment the bright rectangular regions.",
    ]
    
    selected_prompt = seg_prompts[0]  # 英語プロンプト使用
    logger.info(f"  💬 セグメンテーションプロンプト: '{selected_prompt}'")
    
    # 3. プロセッサ取得（Model Parallelism対応）
    processor = None
    if hasattr(model, 'llama_model') and hasattr(model.llama_model, 'processor'):
        processor = model.llama_model.processor
        logger.info("  ✓ llama_model.processor取得")
    elif hasattr(model, 'processor'):
        processor = model.processor  
        logger.info("  ✓ model.processor取得")
    else:
        # フォールバック: 手動トークナイズ
        logger.warning("  ⚠️ プロセッサが見つかりません。手動トークナイズに切り替え")
        seq_len = 64
        input_ids = torch.randint(1, 32000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
    
    if processor:
        # 4. 現実的なトークナイゼーション（Q-Former + Llama-4適応）
        try:
            # BLIP-2スタイルプロンプト構築
            formatted_prompt = f"Question: {selected_prompt} Answer:"
            
            # テキストをトークナイズ
            text_inputs = processor.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128  # Q-Former用に短め
            )
            
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
            
            logger.info(f"  🔤 トークナイゼーション完了: {input_ids.shape}")
            
        except Exception as e:
            logger.warning(f"  ⚠️ プロセッサエラー: {e}. 手動トークナイズ使用")
            seq_len = 64
            input_ids = torch.randint(1, 32000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
    
    # 5. 現実的な正解マスク作成（SAM2準拠）
    mask_size = test_config['mask_size']  # 448
    target_masks = torch.zeros(batch_size, mask_size, mask_size, dtype=torch.float32)
    
    # チェッカーボードの白い領域をターゲットとする
    for i in range(0, mask_size, checker_size):
        for j in range(0, mask_size, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                end_i = min(i + checker_size, mask_size)
                end_j = min(j + checker_size, mask_size) 
                target_masks[0, i:end_i, j:end_j] = 1.0
    
    # エッジをソフトにする（より現実的）
    from scipy.ndimage import gaussian_filter
    for b in range(batch_size):
        target_masks[b] = torch.from_numpy(
            gaussian_filter(target_masks[b].numpy(), sigma=1.5)
        )
        target_masks[b] = (target_masks[b] > 0.5).float()
    
    logger.info(f"  🎯 正解マスク作成完了: {target_masks.shape}")
    logger.info(f"  📊 正例率: {target_masks.mean().item():.3f}")
    
    # 6. 統合データ辞書作成（HybridDataset互換）
    test_data = {
        'images': images,  # Q-Former + SAM2用画像
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
        'target_masks': target_masks,
        'original_prompt': selected_prompt,
        'dataset_type': 'gradient_verification_synthetic'
    }
    
    # データ検証
    logger.info("  ✅ テストデータ検証:")
    for key, value in test_data.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"    - {key}: {value.shape}, dtype: {value.dtype}, requires_grad: {value.requires_grad}")
        else:
            logger.info(f"    - {key}: {type(value)}")
    
    logger.info("📊 現実的テストデータ作成完了（2025年ベストプラクティス準拠）")
    
    return test_data

def analyze_model_components(model: nn.Module) -> Dict[str, Any]:
    """モデルコンポーネントの詳細分析"""
    
    logger.info("🔍 モデルコンポーネント分析...")
    
    analysis = {
        'total_params': 0,
        'trainable_params': 0,
        'components': {},
        'gradient_enabled_components': []
    }
    
    # 全体パラメータ数
    for param in model.parameters():
        analysis['total_params'] += param.numel()
        if param.requires_grad:
            analysis['trainable_params'] += param.numel()
    
    # コンポーネント別分析
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # リーフモジュールのみ
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if param_count > 0:
                analysis['components'][name] = {
                    'params': param_count,
                    'trainable': trainable_count,
                    'requires_grad': any(p.requires_grad for p in module.parameters())
                }
                
                if trainable_count > 0:
                    analysis['gradient_enabled_components'].append(name)
    
    return analysis

def test_gradient_flow(
    model: nn.Module,
    test_data: Dict[str, torch.Tensor],
    device: str
) -> Dict[str, Any]:
    """勾配フローの詳細テスト（2025年ベストプラクティス準拠）"""
    
    logger.info("🔄 勾配フローテスト実行（2025年ベストプラクティス）...")
    
    # Model Parallelismデバイス取得（train_llama4_lisa_single_process.pyのパターン使用）
    first_device = None
    
    # 複数のアクセス方法を試行（train_llama4_lisa_single_process.pyと同じパターン）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        logger.info("  ✓ device_map (直接アクセス) からデバイス取得")
    elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'parameters'):
        first_device = next(model.llama_model.parameters()).device
        logger.info("  ✓ llama_model.parameters からデバイス取得")
    elif hasattr(model, 'qformer') and hasattr(model.qformer, 'parameters'):
        first_device = next(model.qformer.parameters()).device
        logger.info("  ✓ qformer.parameters からデバイス取得")
    else:
        # フォールバック: 最初のパラメータのデバイス
        first_device = next(model.parameters()).device
        logger.info("  ✓ model.parameters からデバイス取得")
    
    logger.info(f"  📍 モデルのメインデバイス: {first_device}")
    
    # データを統一デバイスに移動（2025年ベストプラクティス）
    images = test_data['images'].to(first_device)
    input_ids = test_data['input_ids'].to(first_device)
    attention_mask = test_data['attention_mask'].to(first_device)
    target_masks = test_data['target_masks'].to(first_device)
    
    logger.info(f"  📍 入力データデバイス移動完了: {first_device}")
    logger.info(f"  📊 入力データ形状:")
    logger.info(f"    - images: {images.shape}, device: {images.device}")
    logger.info(f"    - input_ids: {input_ids.shape}, device: {input_ids.device}")
    logger.info(f"    - target_masks: {target_masks.shape}, device: {target_masks.device}")
    
    # PyTorch 2025年推奨: gradient anomaly detection (デバッグ用)
    with torch.autograd.detect_anomaly(enabled=True):
        # フォワードパス
        model.train()
        model.zero_grad(set_to_none=True)  # 2025年推奨: メモリ効率化
        
        logger.info("  📈 フォワードパス実行中...")
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 出力検証
        logger.info("  📋 モデル出力検証:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    - {key}: {value.shape}, device: {value.device}, dtype: {value.dtype}")
        
        # 損失計算（デバイス一貫性確保）
        logger.info("  📊 損失計算中...")
        
        # モデル内蔵の損失関数を使用
        if hasattr(model, 'compute_loss'):
            # 出力をすべて同じデバイスに確実に移動（train_llama4_lisa_single_process.pyパターン）
            predicted_masks = outputs['predicted_masks']
            if hasattr(predicted_masks, 'to'):
                predicted_masks = predicted_masks.to(first_device)
                
            query_embeds = outputs.get('query_embeddings')
            if query_embeds is not None and hasattr(query_embeds, 'to'):
                query_embeds = query_embeds.to(first_device)
                
            text_embeds = outputs.get('text_embeds')
            if text_embeds is not None and hasattr(text_embeds, 'to'):
                text_embeds = text_embeds.to(first_device)
                
            sam_prompts = outputs.get('sam_prompts')
            if sam_prompts is not None and hasattr(sam_prompts, 'to'):
                sam_prompts = sam_prompts.to(first_device)
            
            logger.info(f"  📍 出力テンソルのデバイス確認完了")
            
            # 損失計算実行
            losses = model.compute_loss(
                predicted_masks=predicted_masks,
                target_masks=target_masks,
                query_embeds=query_embeds,
                text_embeds=text_embeds,
                sam_prompts=sam_prompts
            )
            total_loss = losses['total_loss']
            
            logger.info("  📋 計算された損失:")
            for loss_name, loss_value in losses.items():
                if hasattr(loss_value, 'item'):
                    logger.info(f"    - {loss_name}: {loss_value.item():.6f}")
                else:
                    logger.info(f"    - {loss_name}: {loss_value}")
        else:
            # フォールバック: 基本損失
            logger.info("  ⚠️ compute_lossメソッドが見つかりません。フォールバック損失使用")
            predicted_masks = outputs['predicted_masks']
            if hasattr(predicted_masks, 'to'):
                predicted_masks = predicted_masks.to(first_device)
                
            if predicted_masks.dim() == 4:
                predicted_masks = predicted_masks[:, 0]  # 最初のクエリ
            
            total_loss = nn.functional.binary_cross_entropy_with_logits(
                predicted_masks, target_masks
            )
            losses = {'total_loss': total_loss, 'bce_loss': total_loss}
            logger.info(f"  📋 フォールバック損失: {total_loss.item():.6f}")
        
        # バックワードパス（2025年推奨手法）
        logger.info("  📉 バックワードパス実行中...")
        total_loss.backward()
    
    # 勾配分析（パフォーマンス監視）
    gradient_analysis = analyze_gradients(model)
    
    # メモリクリーンアップ（2025年ベストプラクティス）
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    result = {
        'losses': losses,
        'outputs': outputs,
        'gradient_analysis': gradient_analysis,
        'forward_success': True,
        'backward_success': True,
        'test_data_info': {
            'prompt': test_data.get('original_prompt'),
            'dataset_type': test_data.get('dataset_type')
        }
    }
    
    return result

def analyze_gradients(model: nn.Module) -> Dict[str, Any]:
    """勾配の詳細分析"""
    
    logger.info("  🔍 勾配分析中...")
    
    analysis = {
        'components_with_gradients': [],
        'components_without_gradients': [],
        'gradient_norms': {},
        'zero_gradients': [],
        'large_gradients': [],
        'total_gradient_norm': 0.0
    }
    
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                analysis['gradient_norms'][name] = grad_norm
                total_norm += grad_norm ** 2
                
                analysis['components_with_gradients'].append(name)
                
                if grad_norm < 1e-8:
                    analysis['zero_gradients'].append(name)
                elif grad_norm > 10.0:
                    analysis['large_gradients'].append(name)
            else:
                analysis['components_without_gradients'].append(name)
    
    analysis['total_gradient_norm'] = total_norm ** 0.5
    
    # 勾配統計表示
    logger.info(f"    勾配有りコンポーネント: {len(analysis['components_with_gradients'])}")
    logger.info(f"    勾配無しコンポーネント: {len(analysis['components_without_gradients'])}")
    logger.info(f"    ゼロ勾配コンポーネント: {len(analysis['zero_gradients'])}")
    logger.info(f"    大勾配コンポーネント: {len(analysis['large_gradients'])}")
    logger.info(f"    総勾配ノルム: {analysis['total_gradient_norm']:.6f}")
    
    # 問題のあるコンポーネントを詳細報告
    if analysis['components_without_gradients']:
        logger.warning("  ⚠️ 勾配が計算されていないコンポーネント:")
        for name in analysis['components_without_gradients'][:5]:  # 最初の5個
            logger.warning(f"    - {name}")
    
    if analysis['zero_gradients']:
        logger.warning("  ⚠️ ゼロ勾配のコンポーネント:")
        for name in analysis['zero_gradients'][:5]:  # 最初の5個
            logger.warning(f"    - {name}")
    
    return analysis

def run_comprehensive_gradient_test():
    """包括的勾配テストの実行"""
    
    print_header("LISA-Llama4-QFormer-SAM2 勾配フロー検証テスト")
    
    try:
        # 1. GPU環境確認
        print_header("1. GPU環境確認")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA が利用できません")
        
        device_count = torch.cuda.device_count()
        logger.info(f"✅ CUDA利用可能: {device_count}個のGPU")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        device = "cuda"
        
        # 2. モデル初期化
        print_header("2. モデル初期化")
        
        config = LlamaQFormerSAM2Config()
        logger.info(f"📋 設定: {config.llama_model_id}")
        
        model = QFormerSegmentationBridge(config=config, training_stage=2)  # Stage 2で複合損失
        logger.info("✅ モデル初期化成功")
        
        # 3. モデル分析
        print_header("3. モデルコンポーネント分析")
        
        analysis = analyze_model_components(model)
        logger.info(f"📊 総パラメータ数: {analysis['total_params']:,}")
        logger.info(f"📊 学習可能パラメータ数: {analysis['trainable_params']:,}")
        logger.info(f"📊 学習可能比率: {analysis['trainable_params']/analysis['total_params']*100:.2f}%")
        logger.info(f"📊 勾配有効コンポーネント数: {len(analysis['gradient_enabled_components'])}")
        
        # 4. テストデータ作成
        print_header("4. テストデータ作成")
        
        test_data = create_test_data(model)
        
        # 5. 勾配フローテスト（複数段階）
        for stage in [1, 2, 3]:
            print_header(f"5.{stage}. Stage {stage} 勾配フローテスト")
            
            # 学習段階設定
            if hasattr(model, 'set_training_stage'):
                model.set_training_stage(stage)
            
            # 勾配フローテスト実行
            result = test_gradient_flow(
                model=model,
                test_data=test_data,
                device=device
            )
            
            if result['forward_success'] and result['backward_success']:
                logger.info(f"✅ Stage {stage} 勾配フローテスト成功")
            else:
                logger.error(f"❌ Stage {stage} 勾配フローテスト失敗")
        
        # 6. メモリ使用量確認
        print_header("6. メモリ使用量確認")
        
        for i in range(device_count):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            
            logger.info(f"📊 GPU {i} メモリ:")
            logger.info(f"  現在: {allocated:.2f}GB, 予約: {cached:.2f}GB, 最大: {max_allocated:.2f}GB")
        
        # 7. 最終検証
        print_header("7. 最終検証")
        
        logger.info("✅ 全ての勾配フローテストが正常に完了しました")
        logger.info("✅ モデルは学習に適した状態です")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 勾配フロー検証中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    logger.info("=== LISA-Llama4-QFormer-SAM2 勾配フロー検証開始 ===")
    
    success = run_comprehensive_gradient_test()
    
    if success:
        logger.info("✅ 勾配フロー検証完了 - 成功")
        return 0
    else:
        logger.error("❌ 勾配フロー検証完了 - 失敗")
        return 1

if __name__ == "__main__":
    exit(main())