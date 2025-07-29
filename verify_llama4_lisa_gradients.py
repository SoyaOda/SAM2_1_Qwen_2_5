#!/usr/bin/env python3
"""
LISA-Llama4統合モデル 勾配フロー検証スクリプト
verify_llama4_loss_and_gradients.pyを参考に、LISA統合モデル特有の機能もテスト

実行方法:
Lambda Cloud (129.213.148.184):
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@129.213.148.184:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@129.213.148.184 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 python -u verify_llama4_lisa_gradients.py 2>&1"
"""

import os
import sys
import logging
import torch
import warnings
from PIL import Image
from typing import Dict, Any, Tuple
import torch.nn as nn

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
sys.path.insert(0, '.')

# Import LISA components
from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
import config_linux
from utils.dataset import preprocess_sam_image, build_correct_labels_for_llama4

def print_header(title: str):
    """Print formatted header"""
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)

def create_test_data() -> Tuple[Image.Image, str, str]:
    """Create test image and text data with dataset context"""
    # Simple test image (Lambda Cloud標準サイズ)
    test_image = Image.new('RGB', (448, 448), color='red')
    
    # Test prompt for segmentation (HybridDataset形式に準拠)
    test_prompt = "Please segment the red region in this image."
    
    # Dataset name for context
    dataset_name = "test_dataset"
    
    return test_image, test_prompt, dataset_name

def analyze_gradients(model: nn.Module) -> Dict[str, Any]:
    """Analyze gradient information"""
    gradient_info = {
        'total_params': 0,
        'trainable_params': 0,
        'params_with_grad': 0,
        'zero_grad_params': 0,
        'gradient_norm': 0.0,
        'component_analysis': {}
    }
    
    component_grads = {}
    
    for name, param in model.named_parameters():
        gradient_info['total_params'] += 1
        
        if param.requires_grad:
            gradient_info['trainable_params'] += 1
            
            if param.grad is not None:
                gradient_info['params_with_grad'] += 1
                
                # Calculate gradient norm
                grad_norm = param.grad.norm().item()
                gradient_info['gradient_norm'] += grad_norm ** 2
                
                # Component-wise analysis
                component = name.split('.')[0] if '.' in name else name
                if component not in component_grads:
                    component_grads[component] = {
                        'count': 0, 
                        'norm': 0.0,
                        'params': []
                    }
                component_grads[component]['count'] += 1
                component_grads[component]['norm'] += grad_norm ** 2
                component_grads[component]['params'].append(name)
                
                if grad_norm == 0:
                    gradient_info['zero_grad_params'] += 1
    
    # Calculate L2 norm
    gradient_info['gradient_norm'] = gradient_info['gradient_norm'] ** 0.5
    
    # Calculate component norms
    for component in component_grads:
        component_grads[component]['norm'] = component_grads[component]['norm'] ** 0.5
    
    gradient_info['component_analysis'] = component_grads
    
    return gradient_info

def verify_loss_calculation(model: nn.Module, inputs: Dict[str, torch.Tensor], verbose: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Verify loss calculation with detailed debugging and return loss details"""
    model.train()
    loss_details = {}
    
    if verbose:
        logger.info("=== 損失計算検証 ===")
        logger.info(f"入力形状:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
    
    # Forward pass - 実際のLISA統合モデルを使用（SAM機能付き）
    try:
        if verbose:
            logger.info("LISA統合モデル：完全なフォワードパス実行（SAM機能付き）")
        
        # 実際のLISA統合モデルの完全フォワードパス
        # SAMを有効にしてセグメンテーション機能も検証
        # シングルエンコーダー構成: pixel_valuesを削除
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            # pixel_values=inputs.get('pixel_values'),  # シングルエンコーダーでは削除
            sam_pixel_values=inputs.get('sam_pixel_values'),  # SAM画像入力を追加
            labels=inputs['labels'],  # 正しくマスクされたラベルを使用
            ground_truth_masks=inputs.get('ground_truth_masks'),  # GT���スク
            seg_token_mask=inputs.get('seg_token_mask'),  # SEGトークン位置
            original_sizes=inputs.get('original_sizes'),  # 元画像サイズ
            generate_mask=True  # SAM機能を有効化
        )
        
        if verbose:
            logger.info(f"出力型: {type(model_outputs)}")
            logger.info(f"出力キー: {list(model_outputs.keys()) if isinstance(model_outputs, dict) else 'no keys'}")
        
        # CompositeLoss統合による損失取得
        if isinstance(model_outputs, dict):
            # 損失の詳細表示と記録
            if 'losses' in model_outputs and isinstance(model_outputs['losses'], dict):
                if verbose:
                    logger.info("📊 損失の詳細内訳:")
                    for loss_name, loss_value in model_outputs['losses'].items():
                        if loss_value is not None:
                            loss_details[loss_name] = loss_value.item()
                            logger.info(f"  - {loss_name}: {loss_value.item():.6f}")
                        else:
                            loss_details[loss_name] = None
                            logger.info(f"  - {loss_name}: N/A")
            
            # CompositeLossからの統一損失
            if 'text_loss' in model_outputs:
                loss = model_outputs['text_loss']
                if verbose:
                    logger.info("✅ CompositeLoss統合損失を使用")
            # 予備処理：lossキーも確認
            elif 'loss' in model_outputs:
                loss = model_outputs['loss']
                if verbose:
                    logger.info("✅ 通常のloss損失を使用")
            # フォールバック：手動計算
            else:
                if verbose:
                    logger.info("手動損失計算に切り替え...")
                
                logits = model_outputs.get('logits')
                if logits is None:
                    raise ValueError("logitsが見つかりません")
                
                labels = inputs.get('input_ids')
                if labels is None:
                    raise ValueError("input_idsが見つかりません")
                
                # 言語モデリング損失を手動計算
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                if verbose:
                    logger.info("✅ 手動交差エントロピー損失計算完了")
        else:
            # 非辞書型出力の場合
            if hasattr(model_outputs, 'loss'):
                loss = model_outputs.loss
            else:
                raise ValueError("損失が見つかりません")
        
        # SAM機能確認（シングルエンコーダー構成でのマスク出力確認）
        if isinstance(model_outputs, dict) and 'pred_masks' in model_outputs:
            masks = model_outputs['pred_masks']
            if masks is not None:
                if verbose:
                    logger.info(f"✅ SAMマスク生成成功: {masks.shape if hasattr(masks, 'shape') else type(masks)}")
                    # シングルエンコーダー構成では、マスクは元の画像サイズにリサイズ済み
                    if hasattr(masks, 'shape') and len(masks.shape) >= 2:
                        logger.info(f"  マスクサイズ: {masks.shape[-2:]} (元画像サイズ)")
            else:
                if verbose:
                    logger.info("ℹ️ SAMマスク未生成（SEGトークンなしまたはSAM無効）")
        
        if verbose:
            logger.info(f"✅ 合計損失値: {loss.item():.6f}")
            logger.info(f"損失のデバイス: {loss.device}")
            logger.info(f"損失のrequires_grad: {loss.requires_grad}")
        
        # 合計損失も記録
        loss_details['total_loss'] = loss.item()
        
        return loss, loss_details
        
    except Exception as e:
        logger.error(f"❌ 損失計算でエラー: {e}")
        raise

def main():
    """Main verification function"""
    print_header("LISA-Llama4統合モデル 勾配フロー検証")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA が利用できません")
        return False
    
    logger.info(f"✅ CUDA利用可能: {torch.cuda.device_count()}個のGPU")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 損失の詳細を保存する変数
    loss_details = {}
    
    try:
        # Disable dynamic compilation for distributed setup
        torch.compiler.disable()
        logger.info("✅ 動的コンパイル無効化: GPU分散エラー回避のため")
        
        # Step 1: Model initialization
        print_header("ステップ1: モデル初期化")
        
        config = LisaLlama4Config(**config_linux.get_lisa_model_config())
        
        logger.info("LISA-Llama4モデル初期化中...")
        model = LisaLlama4ForCausalLM(config)
        logger.info("✅ モデル初期化完了")
        
        # Step 2: Apply LoRA for training
        print_header("ステップ2: LoRA設定適用")
        
        from peft import LoraConfig, get_peft_model
        
        lora_config_dict = config_linux.get_lora_config()
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            **lora_config_dict
        )
        
        logger.info("LoRA設定適用中...")
        model = get_peft_model(model, lora_config)
        
        # Freeze embeddings and LM head
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = False
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"総パラメータ: {total_params:,}")
        logger.info(f"学習可能パラメータ: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
        logger.info("✅ LoRA適用完了")
        
        # Step 3: Create test data
        print_header("ステップ3: テストデータ作成")
        
        test_image, test_prompt, dataset_name = create_test_data()
        logger.info(f"テスト画像: {test_image.size}")
        logger.info(f"テストプロンプト: {test_prompt}")
        logger.info(f"データセット名: {dataset_name}")
        
        # シングルエンコーダー構成用：HybridDatasetの実装に完全準拠
        logger.info("シングルエンコーダー構成で学習用データ準備中（HybridDataset準拠）...")
        
        # 1. SAM用画像処理（HybridDatasetと同じ処理）
        sam_image_size = config_linux.SAM_IMAGE_SIZE  # 1024
        sam_pixel_values = preprocess_sam_image(test_image, sam_image_size)
        if sam_pixel_values.dim() == 4:
            sam_pixel_values = sam_pixel_values.squeeze(0)
        logger.info(f"SAM画像入力生成: shape={sam_pixel_values.shape}")
        
        # 2. Llama-4ネイティブフォーマットでテキスト処理（HybridDataset準拠）
        # プロンプトのクリーンアップ（HybridDatasetと同じ）
        clean_prompt = test_prompt.strip()
        if "[SEG]" in clean_prompt and not clean_prompt.endswith("."):
            clean_prompt = clean_prompt.replace("[SEG]", "[SEG].")
        
        # メッセージフォーマット（HybridDataset形式）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # 画像プレースホルダー
                    {"type": "text", "text": clean_prompt}
                ]
            }
        ]
        
        # apply_chat_templateでLlama-4形式のテキストを生成
        # これにより<|image|>トークンが自動的に挿入される
        # LoRA適用後のモデルでは基底モデルから取得
        if hasattr(model, 'base_model'):
            llama_processor = model.base_model.model.llama_processor if hasattr(model.base_model, 'model') else model.base_model.llama_processor
        else:
            llama_processor = model.llama_processor
            
        formatted_prompt = llama_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # テキストとして取得
        )
        
        # アシスタント応答を追加（HybridDataset形式）
        # セグメンテーション要求を判定（"segment"が含まれるかチェック）
        if "segment" in clean_prompt.lower():
            # セグメンテーションタスクの場合
            formatted_prompt += " Sure, [SEG].</s>"
        else:
            # 通常のVQAタスクの場合（テスト用のダミー応答）
            formatted_prompt += " This is a test response.</s>"
        
        logger.info(f"フォーマット済みプロンプト（最初の100文字）: {formatted_prompt[:100]}...")
        
        # 3. テキストのみをトークン化（画像処理はSAMが行う）
        text_inputs = llama_processor.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=config_linux.MODEL_MAX_LENGTH
        )
        
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        # 4. ラベルを正しく構築（HybridDataset準拠）
        labels = build_correct_labels_for_llama4(input_ids, llama_processor.tokenizer)
        
        # 5. [SEG]トークンマスクの作成
        # config_linux.SEG_TOKENを使用して直接トークンIDを取得
        seg_token = config_linux.SEG_TOKEN  # "[SEG]"
        seg_token_idx = llama_processor.tokenizer.convert_tokens_to_ids(seg_token)
        logger.info(f"[SEG]トークンID: {seg_token_idx}")
        
        seg_token_mask = (input_ids == seg_token_idx)
        
        # 6. ダミーのグラウンドトゥルースマスク（テスト用）
        ground_truth_mask = torch.zeros((1, test_image.size[1], test_image.size[0]), dtype=torch.float32)
        if "segment" in clean_prompt.lower():
            # セグメンテーションタスクの場合、ダミーマスクを生成（左上1/4を赤い領域と仮定）
            ground_truth_mask[0, :test_image.size[1]//2, :test_image.size[0]//2] = 1.0
        
        # 7. 入力データの準備（HybridDataset互換形式）
        inputs = {
            'input_ids': input_ids.unsqueeze(0),  # batch次元を追加
            'attention_mask': attention_mask.unsqueeze(0),
            'sam_pixel_values': sam_pixel_values.unsqueeze(0),
            'labels': labels.unsqueeze(0),
            'seg_token_mask': seg_token_mask.unsqueeze(0),
            'ground_truth_masks': ground_truth_mask.unsqueeze(0),  # batch次元を追加
            'original_sizes': [(test_image.size[1], test_image.size[0])],  # (H, W)形式
            'has_mask': torch.tensor([1 if "segment" in clean_prompt.lower() else 0]),
            'dataset_name': dataset_name
        }
        
        logger.info(f"✅ 学習用データ準備完了（HybridDataset互換）: {list(inputs.keys())}")
        
        # データ構造の検証（verify_dataset_integrityと同じチェック）
        if 'sam_pixel_values' in inputs:
            logger.info(f"✅ SAM画像入力: shape={inputs['sam_pixel_values'].shape}")
        else:
            logger.error("❌ SAM画像入力が存在しません")
        
        if 'pixel_values' in inputs:
            logger.warning("⚠️ pixel_valuesが存在します（シングルエンコーダーでは不要）")
        else:
            logger.info("✅ pixel_valuesなし（シングルエンコーダー構成）")
        
        # <|image|>トークンの確認
        if '<|image|>' in formatted_prompt:
            logger.info("✅ Llama-4ネイティブ<|image|>トークン検出")
        else:
            logger.warning("⚠️ <|image|>トークンが見つかりません")
        
        # [SEG]トークンの確認
        if seg_token_mask.any():
            seg_positions = torch.nonzero(seg_token_mask).squeeze().tolist()
            logger.info(f"✅ [SEG]トークン位置: {seg_positions}")
        else:
            logger.info("ℹ️ [SEG]トークンなし（VQAタスクなど）")
        
        # GPU移動
        first_device = next(iter(model.hf_device_map.values())) if hasattr(model, 'hf_device_map') else next(model.parameters()).device
        inputs = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        logger.info(f"入力準備完了: {list(inputs.keys())}, device: {first_device}")
        
        # Step 4: Initial gradient state
        print_header("ステップ4: 初期勾配状態確認")
        
        model.zero_grad()
        initial_grad_info = analyze_gradients(model)
        logger.info(f"初期状態:")
        logger.info(f"  総パラメータ: {initial_grad_info['total_params']}")
        logger.info(f"  学習可能パラメータ: {initial_grad_info['trainable_params']}")
        logger.info(f"  勾配保持パラメータ: {initial_grad_info['params_with_grad']}")
        
        # Step 5: Forward pass and loss calculation
        print_header("ステップ5: 順伝播と損失計算")
        
        loss, loss_details = verify_loss_calculation(model, inputs, verbose=True)
        logger.info(f"✅ 損失計算成功: {loss.item():.6f}")
        
        # Step 6: Backward pass
        print_header("ステップ6: 逆伝播")
        
        logger.info("逆伝播実行中...")
        loss.backward()
        logger.info("✅ 逆伝播完了")
        
        # Step 7: Post-backward gradient analysis
        print_header("ステップ7: 逆伝播後勾配分析")
        
        post_grad_info = analyze_gradients(model)
        logger.info(f"逆伝播後:")
        logger.info(f"  勾配保持パラメータ: {post_grad_info['params_with_grad']}")
        logger.info(f"  ゼロ勾配パラメータ: {post_grad_info['zero_grad_params']}")
        logger.info(f"  勾配L2ノルム: {post_grad_info['gradient_norm']:.6f}")
        
        logger.info("\nコンポーネント別勾配分析:")
        for component, info in post_grad_info['component_analysis'].items():
            logger.info(f"  {component}: {info['count']}個, ノルム={info['norm']:.6f}")
        
        # Step 8: Gradient flow verification
        print_header("ステップ8: 勾配フロー検証")
        
        if post_grad_info['params_with_grad'] > 0:
            logger.info(f"✅ 勾配フロー正常: {post_grad_info['params_with_grad']}個のパラメータに勾配")
            
            # Check if gradients are meaningful (not all zeros)
            if post_grad_info['gradient_norm'] > 1e-8:
                logger.info(f"✅ 勾配は意味のある値です (L2ノルム: {post_grad_info['gradient_norm']:.6f})")
            else:
                logger.warning(f"⚠️ 勾配が非常に小さいです (L2ノルム: {post_grad_info['gradient_norm']:.6f})")
            
            # Check LoRA components specifically
            lora_components = [comp for comp in post_grad_info['component_analysis'].keys() 
                             if 'lora' in comp.lower() or 'base_layer' in comp.lower()]
            if lora_components:
                logger.info(f"✅ LoRAコンポーネントに勾配フロー: {lora_components}")
            
            # Check multi-modal projector
            projector_components = [comp for comp in post_grad_info['component_analysis'].keys() 
                                  if 'projector' in comp.lower()]
            if projector_components:
                logger.info(f"✅ マルチモーダルプロジェクタに勾配フロー: {projector_components}")
            
            success = True
        else:
            logger.error("❌ 勾配フローが検出されません")
            success = False
        
        # Step 9: Summary
        print_header("検証結果サマリ")
        
        logger.info(f"モデル初期化: ✅")
        logger.info(f"LoRA適用: ✅")
        logger.info(f"データ準備: ✅")
        logger.info(f"順伝播: ✅")
        logger.info(f"損失計算: ✅ (合計損失: {loss.item():.6f})")
        
        # 損失の詳細を表示
        if loss_details:
            logger.info(f"損失内訳:")
            for loss_name, loss_value in loss_details.items():
                if loss_name != 'total_loss' and loss_value is not None:
                    logger.info(f"  - {loss_name}: {loss_value:.6f}")
        
        logger.info(f"逆伝播: ✅")
        logger.info(f"勾配フロー: {'✅' if success else '❌'}")
        
        if success:
            logger.info("🎉 勾配フロー検証成功！LISA-Llama4統合モデルは学習準備完了です")
        else:
            logger.error("❌ 勾配フロー検証失敗")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 検証中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("LISA-Llama4統合モデル 勾配フロー検証開始")
    success = main()
    if success:
        logger.info("🎉 検証完了 - 成功")
        sys.exit(0)
    else:
        logger.error("❌ 検証失敗")
        sys.exit(1) 