#!/usr/bin/env python3
"""
LISA-Gemma3 段階的検証プロトコル
仕様書第6章「実行および検証プロトコル」に従った実装
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from functools import partial
import time
import json
import argparse

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_linux import *
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss, IoUMetric
from utils.reason_seg_dataset import ReasonSegDataset
from peft import LoraConfig, get_peft_model, TaskType


class VerificationProtocol:
    """仕様書第6章に従った段階的検証プロトコル"""
    
    def __init__(self, skip_successful=False):
        self.use_sam = True  # 常にSAMを使用
        self.skip_successful = skip_successful  # 成功した部分をスキップするオプション
        self.model = None
        self.dataloader = None
        self.loss_fn = None
        self.results = {}
        
    def setup_model(self):
        """モデルとLoRAの設定"""
        if self.skip_successful:
            print("⏭️ モデル設定をスキップ（簡易初期化のみ）")
            # スキップ時も最小限の初期化は行う
            config = LisaGemmaConfig(
                gemma_model_id=GEMMA_MODEL_ID,
                sam_checkpoint_path=SAM_CHECKPOINT_PATH if self.use_sam else "",
                seg_token="[SEG]",
                gemma_hidden_size=2560,
                sam_prompt_embed_dim=256,
            )
            
            print("モデルを簡易初期化中...")
            self.model = LisaGemmaForCausalLM(config)
            self.tokenizer = self.model.gemma_processor.tokenizer
            
            # LoRA設定（詳細な出力は省略）
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            # MLPプロジェクタを訓練可能に設定
            if hasattr(self.model.base_model.model, 'mlp_projector'):
                for param in self.model.base_model.model.mlp_projector.parameters():
                    param.requires_grad = True
            elif hasattr(self.model.base_model, 'mlp_projector'):
                for param in self.model.base_model.mlp_projector.parameters():
                    param.requires_grad = True
            
            print("✅ 簡易モデル設定完了")
            return
            
        print("=" * 60)
        print("🔧 モデルとLoRAの設定")
        print("=" * 60)
        
        # 設定作成
        config = LisaGemmaConfig(
            gemma_model_id=GEMMA_MODEL_ID,
            sam_checkpoint_path=SAM_CHECKPOINT_PATH,
            seg_token="[SEG]",
            gemma_hidden_size=2560,
            sam_prompt_embed_dim=256,
        )
        
        # モデル初期化
        print("モデルを初期化中...")
        self.model = LisaGemmaForCausalLM(config)
        
        # モデルから正しいトークナイザーを取得（[SEG]トークンが既に追加されている）
        self.tokenizer = self.model.gemma_processor.tokenizer
        print(f"トークナイザー語彙サイズ: {len(self.tokenizer)}")
        
        # [SEG]トークンが正しく追加されているか確認
        seg_token_id = self.tokenizer.convert_tokens_to_ids("[SEG]")
        print(f"[SEG]トークンID: {seg_token_id}")
        if seg_token_id == self.tokenizer.unk_token_id:
            print("⚠️ 警告: [SEG]トークンがUNKトークンとして認識されています")
        else:
            print("✅ [SEG]トークンが正しく認識されています")
        
        # LoRA設定
        print("LoRA設定を適用中...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # LoRAアダプタの適用
        self.model = get_peft_model(self.model, lora_config)
        
        # パラメータ設定の詳細表示
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"総パラメータ数: {total_params:,}")
        print(f"訓練可能パラメータ数: {trainable_params:,}")
        print(f"訓練可能パラメータ率: {100 * trainable_params / total_params:.2f}%")
        
        # MLPプロジェクタを訓練可能にする
        if hasattr(self.model.base_model.model, 'mlp_projector'):
            print("MLPプロジェクタを訓練可能に設定中...")
            for param in self.model.base_model.model.mlp_projector.parameters():
                param.requires_grad = True
                print(f"  MLPパラメータ: {param.shape} -> requires_grad = True")
        elif hasattr(self.model.base_model, 'mlp_projector'):
            print("ベースモデルのMLPプロジェクタを訓練可能に設定中...")
            for param in self.model.base_model.mlp_projector.parameters():
                param.requires_grad = True
                print(f"  MLPパラメータ: {param.shape} -> requires_grad = True")
        else:
            print("⚠️ 警告: MLPプロジェクタが見つかりません")
        
        # 更新された訓練可能パラメータ数を表示
        trainable_params_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"MLP追加後の訓練可能パラメータ数: {trainable_params_after:,}")
        print(f"MLP追加後の訓練可能パラメータ率: {100 * trainable_params_after / total_params:.2f}%")
        
        print("✅ モデルとLoRAの設定完了")
        
    def setup_data(self, batch_size=2, samples=10):
        """データセットとデータローダーの設定"""
        if self.skip_successful:
            print("⏭️ データ設定をスキップ（簡易初期化のみ）")
            # スキップ時も最小限の初期化は行う
            gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            
            # [SEG]トークンをプロセッサーのトークナイザーにも追加
            seg_token = "[SEG]"
            if seg_token not in gemma_processor.tokenizer.get_vocab():
                gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
            
            # データセット作成
            dataset = ReasonSegDataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=gemma_processor.tokenizer,
                samples_per_epoch=samples,
                reason_seg_data="ReasonSeg|train",
                explanatory=0.1
            )
            
            # collate_fn定義（簡略版）
            def collate_fn(batch):
                valid_samples = []
                for item in batch:
                    if len(item) >= 5:
                        image_path, pil_image, conversation_text, mask_tensor, label_tensor = item
                        if pil_image is not None and conversation_text is not None and conversation_text.strip():
                            valid_samples.append((pil_image, conversation_text, mask_tensor))
                
                if not valid_samples:
                    return {
                        'input_ids': torch.empty(0, dtype=torch.long),
                        'attention_mask': torch.empty(0, dtype=torch.long),
                        'pixel_values': torch.empty(0),
                        'labels': torch.empty(0, dtype=torch.long),
                        'images': [],
                        'masks': None,
                    }
                
                images = [sample[0] for sample in valid_samples]
                texts = [sample[1] for sample in valid_samples]
                masks = [sample[2] for sample in valid_samples if sample[2] is not None and sample[2].numel() > 0]
                
                # 簡易処理（詳細出力は省略）
                processed_samples = []
                for i, (image, text) in enumerate(zip(images, texts)):
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": text}
                            ]
                        }
                    ]
                    
                    processed = gemma_processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_tensors="pt",
                        return_dict=True,
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    processed_samples.append(processed)
                
                # バッチ化（パディング処理を追加）
                max_length = max(s['input_ids'].size(1) for s in processed_samples)
                
                batch_input_ids = []
                batch_attention_mask = []
                batch_pixel_values = []
                
                for s in processed_samples:
                    input_ids = s['input_ids'].squeeze(0)
                    attention_mask = s['attention_mask'].squeeze(0)
                    pixel_values = s['pixel_values'].squeeze(0)
                    
                    # パディング
                    if input_ids.size(0) < max_length:
                        pad_length = max_length - input_ids.size(0)
                        input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=input_ids.dtype)])
                        attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=attention_mask.dtype)])
                    
                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_pixel_values.append(pixel_values)
                
                batch_input_ids = torch.stack(batch_input_ids)
                batch_attention_mask = torch.stack(batch_attention_mask)
                batch_pixel_values = torch.stack(batch_pixel_values)
                
                return {
                    'input_ids': batch_input_ids,
                    'attention_mask': batch_attention_mask,
                    'pixel_values': batch_pixel_values,
                    'labels': batch_input_ids,
                    'images': images,
                    'masks': masks if masks else None,
                }
            
            # データローダー作成
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            print("✅ 簡易データ設定完了")
            return
            
        print("=" * 60)
        print("📊 データセットとデータローダーの設定")
        print("=" * 60)
        
        # Gemma-3プロセッサーの初期化
        gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
        
        # [SEG]トークンをプロセッサーのトークナイザーにも追加
        # （モデルとデータセットで一貫性を保つため）
        seg_token = "[SEG]"
        if seg_token not in gemma_processor.tokenizer.get_vocab():
            print(f"プロセッサーのトークナイザーに{seg_token}トークンを追加中...")
            gemma_processor.tokenizer.add_tokens([seg_token], special_tokens=True)
            print(f"プロセッサーのトークナイザー語彙サイズ: {len(gemma_processor.tokenizer)}")
        
        # データセット作成（test_real_datasets.pyで成功したReasonSegDatasetを使用）
        dataset = ReasonSegDataset(
            base_image_dir=DATASET_BASE_DIR,
            tokenizer=gemma_processor.tokenizer,
            samples_per_epoch=samples,
            reason_seg_data="ReasonSeg|train",
            explanatory=0.1
        )
        
        # データローダー作成（カスタムcollate_fnを使用）
        def collate_fn(batch):
            """
            バッチ処理用のカスタムcollate関数
            ReasonSegDatasetの出力形式: (image_path, pil_image, conversation_text, masks, label)
            Gemma-3プロセッサーのapply_chat_templateメソッドを使用
            """
            valid_samples = []
            
            # 有効なサンプルのみを抽出
            for item in batch:
                if len(item) >= 5:
                    image_path, pil_image, conversation_text, mask_tensor, label_tensor = item
                    # 画像とテキストが有効かチェック
                    if pil_image is not None and conversation_text is not None and conversation_text.strip():
                        valid_samples.append((pil_image, conversation_text, mask_tensor))
            
            if not valid_samples:
                # 有効なサンプルがない場合は空のバッチを返す
                return {
                    'input_ids': torch.empty(0, dtype=torch.long),
                    'attention_mask': torch.empty(0, dtype=torch.long),
                    'pixel_values': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'images': [],
                    'masks': None,
                }
            
            # 有効なサンプルから画像、テキスト、マスクを分離
            images = [sample[0] for sample in valid_samples]
            texts = [sample[1] for sample in valid_samples]
            masks = [sample[2] for sample in valid_samples if sample[2] is not None and sample[2].numel() > 0]
            
            print(f"🔍 Collate処理: {len(images)} 画像, {len(texts)} テキスト")
            
            # Gemma-3プロセッサーでapply_chat_templateを使用
            try:
                # 各サンプルを個別に処理してからバッチ化
                processed_samples = []
                
                for i, (image, text) in enumerate(zip(images, texts)):
                    try:
                        # Gemma-3形式のメッセージを作成
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": text}
                                ]
                            }
                        ]
                        
                        # apply_chat_templateを使用して処理
                        processed = gemma_processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            add_generation_prompt=False,
                            do_pan_and_scan=False  # 高速化のため無効
                        )
                        
                        processed_samples.append(processed)
                        print(f"  ✅ サンプル{i+1}: 成功")
                        
                    except Exception as e:
                        print(f"  ❌ サンプル{i+1}: エラー - {e}")
                        # エラーの場合はテキストのみ処理
                        try:
                            tokenized = gemma_processor.tokenizer(
                                text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=MODEL_MAX_LENGTH,
                            )
                            processed_samples.append({
                                'input_ids': tokenized['input_ids'],
                                'attention_mask': tokenized['attention_mask'],
                                'pixel_values': torch.empty(0),
                            })
                        except Exception as e2:
                            print(f"    ❌ テキストのみ処理も失敗: {e2}")
                            continue
                
                if processed_samples:
                    # バッチ化（パディング処理を追加）
                    max_length = max(s['input_ids'].size(1) for s in processed_samples)
                    
                    batch_input_ids = []
                    batch_attention_mask = []
                    batch_pixel_values = []
                    
                    for s in processed_samples:
                        input_ids = s['input_ids'].squeeze(0)
                        attention_mask = s['attention_mask'].squeeze(0)
                        pixel_values = s['pixel_values'].squeeze(0)
                        
                        # パディング
                        if input_ids.size(0) < max_length:
                            pad_length = max_length - input_ids.size(0)
                            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=input_ids.dtype)])
                            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=attention_mask.dtype)])
                        
                        batch_input_ids.append(input_ids)
                        batch_attention_mask.append(attention_mask)
                        batch_pixel_values.append(pixel_values)
                    
                    batch_input_ids = torch.stack(batch_input_ids)
                    batch_attention_mask = torch.stack(batch_attention_mask)
                    batch_pixel_values = torch.stack(batch_pixel_values)
                    
                    print(f"✅ バッチ処理成功:")
                    print(f"  - input_ids: {batch_input_ids.shape}")
                    print(f"  - attention_mask: {batch_attention_mask.shape}")
                    print(f"  - pixel_values: {batch_pixel_values.shape}")
                    
                    return {
                        'input_ids': batch_input_ids,
                        'attention_mask': batch_attention_mask,
                        'pixel_values': batch_pixel_values,
                        'labels': batch_input_ids.clone(),
                        'images': images,
                        'masks': masks if masks else None,
                    }
                
            except Exception as e:
                print(f"❌ apply_chat_template処理エラー: {e}")
                print("テキストのみで処理します...")
                
            # フォールバック: テキストのみを処理
            try:
                tokenized = gemma_processor.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MODEL_MAX_LENGTH,
                )
                
                print(f"✅ テキストのみ処理成功: {len(texts)} サンプル")
                
                return {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'pixel_values': torch.empty(0),  # 空のテンソル
                    'labels': tokenized['input_ids'].clone(),
                    'images': images,
                    'masks': masks if masks else None,
                }
                
            except Exception as e:
                print(f"❌ テキスト処理エラー: {e}")
                return {
                    'input_ids': torch.empty(0, dtype=torch.long),
                    'attention_mask': torch.empty(0, dtype=torch.long),
                    'pixel_values': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'images': images,
                    'masks': masks if masks else None,
                }
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 損失関数設定
        self.loss_fn = CompositeLoss(
            ce_loss_weight=CE_LOSS_WEIGHT,
            dice_loss_weight=DICE_LOSS_WEIGHT,
            bce_loss_weight=BCE_LOSS_WEIGHT
        )
        
        print(f"✅ データセット作成完了: {len(dataset)} サンプル")
        print(f"✅ データローダー作成完了: バッチサイズ {batch_size}")
        
        # データセットから実際に1つのサンプルを取得してテスト
        try:
            sample = dataset[0]
            print(f"✅ サンプル取得テスト成功: {type(sample)}")
            if isinstance(sample, tuple):
                print(f"   タプル長: {len(sample)}")
                print(f"   要素タイプ: {[type(item).__name__ for item in sample]}")
                if len(sample) >= 3 and isinstance(sample[2], str):
                    print(f"   テキストサンプル: {sample[2][:100]}...")
        except Exception as e:
            print(f"⚠️ サンプル取得テストでエラー: {e}")
    
    def validate_data_format(self):
        """データ形式の検証"""
        print("=" * 60)
        print("📊 データ形式の検証")
        print("=" * 60)
        
        try:
            # データローダーから1バッチ取得
            batch = next(iter(self.dataloader))
            
            print("✅ データローダーからバッチ取得成功")
            print(f"バッチサイズ: {len(batch['input_ids'])}")
            
            # 入力形式の確認
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch['pixel_values']
            labels = batch['input_ids']
            
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
            print(f"Pixel values shape: {pixel_values.shape}")
            print(f"Labels shape: {labels.shape}")
            
            print("\n🔍 テキストプロンプトの検証:")
            for i in range(min(2, input_ids.shape[0])):
                # モデルのトークナイザーを使用（[SEG]トークンが含まれている）
                decoded = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
                print(f"  サンプル{i+1}: {decoded[:100]}...")
                
                # SEGトークンの存在確認
                seg_token_id = self.tokenizer.convert_tokens_to_ids("[SEG]")
                has_seg = seg_token_id in input_ids[i]
                print(f"    [SEG]トークン: {'✅ 存在' if has_seg else '❌ なし'}")
                print(f"    [SEG]トークンID: {seg_token_id}")
                
                # 実際のトークンIDを確認
                seg_positions = (input_ids[i] == seg_token_id).nonzero(as_tuple=True)[0]
                if len(seg_positions) > 0:
                    print(f"    [SEG]トークン位置: {seg_positions.tolist()}")
                else:
                    # デバッグ用：全体のテキストを表示
                    full_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
                    print(f"    完全なテキスト: {full_text}")
                    
                    # [SEG]文字列が含まれているかチェック
                    if "[SEG]" in full_text:
                        print("    ⚠️ テキストには[SEG]が含まれているが、正しくトークン化されていない")
                    else:
                        print("    ⚠️ テキストに[SEG]が含まれていない")
            
            return True
            
        except Exception as e:
            print(f"❌ データ検証エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step1_data_sanity_check(self):
        """ステップ1: データ健全性チェック（仕様書第6章.2.1）"""
        if self.skip_successful:
            print("⏭️ ステップ1をスキップ（既に成功済み）")
            # スキップ時は最初のバッチを返す
            try:
                return next(iter(self.dataloader))
            except:
                return None
                
        print("\n" + "=" * 60)
        print("📋 ステップ1: データ健全性チェック")
        print("=" * 60)
        
        try:
            # データ形式の詳細検証
            data_valid = self.validate_data_format()
            if not data_valid:
                self.results['step1'] = {'status': 'failed', 'error': 'データ形式検証失敗'}
                return None
            
            # 1バッチ取得
            batch = next(iter(self.dataloader))
            
            print("🔍 バッチ内容の検証:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, list):
                    print(f"  {key}: list of {len(value)} items")
                else:
                    print(f"  {key}: {type(value)}")
            
            # 画像テンソル形状の検証
            pixel_values = batch['pixel_values']
            expected_gemma_shape = (len(batch['images']), 3, GEMMA_IMAGE_SIZE, GEMMA_IMAGE_SIZE)
            
            if pixel_values.shape == expected_gemma_shape:
                print(f"✅ Gemma画像形状が正しい: {pixel_values.shape}")
            else:
                print(f"❌ Gemma画像形状が不正: 期待{expected_gemma_shape}, 実際{pixel_values.shape}")
            
            # マスクの可視化（最初のサンプルのみ）
            if batch['masks'] and len(batch['masks']) > 0:
                self._visualize_mask(batch['images'][0], batch['masks'][0], "step1_mask_overlay.png")
                print(f"✅ マスクの可視化を保存: step1_mask_overlay.png")
            
            self.results['step1'] = {'status': 'success', 'batch_info': {k: str(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) for k, v in batch.items()}}
            print("✅ ステップ1: データ健全性チェック完了")
            return batch
            
        except Exception as e:
            print(f"❌ ステップ1失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step1'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step2_forward_pass_test(self, batch):
        """ステップ2: フォワードパス・テスト（仕様書第6章.2.2）"""
        if self.skip_successful:
            print("⏭️ ステップ2をスキップ（既に成功済み）")
            return {'logits': torch.empty(0)}  # ダミー出力を返す
            
        print("\n" + "=" * 60)
        print("🚀 ステップ2: フォワードパス・テスト")
        print("=" * 60)
        
        if batch is None:
            print("❌ バッチが無効のためスキップ")
            return None
            
        try:
            self.model.eval()
            device = next(self.model.parameters()).device
            
            # バッチをGPUに移動
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            print("🔍 フォワードパス実行中...")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['input_ids'],
                    generate_mask=self.use_sam,
                )
            
            print("🔍 出力の検証:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                    
                    # 次元不一致エラーの検証
                    if 'mask' in key.lower() and self.use_sam:
                        print(f"    マスク値範囲: [{value.min().item():.4f}, {value.max().item():.4f}]")
                elif value is not None:
                    print(f"  {key}: {type(value)}")
                else:
                    print(f"  {key}: None")
            
            # 期待する出力の検証
            required_outputs = ['logits']
            if self.use_sam:
                required_outputs.append('predicted_masks')
                
            for req_output in required_outputs:
                if req_output in outputs and outputs[req_output] is not None:
                    print(f"✅ {req_output}: 正常")
                else:
                    print(f"⚠️ {req_output}: 欠損または None")
            
            self.results['step2'] = {'status': 'success', 'outputs': list(outputs.keys())}
            print("✅ ステップ2: フォワードパス・テスト完了")
            return outputs
            
        except Exception as e:
            print(f"❌ ステップ2失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step2'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step3_single_batch_overfitting(self, batch, max_steps=100):
        """ステップ3: 単一バッチ過学習（仕様書第6章.2.3 - 最も重要なテスト）"""
        if self.skip_successful:
            print("⏭️ ステップ3をスキップ（既に成功済み）")
            return [1.0, 0.1]  # ダミーの損失リストを返す
            
        print("\n" + "=" * 60)
        print("🎯 ステップ3: 単一バッチ過学習（最重要テスト）")
        print("=" * 60)
        
        if batch is None:
            print("❌ バッチが無効のためスキップ")
            return None
            
        try:
            self.model.train()
            device = next(self.model.parameters()).device
            
            # バッチをGPUに移動
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # オプティマイザ設定
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
            
            print(f"🎯 {max_steps}ステップの過学習開始...")
            losses = []
            
            for step in range(max_steps):
                optimizer.zero_grad()
                
                # フォワードパス
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['input_ids'],
                    generate_mask=self.use_sam,
                )
                
                # 損失計算
                masks = batch.get('masks')
                if masks is not None and isinstance(masks, list):
                    # リストの場合はテンソルに変換
                    # 異なるサイズのマスクを統一サイズにリサイズ
                    processed_masks = []
                    target_size = (256, 256)  # SAMの出力サイズに合わせる
                    
                    for mask in masks:
                        # マスクをGPUに移動
                        mask = mask.to(device)
                        
                        if mask.dim() == 3:  # (C, H, W)
                            mask_resized = F.interpolate(
                                mask.unsqueeze(0), 
                                size=target_size, 
                                mode='nearest'
                            ).squeeze(0)
                        elif mask.dim() == 2:  # (H, W)
                            mask_resized = F.interpolate(
                                mask.unsqueeze(0).unsqueeze(0), 
                                size=target_size, 
                                mode='nearest'
                            ).squeeze(0).squeeze(0)
                        else:
                            mask_resized = mask
                        processed_masks.append(mask_resized)
                    
                    masks = torch.stack(processed_masks, dim=0)
                
                batch_for_loss = {
                    'labels': batch['input_ids'],
                    'ground_truth_mask': masks
                }
                
                loss_dict = self.loss_fn(outputs, batch_for_loss)
                total_loss = loss_dict['total_loss']
                
                # バックワード
                total_loss.backward()
                optimizer.step()
                
                losses.append(total_loss.item())
                
                # ログ出力
                if step % 10 == 0 or step == max_steps - 1:
                    print(f"  Step {step:3d}: Loss = {total_loss.item():.4f}")
            
            # 収束判定
            initial_loss = losses[0]
            final_loss = losses[-1]
            reduction_ratio = (initial_loss - final_loss) / initial_loss
            
            print(f"\n📊 過学習結果:")
            print(f"  初期損失: {initial_loss:.4f}")
            print(f"  最終損失: {final_loss:.4f}")
            print(f"  損失減少率: {reduction_ratio*100:.2f}%")
            
            # 成功基準の判定
            success_threshold = 0.1  # 10%以上の損失減少
            if reduction_ratio > success_threshold:
                print(f"✅ 過学習成功: 損失が{reduction_ratio*100:.1f}%減少（閾値{success_threshold*100}%以上）")
                status = 'success'
            else:
                print(f"⚠️ 過学習不十分: 損失減少{reduction_ratio*100:.1f}%（閾値{success_threshold*100}%未満）")
                status = 'insufficient'
            
            # 損失カーブの保存
            self._plot_loss_curve(losses, "step3_overfitting_curve.png")
            
            self.results['step3'] = {
                'status': status,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'reduction_ratio': reduction_ratio,
                'losses': losses
            }
            
            print("✅ ステップ3: 単一バッチ過学習完了")
            return losses
            
        except Exception as e:
            print(f"❌ ステップ3失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step3'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step4_gradient_flow_inspection(self):
        """ステップ4: 勾配フロー検査（仕様書第6章.2.4）"""
        print("\n" + "=" * 60)
        print("🔍 ステップ4: 勾配フロー検査")
        print("=" * 60)
        
        try:
            # 勾配検査のために1回のフォワード・バックワードパスを実行
            print("🔍 勾配計算のためのフォワード・バックワードパス実行中...")
            
            self.model.train()
            device = next(self.model.parameters()).device
            
            # データローダーから1バッチ取得
            batch = next(iter(self.dataloader))
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # 勾配をクリア
            self.model.zero_grad()
            
            # フォワードパス
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                labels=batch['input_ids'],
                generate_mask=self.use_sam,
            )
            
            # 損失計算
            masks = batch.get('masks')
            if masks is not None and isinstance(masks, list):
                # リストの場合はテンソルに変換
                # 異なるサイズのマスクを統一サイズにリサイズ
                processed_masks = []
                target_size = (256, 256)  # SAMの出力サイズに合わせる
                
                for mask in masks:
                    # マスクをGPUに移動
                    mask = mask.to(device)
                    
                    if mask.dim() == 3:  # (C, H, W)
                        mask_resized = F.interpolate(
                            mask.unsqueeze(0), 
                            size=target_size, 
                            mode='nearest'
                        ).squeeze(0)
                    elif mask.dim() == 2:  # (H, W)
                        mask_resized = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0), 
                            size=target_size, 
                            mode='nearest'
                        ).squeeze(0).squeeze(0)
                    else:
                        mask_resized = mask
                    processed_masks.append(mask_resized)
                
                masks = torch.stack(processed_masks, dim=0)
            
            batch_for_loss = {
                'labels': batch['input_ids'],
                'ground_truth_mask': masks
            }
            
            loss_dict = self.loss_fn(outputs, batch_for_loss)
            total_loss = loss_dict['total_loss']
            
            print(f"  計算された損失: {total_loss.item():.4f}")
            print(f"  損失のrequires_grad: {total_loss.requires_grad}")
            
            # バックワードパス
            total_loss.backward()
            print("  ✅ バックワードパス完了")
            
            print("\n🔍 訓練可能パラメータの勾配確認:")
            
            gradient_info = {
                'with_gradients': [],
                'without_gradients': [],
                'total_trainable': 0,
                'total_with_grad': 0
            }
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    gradient_info['total_trainable'] += 1
                    
                    if param.grad is not None:
                        gradient_info['total_with_grad'] += 1
                        gradient_info['with_gradients'].append({
                            'name': name,
                            'shape': list(param.shape),
                            'grad_norm': param.grad.norm().item()
                        })
                        print(f"  ✅ {name}: grad_norm = {param.grad.norm().item():.6f}")
                    else:
                        gradient_info['without_gradients'].append(name)
                        print(f"  ❌ {name}: grad = None")
            
            print(f"\n📊 勾配フロー統計:")
            print(f"  訓練可能パラメータ数: {gradient_info['total_trainable']}")
            print(f"  勾配を持つパラメータ数: {gradient_info['total_with_grad']}")
            print(f"  勾配フロー率: {gradient_info['total_with_grad']/gradient_info['total_trainable']*100:.1f}%")
            
            # 重要コンポーネントの勾配確認
            print(f"\n🔍 重要コンポーネントの勾配確認:")
            important_components = {
                'MLP Projector': 'mlp_projector',
                'LoRA Adapters': 'lora',
                'SAM Decoder': 'sam_mask_decoder' if self.use_sam else None
            }
            
            for comp_name, comp_key in important_components.items():
                if comp_key is None:
                    continue
                    
                comp_grads = [info for info in gradient_info['with_gradients'] if comp_key in info['name']]
                if comp_grads:
                    avg_grad_norm = np.mean([info['grad_norm'] for info in comp_grads])
                    print(f"  ✅ {comp_name}: {len(comp_grads)}個のパラメータ, 平均勾配ノルム = {avg_grad_norm:.6f}")
                else:
                    print(f"  ❌ {comp_name}: 勾配なし")
            
            # 成功判定
            success_rate = gradient_info['total_with_grad'] / gradient_info['total_trainable']
            if success_rate > 0.9:  # 90%以上
                status = 'success'
                print("✅ 勾配フロー検査成功: 大部分のパラメータに勾配が流れています")
            else:
                status = 'partial'
                print(f"⚠️ 勾配フロー不完全: {success_rate*100:.1f}%のパラメータにのみ勾配")
            
            self.results['step4'] = {
                'status': status,
                'gradient_flow_rate': success_rate,
                'gradient_info': gradient_info
            }
            
            print("✅ ステップ4: 勾配フロー検査完了")
            return gradient_info
            
        except Exception as e:
            print(f"❌ ステップ4失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step4'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step5_results_interpretation(self):
        """ステップ5: 結果の解釈と定性的評価（仕様書第6章.2.5）"""
        print("\n" + "=" * 60)
        print("📈 ステップ5: 結果の解釈と定性的評価")
        print("=" * 60)
        
        try:
            print("📊 全体的な検証結果サマリー:")
            
            total_steps = 5
            successful_steps = 0
            
            for step_num in range(1, total_steps + 1):
                step_key = f'step{step_num}'
                if step_key in self.results:
                    status = self.results[step_key]['status']
                    if status == 'success':
                        successful_steps += 1
                        print(f"  ✅ ステップ{step_num}: 成功")
                    elif status == 'partial' or status == 'insufficient':
                        print(f"  ⚠️ ステップ{step_num}: 部分的成功")
                    else:
                        print(f"  ❌ ステップ{step_num}: 失敗")
                else:
                    print(f"  ❓ ステップ{step_num}: 未実行")
            
            success_rate = successful_steps / total_steps
            print(f"\n📊 総合成功率: {success_rate*100:.1f}% ({successful_steps}/{total_steps})")
            
            # 推奨事項
            print(f"\n💡 推奨事項:")
            
            if success_rate >= 0.8:
                print("  🎉 優秀な結果です！本格的な学習を開始できます。")
                recommendation = "ready_for_training"
            elif success_rate >= 0.6:
                print("  👍 基本的な動作は確認できました。一部の問題を修正後、学習を開始してください。")
                recommendation = "minor_fixes_needed"
            else:
                print("  ⚠️ 重要な問題があります。学習前に根本的な修正が必要です。")
                recommendation = "major_fixes_needed"
            
            # 具体的な修正提案
            if 'step3' in self.results and self.results['step3']['status'] != 'success':
                print("    - 単一バッチ過学習が不十分です。学習率やモデル設定を確認してください。")
            
            if 'step4' in self.results and self.results['step4']['status'] != 'success':
                print("    - 勾配フローに問題があります。パラメータの凍結設定を確認してください。")
            
            # IoU評価（SAM使用時）
            if self.use_sam and hasattr(self, 'model') and self.model.sam_mask_decoder:
                print("\n🎯 セグメンテーション性能評価:")
                iou_metric = IoUMetric()
                print("  セグメンテーション機能が有効です。")
            else:
                print("\n🎯 セグメンテーション性能評価:")
                print("  SAMが無効のため、セグメンテーション評価はスキップされました。")
            
            self.results['step5'] = {
                'status': 'success',
                'success_rate': success_rate,
                'recommendation': recommendation,
                'successful_steps': successful_steps,
                'total_steps': total_steps
            }
            
            print("✅ ステップ5: 結果の解釈と定性的評価完了")
            return self.results
            
        except Exception as e:
            print(f"❌ ステップ5失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step5'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def _visualize_mask(self, pil_image, mask, filename):
        """マスクの可視化"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 元画像
            axes[0].imshow(pil_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # マスク
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            # オーバーレイ
            overlay = np.array(pil_image)
            if len(mask_np.shape) == 2:
                mask_colored = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
                mask_colored[:, :, 0] = mask_np  # 赤チャンネル
                overlay = overlay * 0.7 + mask_colored * 0.3 * 255
            
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"可視化エラー: {e}")
    
    def _plot_loss_curve(self, losses, filename):
        """損失カーブのプロット"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, 'b-', linewidth=2)
            plt.title('Single Batch Overfitting - Loss Curve')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"プロットエラー: {e}")
    
    def run_full_verification(self, use_sam=None):
        """完全な検証プロトコルの実行"""
        if use_sam is not None:
            self.use_sam = use_sam
            
        print("🚀 LISA-Gemma3 段階的検証プロトコル開始")
        print("仕様書第6章「実行および検証プロトコル」に従った実装")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # セットアップ
            self.setup_model()
            self.setup_data()
            
            # 損失関数の初期化
            from model.losses import CompositeLoss
            self.loss_fn = CompositeLoss()
            
            # 段階的検証実行
            batch = self.step1_data_sanity_check()
            outputs = self.step2_forward_pass_test(batch)
            losses = self.step3_single_batch_overfitting(batch)
            gradients = self.step4_gradient_flow_inspection()
            results = self.step5_results_interpretation()
            
            # 実行時間
            elapsed_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("🎯 検証プロトコル完了")
            print("=" * 80)
            print(f"⏱️ 実行時間: {elapsed_time:.1f}秒")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ 検証プロトコル実行エラー: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='LISA-Gemma3 段階的検証プロトコル')
    parser.add_argument('--skip-successful', action='store_true', 
                       help='成功済みのステップをスキップ（ステップ1-3）')
    parser.add_argument('--step', type=int, choices=[1,2,3,4,5], 
                       help='特定のステップのみを実行')

    parser.add_argument('--gradient-only', action='store_true',
                       help='勾配フロー検査のみを実行（ステップ4）')
    
    args = parser.parse_args()
    
    print("LISA-Gemma3 段階的検証プロトコル")
    print("=" * 80)
    
    if args.gradient_only:
        # 勾配フロー検査のみを実行
        print("\n🔍 勾配フロー検査のみを実行...")
        protocol = VerificationProtocol(skip_successful=True)
        protocol.setup_model()
        protocol.setup_data()
        
        # 損失関数の初期化
        from model.losses import CompositeLoss
        protocol.loss_fn = CompositeLoss()
        
        gradient_info = protocol.step4_gradient_flow_inspection()
        return
    
    if args.step:
        # 特定のステップのみを実行
        print(f"\n🎯 ステップ{args.step}のみを実行...")
        protocol = VerificationProtocol(skip_successful=False)
        protocol.setup_model()
        protocol.setup_data()
        
        # 損失関数の初期化
        from model.losses import CompositeLoss
        protocol.loss_fn = CompositeLoss()
        
        if args.step == 1:
            batch = protocol.step1_data_sanity_check()
        elif args.step == 2:
            batch = protocol.step1_data_sanity_check()
            protocol.step2_forward_pass_test(batch)
        elif args.step == 3:
            batch = protocol.step1_data_sanity_check()
            protocol.step3_single_batch_overfitting(batch)
        elif args.step == 4:
            protocol.step4_gradient_flow_inspection()
        elif args.step == 5:
            protocol.step5_results_interpretation()
        return
    
    # 通常の実行（全ステップまたはスキップ付き）
    use_skip = args.skip_successful
    
    # SAMありでの検証（常にSAMを使用）
    if os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"\n🔧 SAMありでの検証開始{'（スキップモード）' if use_skip else ''}...")
        protocol = VerificationProtocol(skip_successful=use_skip)
        results = protocol.run_full_verification()
    else:
        print(f"\n⚠️ SAMチェックポイントが見つかりません: {SAM_CHECKPOINT_PATH}")
        print("SAMチェックポイントを配置してから再実行してください。")
        return
    
    # 最終レポート
    print("\n" + "=" * 80)
    print("📋 最終検証レポート")
    print("=" * 80)
    
    if results:
        success_rate = results.get('step5', {}).get('success_rate', 0)
        print(f"SAMあり検証: {success_rate*100:.1f}% 成功")
    
    print("\n🎯 次のステップ:")
    print("1. 検証結果を確認し、必要に応じて修正を実施")
    print("2. 成功率が80%以上の場合、本格的な学習を開始")
    print("3. DeepSpeedを使用した分散学習の実行")
    
    print("\n💡 使用例:")
    print("  python test_verification_protocol.py --skip-successful    # 成功済みステップをスキップ")
    print("  python test_verification_protocol.py --gradient-only      # 勾配フロー検査のみ")
    print("  python test_verification_protocol.py --step 4             # ステップ4のみ実行")



if __name__ == "__main__":
    main()