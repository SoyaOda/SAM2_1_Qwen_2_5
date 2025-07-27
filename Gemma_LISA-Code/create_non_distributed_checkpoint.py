#!/usr/bin/env python3
"""
非分散環境での事前処理チェックポイント作成スクリプト
DTensor問題を完全に回避するため、分散学習コンポーネントを使用せずに
直接的なモデル操作でリサイズを実行

戦略:
1. 分散学習環境を完全に無効化
2. Gemmaモデルを直接ロード
3. 埋め込み層を手動で拡張
4. SAMとプロジェクションを追加
5. LoRA適用
6. 保存
"""

import os
import sys
import torch
import json
from datetime import datetime
from pathlib import Path

# 分散学習を完全に無効化
os.environ['WORLD_SIZE'] = '0'  # 分散学習を無効化
os.environ['RANK'] = '-1'       # 非分散ランク
if 'LOCAL_RANK' in os.environ:
    del os.environ['LOCAL_RANK']
if 'MASTER_ADDR' in os.environ:
    del os.environ['MASTER_ADDR']
if 'MASTER_PORT' in os.environ:
    del os.environ['MASTER_PORT']

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_linux as config

def create_non_distributed_checkpoint():
    """
    非分散環境で前処理されたチェックポイントを作成
    """
    print("🔧 非分散環境での事前処理チェックポイント作成開始")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 出力ディレクトリの準備（data/に保存で転送作業で消えない）
    output_dir = Path("/lambda/nfs/lisa-gemma-project-fs/data/preprocessed_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_dir / f"lisa_gemma_non_distributed_{timestamp}"
    checkpoint_path.mkdir(exist_ok=True)
    
    print(f"📁 チェックポイント保存先: {checkpoint_path}")
    
    try:
        # Step 1: デバイス設定
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🎯 使用デバイス: {device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Step 2: メモリ最適化
        print("\n🧠 メモリ最適化設定...")
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Step 3: Gemmaモデルの直接初期化（非分散）
        print("\n🚀 Gemmaモデルの直接初期化（非分散環境）...")
        
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            AutoProcessor,
            AutoModelForCausalLM
        )
        
        # Gemma3クラスを安全にインポート（gemma-3-4b-it用）
        try:
            from transformers import Gemma3ForConditionalGeneration
            gemma3_available = True
            print("   ✅ Gemma3ForConditionalGeneration利用可能")
        except ImportError:
            gemma3_available = False
            print("   ⚠️ Gemma3ForConditionalGeneration利用不可、AutoModelForCausalLMを使用")
        
        model_id = config.GEMMA_MODEL_ID
        print(f"   ベースモデル: {model_id}")
        
        # 設定をロード
        print("   設定をロード中...")
        
        # Gemma3Config使用（gemma-3-4b-it用）
        try:
            from transformers import Gemma3Config
            gemma_config = Gemma3Config.from_pretrained(model_id, trust_remote_code=True)
            print("   ✅ Gemma3Config使用")
        except ImportError:
            gemma_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            print("   ⚠️ Gemma3Config利用不可、AutoConfig使用")
        
        # 語彙サイズを事前に拡張
        required_vocab_size = 262209  # SEGトークン1個のみ（仕様書準拠）
        original_vocab_size = getattr(gemma_config, 'vocab_size', 262208)
        
        print(f"   元の語彙サイズ: {original_vocab_size}")
        print(f"   拡張後語彙サイズ: {required_vocab_size}")
        
        gemma_config.vocab_size = required_vocab_size
        
        # モデルをロード（拡張された語彙サイズで）
        print("   Gemma3モデルをロード中...")
        
        # 優先順位付きでモデルをロード（Gemma3を最優先）
        gemma_model = None
        
        if gemma3_available:
            try:
                print("   → Gemma3ForConditionalGenerationを試行...")
                gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id,
                    config=gemma_config,
                    torch_dtype=torch.bfloat16,
                    device_map={"": device},
                    trust_remote_code=True
                )
                print("   ✅ Gemma3ForConditionalGeneration使用成功")
            except Exception as e:
                print(f"   ❌ Gemma3失敗: {e}")
                gemma_model = None
        
        if gemma_model is None:
            try:
                print("   → AutoModelForCausalLMを試行...")
                gemma_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=gemma_config,
                    torch_dtype=torch.bfloat16,
                    device_map={"": device},
                    trust_remote_code=True
                )
                print("   ✅ AutoModelForCausalLM使用成功")
            except Exception as e:
                print(f"   ❌ AutoModelForCausalLM失敗: {e}")
                return False
        
        # 埋め込み層のサイズ確認と手動拡張
        print("\n🔍 埋め込み層の確認と拡張...")
        
        # 強制的な手動リサイズ実装
        def force_resize_embeddings(model, new_vocab_size):
            """確実に埋め込み層をリサイズする関数"""
            print(f"   🔧 強制的な埋め込み層リサイズ: → {new_vocab_size}")
            
            # 入力埋め込み層の処理
            input_embeddings = model.get_input_embeddings()
            old_vocab_size, embed_dim = input_embeddings.weight.shape
            print(f"   入力埋め込み: {old_vocab_size} → {new_vocab_size}")
            
            if old_vocab_size != new_vocab_size:
                import torch.nn as nn
                
                # 新しい入力埋め込み層を作成
                new_input_embeddings = nn.Embedding(
                    new_vocab_size, 
                    embed_dim,
                    padding_idx=input_embeddings.padding_idx,
                    device=input_embeddings.weight.device,
                    dtype=input_embeddings.weight.dtype
                )
                
                # 既存の重みをコピー
                with torch.no_grad():
                    # 既存のトークンの重みをコピー
                    copy_size = min(old_vocab_size, new_vocab_size)
                    new_input_embeddings.weight[:copy_size] = input_embeddings.weight[:copy_size]
                    
                    # 新しいトークンは既存トークンの平均で初期化
                    if new_vocab_size > old_vocab_size:
                        mean_embedding = input_embeddings.weight.mean(dim=0, keepdim=True)
                        new_input_embeddings.weight[old_vocab_size:] = mean_embedding.expand(
                            new_vocab_size - old_vocab_size, -1
                        )
                
                # 入力埋め込み層を置き換え
                model.set_input_embeddings(new_input_embeddings)
                print(f"   ✅ 入力埋め込み層リサイズ完了: {new_input_embeddings.weight.shape}")
            
            # 出力埋め込み層（lm_head）の処理
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                old_output_size, embed_dim = output_embeddings.weight.shape
                print(f"   出力埋め込み: {old_output_size} → {new_vocab_size}")
                
                if old_output_size != new_vocab_size:
                    # 新しい出力埋め込み層を作成
                    new_output_embeddings = nn.Linear(
                        embed_dim,
                        new_vocab_size,
                        bias=output_embeddings.bias is not None,
                        device=output_embeddings.weight.device,
                        dtype=output_embeddings.weight.dtype
                    )
                    
                    # 既存の重みをコピー
                    with torch.no_grad():
                        copy_size = min(old_output_size, new_vocab_size)
                        new_output_embeddings.weight[:copy_size] = output_embeddings.weight[:copy_size]
                        
                        # 新しいトークンは既存トークンの平均で初期化
                        if new_vocab_size > old_output_size:
                            mean_weight = output_embeddings.weight.mean(dim=0, keepdim=True)
                            new_output_embeddings.weight[old_output_size:] = mean_weight.expand(
                                new_vocab_size - old_output_size, -1
                            )
                        
                        # バイアスの処理
                        if output_embeddings.bias is not None:
                            new_output_embeddings.bias[:copy_size] = output_embeddings.bias[:copy_size]
                            if new_vocab_size > old_output_size:
                                new_output_embeddings.bias[old_output_size:] = 0.0
                    
                    # 出力埋め込み層を置き換え
                    model.lm_head = new_output_embeddings
                    print(f"   ✅ 出力埋め込み層リサイズ完了: {new_output_embeddings.weight.shape}")
            
            # 最終検証
            final_input_size = model.get_input_embeddings().weight.shape[0]
            final_output_size = model.get_output_embeddings().weight.shape[0] if model.get_output_embeddings() is not None else "N/A"
            print(f"   🎯 最終語彙サイズ検証:")
            print(f"      入力埋め込み: {final_input_size}")
            print(f"      出力埋め込み: {final_output_size}")
            
            return final_input_size == new_vocab_size
        
        # 現在の状態を確認
        current_input_size = gemma_model.get_input_embeddings().weight.shape[0]
        current_output_size = gemma_model.get_output_embeddings().weight.shape[0] if gemma_model.get_output_embeddings() is not None else 0
        
        print(f"   現在の語彙サイズ:")
        print(f"      入力埋め込み: {current_input_size}")
        print(f"      出力埋め込み: {current_output_size}")
        print(f"   目標語彙サイズ: {required_vocab_size}")
        
        # 強制リサイズを実行
        if current_input_size != required_vocab_size:
            print("   🚀 強制的な埋め込み層リサイズを実行...")
            resize_success = force_resize_embeddings(gemma_model, required_vocab_size)
            
            if resize_success:
                print("   ✅ 埋め込み層リサイズ成功")
            else:
                print("   ❌ 埋め込み層リサイズ失敗")
                return False
        else:
            print("   ✅ 埋め込み層サイズが既に適切です")
        
        # Step 4: トークナイザーとプロセッサーの準備
        print("\n📝 トークナイザーとプロセッサーの準備...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # SEGトークンを追加
        seg_token = config.SEG_TOKEN
        print(f"   SEGトークン '{seg_token}' を追加中...")
        
        special_tokens = {"additional_special_tokens": [seg_token]}
        tokenizer.add_special_tokens(special_tokens)
        
        seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
        print(f"   SEGトークンID: {seg_token_id}")
        
        # Step 5: SAMコンポーネントの準備
        print("\n🎨 SAMコンポーネントの初期化...")
        
        from model.segment_anything import build_sam_vit_h
        
        sam_checkpoint = config.SAM_CHECKPOINT_PATH
        if os.path.exists(sam_checkpoint):
            sam_model = build_sam_vit_h(checkpoint=sam_checkpoint)
            sam_image_encoder = sam_model.image_encoder
            sam_prompt_encoder = sam_model.prompt_encoder
            sam_mask_decoder = sam_model.mask_decoder
            
            # SAMをデバイスに移動
            sam_image_encoder = sam_image_encoder.to(device)
            sam_prompt_encoder = sam_prompt_encoder.to(device)
            sam_mask_decoder = sam_mask_decoder.to(device)
            
            print("   ✅ SAMコンポーネント初期化完了")
        else:
            print(f"   ❌ SAMチェックポイントが見つかりません: {sam_checkpoint}")
            return False
        
        # Step 6: MLPプロジェクタの作成
        print("\n🔗 MLPプロジェクタの作成...")
        
        import torch.nn as nn
        
        sam_prompt_embed_dim = config.SEG_PROJECTION_DIM
        gemma_hidden_size = config.GEMMA_HIDDEN_SIZE
        
        mlp_projector = nn.Sequential(
            nn.Linear(gemma_hidden_size, gemma_hidden_size),
            nn.GELU(),
            nn.Linear(gemma_hidden_size, sam_prompt_embed_dim)
        ).to(device)
        
        print(f"   プロジェクション: {gemma_hidden_size} → {gemma_hidden_size} → {sam_prompt_embed_dim}")
        print("   ✅ MLPプロジェクタ作成完了")
        
        # Step 7: LoRA設定の適用
        print("\n🎯 LoRA設定の適用...")
        
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        gemma_model = get_peft_model(gemma_model, lora_config)
        print("   ✅ LoRA設定適用完了")
        
        # LoRA適用後の語彙サイズ再確認と必要に応じて再リサイズ
        print("\n🔍 LoRA適用後の語彙サイズ再確認...")
        post_lora_input_size = gemma_model.get_input_embeddings().weight.shape[0]
        post_lora_output_size = gemma_model.get_output_embeddings().weight.shape[0] if gemma_model.get_output_embeddings() is not None else 0
        
        print(f"   LoRA適用後の語彙サイズ:")
        print(f"      入力埋め込み: {post_lora_input_size}")
        print(f"      出力埋め込み: {post_lora_output_size}")
        
        # LoRA適用後に語彙サイズが変わった場合は再リサイズ
        if post_lora_input_size != required_vocab_size:
            print("   ⚠️ LoRA適用後に語彙サイズが変更されました！再リサイズを実行...")
            
            # 同じforce_resize_embeddings関数を再使用
            resize_success = force_resize_embeddings(gemma_model, required_vocab_size)
            
            if resize_success:
                print("   ✅ LoRA適用後の再リサイズ成功")
            else:
                print("   ❌ LoRA適用後の再リサイズ失敗")
                return False
        else:
            print("   ✅ LoRA適用後も語彙サイズが正しく保持されています")
        
        # Step 8: 統合モデル状態の作成
        print("\n📦 統合モデル状態の作成...")
        
        # 全コンポーネントの状態辞書を統合
        unified_state_dict = {}
        
        # Gemmaモデル（LoRA適用済み）
        for key, value in gemma_model.state_dict().items():
            unified_state_dict[f"gemma_model.{key}"] = value
        
        # SAMコンポーネント
        for key, value in sam_image_encoder.state_dict().items():
            unified_state_dict[f"sam_image_encoder.{key}"] = value
        
        for key, value in sam_prompt_encoder.state_dict().items():
            unified_state_dict[f"sam_prompt_encoder.{key}"] = value
        
        for key, value in sam_mask_decoder.state_dict().items():
            unified_state_dict[f"sam_mask_decoder.{key}"] = value
        
        # MLPプロジェクタ
        for key, value in mlp_projector.state_dict().items():
            unified_state_dict[f"mlp_projector.{key}"] = value
        
        print(f"   統合状態辞書のキー数: {len(unified_state_dict)}")
        print("   ✅ 統合モデル状態作成完了")
        
        # Step 9: パラメータ情報の計算
        print("\n📊 パラメータ情報の計算...")
        
        total_params = sum(p.numel() for p in unified_state_dict.values())
        trainable_params = sum(p.numel() for p in gemma_model.parameters() if p.requires_grad)
        
        # 最終検証 - より詳細な語彙サイズ確認
        print("\n🔍 最終語彙サイズ検証...")
        final_input_embeddings = gemma_model.get_input_embeddings()
        final_output_embeddings = gemma_model.get_output_embeddings()
        
        final_vocab_size = final_input_embeddings.weight.shape[0]
        final_embed_dim = final_input_embeddings.weight.shape[1]
        final_output_size = final_output_embeddings.weight.shape[0] if final_output_embeddings is not None else 0
        
        print(f"   最終語彙サイズ詳細:")
        print(f"      入力埋め込み形状: {final_input_embeddings.weight.shape}")
        print(f"      出力埋め込み形状: {final_output_embeddings.weight.shape if final_output_embeddings is not None else 'N/A'}")
        print(f"      語彙サイズ: {final_vocab_size}")
        print(f"      埋め込み次元: {final_embed_dim}")
        print(f"      目標サイズ: {required_vocab_size}")
        
        # 語彙サイズが目標と一致しない場合はエラー
        if final_vocab_size != required_vocab_size:
            print(f"   ❌ エラー: 最終語彙サイズ({final_vocab_size})が目標サイズ({required_vocab_size})と一致しません！")
            return False
        
        print(f"   ✅ 語彙サイズ検証成功: {final_vocab_size}")
        
        # 画像トークン範囲の確認
        image_token_start = 262146
        image_token_end = 262401
        print(f"   🖼️ 画像トークン範囲確認: {image_token_start} - {image_token_end}")
        
        if final_vocab_size > image_token_end:
            print(f"   ✅ 画像トークン範囲カバー: OK ({image_token_end} < {final_vocab_size})")
        else:
            print(f"   ❌ 画像トークン範囲カバー: NG ({image_token_end} >= {final_vocab_size})")
            return False
        
        model_info = {
            "timestamp": timestamp,
            "creation_method": "non_distributed_direct_v2",
            "model_id": model_id,
            "vocab_size": final_vocab_size,
            "embed_dim": final_embed_dim,
            "hidden_size": gemma_hidden_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "seg_token_id": seg_token_id,
            "image_token_start": image_token_start,
            "image_token_end": image_token_end,
            "device": str(device),
            "pytorch_version": torch.__version__,
            "success": True
        }
        
        print(f"   最終語彙サイズ: {final_vocab_size}")
        print(f"   総パラメータ数: {total_params:,}")
        print(f"   学習可能パラメータ数: {trainable_params:,}")
        print(f"   LoRA効率: {trainable_params/total_params*100:.2f}%")
        
        # Step 10: チェックポイント保存
        print("\n💾 チェックポイント保存...")
        
        # 統合モデル状態の保存
        model_state_path = checkpoint_path / "pytorch_model.bin"
        torch.save(unified_state_dict, model_state_path)
        print(f"   ✅ 統合モデル状態: {model_state_path}")
        
        # トークナイザー保存
        tokenizer.save_pretrained(checkpoint_path / "tokenizer")
        print(f"   ✅ トークナイザー: {checkpoint_path / 'tokenizer'}")
        
        # プロセッサー保存
        processor.save_pretrained(checkpoint_path / "processor")
        print(f"   ✅ プロセッサー: {checkpoint_path / 'processor'}")
        
        # 設定情報保存
        config_path = checkpoint_path / "model_info.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        print(f"   ✅ モデル情報: {config_path}")
        
        # LISA設定保存
        lisa_config_path = checkpoint_path / "lisa_config.json"
        lisa_config_dict = {
            "gemma_model_id": model_id,
            "sam_checkpoint_path": config.SAM_CHECKPOINT_PATH,
            "seg_token": config.SEG_TOKEN,
            "seg_token_id": seg_token_id,
            "gemma_hidden_size": config.GEMMA_HIDDEN_SIZE,
            "sam_prompt_embed_dim": config.SEG_PROJECTION_DIM,
            "gemma_image_size": config.GEMMA_IMAGE_SIZE,
            "sam_image_size": config.SAM_IMAGE_SIZE,
            "model_max_length": config.MODEL_MAX_LENGTH,
        }
        
        with open(lisa_config_path, 'w', encoding='utf-8') as f:
            json.dump(lisa_config_dict, f, indent=2, ensure_ascii=False)
        print(f"   ✅ LISA設定: {lisa_config_path}")
        
        # Step 11: 成功レポート
        print("\n🎉 非分散事前処理チェックポイント作成完了！")
        print("=" * 80)
        print(f"📁 チェックポイントパス: {checkpoint_path}")
        print(f"📊 最終語彙サイズ: {final_vocab_size}")
        print(f"🧠 総パラメータ: {total_params:,}")
        print(f"🎯 学習可能パラメータ: {trainable_params:,}")
        print(f"⚡ LoRA効率: {trainable_params/total_params*100:.2f}%")
        print(f"🎪 SEGトークンID: {seg_token_id}")
        print("=" * 80)
        
        print("\n🚀 次のステップ:")
        print("1. DeepSpeed学習スクリプトでこのチェックポイントをロード")
        print("2. DTensor問題を完全に回避した分散学習の実行")
        print("3. 画像トークン範囲(262146-262401)への正常アクセス確認")
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print(f"エラータイプ: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 メモリクリーンアップ完了")

if __name__ == "__main__":
    success = create_non_distributed_checkpoint()
    if success:
        print("\n✅ 非分散事前処理チェックポイント作成に成功しました")
        sys.exit(0)
    else:
        print("\n❌ 非分散事前処理チェックポイント作成に失敗しました")
        sys.exit(1) 