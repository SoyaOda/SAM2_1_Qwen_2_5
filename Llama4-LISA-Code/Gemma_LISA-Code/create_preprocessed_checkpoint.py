#!/usr/bin/env python3
"""
シングルGPU環境での事前処理チェックポイント作成スクリプト
DeepSpeed環境で発生するDTensorリサイズ問題を回避するため、
正しい埋め込み層サイズのモデルを事前作成

目的: 
- gemma-3-4b-itをロード
- 埋め込み層を262209サイズにリサイズ（SEGトークン1個のみ）
- DeepSpeed互換フォーマットで保存
- DeepSpeed環境で直接ロード可能な状態にする
"""

import os
import sys
import torch
import json
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
import config_linux as config

def create_preprocessed_checkpoint():
    """
    シングルGPU環境で前処理されたチェックポイントを作成
    """
    print("🔧 シングルGPU事前処理チェックポイント作成開始")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 出力ディレクトリの準備
    output_dir = Path("/lambda/nfs/lisa-gemma-project-fs/data/preprocessed_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_dir / f"lisa_gemma_preprocessed_{timestamp}"
    checkpoint_path.mkdir(exist_ok=True)
    
    print(f"📁 チェックポイント保存先: {checkpoint_path}")
    
    try:
        # Step 1: デバイス設定（シングルGPU強制）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🎯 使用デバイス: {device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Step 2: 設定の準備
        print("\n📋 LISA-Gemma設定の準備...")
        
        lisa_config = LisaGemmaConfig(
            gemma_model_id=config.GEMMA_MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            seg_token=config.SEG_TOKEN,
            gemma_hidden_size=config.GEMMA_HIDDEN_SIZE,
            sam_prompt_embed_dim=config.SEG_PROJECTION_DIM,
            gemma_image_size=config.GEMMA_IMAGE_SIZE,
            sam_image_size=config.SAM_IMAGE_SIZE,
            model_max_length=config.MODEL_MAX_LENGTH,
        )
        
        print(f"   ベースモデル: {lisa_config.gemma_model_id}")
        print(f"   SAMチェックポイント: {lisa_config.sam_checkpoint_path}")
        print(f"   隠れ層サイズ: {lisa_config.gemma_hidden_size}")
        
        # Step 3: メモリ最適化設定
        print("\n🧠 メモリ最適化設定...")
        torch.cuda.empty_cache()
        
        # メモリ効率のための設定
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Step 4: モデル初期化（シングルGPU環境）
        print("\n🚀 LISA-Gemmaモデル初期化（シングルGPU環境）...")
        
        # 分散環境フラグを無効化
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        
        # モデル初期化
        model = LisaGemmaForCausalLM(lisa_config)
        
        print("✅ モデル初期化完了")
        
        # Step 5: デバイス移動
        print("\n📱 モデルのデバイス移動...")
        model = model.to(device)
        print(f"✅ デバイス移動完了: {device}")
        
        # Step 6: 埋め込み層サイズ検証
        print("\n🔍 埋め込み層サイズの最終検証...")
        
        input_embeddings = model.gemma_model.get_input_embeddings()
        actual_vocab_size = input_embeddings.weight.shape[0]
        required_vocab_size = 262209  # SEGトークン1個のみ（仕様書準拠）
        
        print(f"   実際の語彙サイズ: {actual_vocab_size}")
        print(f"   必要最小サイズ: {required_vocab_size}")
        
        if actual_vocab_size >= required_vocab_size:
            print("✅ 埋め込み層サイズが適切です")
        else:
            print(f"❌ 埋め込み層サイズが不足しています（不足: {required_vocab_size - actual_vocab_size}）")
            return False
        
        # Step 7: LoRA設定の適用
        print("\n🔧 LoRA設定の適用...")
        
        # LoRA設定
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # LoRA適用
        model = get_peft_model(model, lora_config)
        print("✅ LoRA設定の適用完了")
        
        # Step 8: モデル情報の収集
        print("\n📊 モデル情報の収集...")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "timestamp": timestamp,
            "model_id": lisa_config.gemma_model_id,
            "vocab_size": actual_vocab_size,
            "hidden_size": lisa_config.gemma_hidden_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "device": str(device),
            "pytorch_version": torch.__version__,
        }
        
        print(f"   総パラメータ数: {total_params:,}")
        print(f"   学習可能パラメータ数: {trainable_params:,}")
        print(f"   LoRA効率: {trainable_params/total_params*100:.2f}%")
        
        # Step 9: チェックポイント保存
        print("\n💾 チェックポイント保存...")
        
        # モデル状態の保存
        model_state_path = checkpoint_path / "pytorch_model.bin"
        torch.save(model.state_dict(), model_state_path)
        print(f"✅ モデル状態保存: {model_state_path}")
        
        # 設定の保存
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        print(f"✅ 設定情報保存: {config_path}")
        
        # LISA設定の保存
        lisa_config_path = checkpoint_path / "lisa_config.json"
        lisa_config_dict = {
            "gemma_model_id": lisa_config.gemma_model_id,
            "sam_checkpoint_path": lisa_config.sam_checkpoint_path,
            "seg_token": lisa_config.seg_token,
            "gemma_hidden_size": lisa_config.gemma_hidden_size,
            "sam_prompt_embed_dim": lisa_config.sam_prompt_embed_dim,
            "gemma_image_size": lisa_config.gemma_image_size,
            "sam_image_size": lisa_config.sam_image_size,
            "model_max_length": lisa_config.model_max_length,
        }
        
        with open(lisa_config_path, 'w', encoding='utf-8') as f:
            json.dump(lisa_config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ LISA設定保存: {lisa_config_path}")
        
        # Step 10: 検証用スクリプトの生成
        print("\n📝 検証用スクリプトの生成...")
        
        validation_script = checkpoint_path / "validate_checkpoint.py"
        validation_code = f'''#!/usr/bin/env python3
"""
事前処理チェックポイントの検証スクリプト
生成時刻: {timestamp}
"""

import torch
import json
from pathlib import Path

def validate_checkpoint():
    checkpoint_dir = Path("{checkpoint_path}")
    
    print("🔍 チェックポイント検証開始")
    print(f"チェックポイントディレクトリ: {{checkpoint_dir}}")
    
    # ファイル存在確認
    required_files = ["pytorch_model.bin", "config.json", "lisa_config.json"]
    for file in required_files:
        file_path = checkpoint_dir / file
        if file_path.exists():
            print(f"✅ {{file}} - 存在")
            if file.endswith('.json'):
                with open(file_path) as f:
                    data = json.load(f)
                print(f"   主要キー: {{list(data.keys())}}")
        else:
            print(f"❌ {{file}} - 欠落")
            return False
    
    # モデル状態の検証
    model_state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location='cpu')
    print(f"✅ モデル状態ロード成功（キー数: {{len(model_state)}}）")
    
    # 埋め込み層の確認
    embedding_key = None
    for key in model_state.keys():
        if 'embed_tokens' in key or 'embeddings' in key:
            embedding_key = key
            break
    
    if embedding_key:
        embedding_shape = model_state[embedding_key].shape
        print(f"✅ 埋め込み層発見: {{embedding_key}} - Shape: {{embedding_shape}}")
        if embedding_shape[0] >= 262209:
            print("✅ 埋め込み層サイズが適切です")
        else:
            print("❌ 埋め込み層サイズが不足しています")
    else:
        print("❌ 埋め込み層が見つかりません")
    
    print("🎉 チェックポイント検証完了")
    return True

if __name__ == "__main__":
    validate_checkpoint()
'''
        
        with open(validation_script, 'w', encoding='utf-8') as f:
            f.write(validation_code)
        print(f"✅ 検証スクリプト生成: {validation_script}")
        
        # Step 11: 成功レポート
        print("\n🎉 事前処理チェックポイント作成完了！")
        print("=" * 80)
        print(f"📁 チェックポイントパス: {checkpoint_path}")
        print(f"📊 語彙サイズ: {actual_vocab_size}")
        print(f"🧠 総パラメータ: {total_params:,}")
        print(f"🎯 学習可能パラメータ: {trainable_params:,}")
        print(f"⚡ LoRA効率: {trainable_params/total_params*100:.2f}%")
        print("=" * 80)
        
        print("\n🚀 次のステップ:")
        print("1. DeepSpeed学習スクリプトでこのチェックポイントをロード")
        print("2. DTensorリサイズ問題を回避した分散学習の実行")
        print(f"3. 検証: python {validation_script}")
        
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
    success = create_preprocessed_checkpoint()
    if success:
        print("\n✅ 事前処理チェックポイント作成に成功しました")
        sys.exit(0)
    else:
        print("\n❌ 事前処理チェックポイント作成に失敗しました")
        sys.exit(1) 