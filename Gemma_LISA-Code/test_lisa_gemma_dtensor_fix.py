#!/usr/bin/env python3
"""
LISA-Gemma DTensor問題解決テスト
バージョンダウングレード後の実際のモデルテスト
"""

import os
import sys
import torch
import warnings
from datetime import datetime

# 環境変数設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def print_banner(title):
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def main():
    print_banner("LISA-Gemma DTensor問題解決テスト")
    print(f"実行時刻: {datetime.now()}")
    
    # Step 1: 基本環境確認
    print_banner("Step 1: 基本環境確認")
    import torch
    import transformers
    import accelerate
    
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ transformers: {transformers.__version__}")
    print(f"✅ accelerate: {accelerate.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    
    # Step 2: LISA-Gemmaモデル初期化テスト
    print_banner("Step 2: LISA-Gemmaモデル初期化テスト")
    
    try:
        # config_linux.py読み込み
        sys.path.append('.')
        import config_linux
        model_path = config_linux.GEMMA_MODEL_ID
        print("✅ config_linux.py loaded")
        
        from model.gemma_lisa import LisaGemmaForCausalLM
        print("✅ LisaGemmaForCausalLM imported")
        
        # モデル初期化
        print("🔧 モデル初期化中...")
        model = LisaGemmaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ モデル初期化成功")
        
        # 語彙サイズ確認
        current_vocab = model.gemma_model.get_input_embeddings().num_embeddings
        print(f"✅ 現在の語彙サイズ: {current_vocab}")
        
        # 必要語彙サイズ計算
        required_vocab = current_vocab + 1 + 256  # SEG + IMAGE tokens
        print(f"✅ 必要語彙サイズ: {required_vocab}")
        
        # ⭐ DTensor問題が発生していた部分をテスト
        print("🎯 重要: resize_token_embeddings実行...")
        model.gemma_model.resize_token_embeddings(required_vocab, mean_resizing=False)
        print("✅ resize_token_embeddings成功！DTensor問題解決確認")
        
        # 最終確認
        final_vocab = model.gemma_model.get_input_embeddings().num_embeddings
        print(f"✅ 最終語彙サイズ: {final_vocab}")
        
        print_banner("🎉 LISA-Gemma DTensor問題完全解決！")
        
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: メモリクリーンアップ
    print_banner("Step 3: メモリクリーンアップ")
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    print("✅ メモリクリーンアップ完了")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 すべてのテスト成功！DTensor問題は完全に解決されました。")
        print("🚀 A100*8環境での本格的分散学習の準備が整いました。")
    else:
        print("\n❌ テスト失敗。さらなる調査が必要です。") 