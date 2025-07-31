#!/usr/bin/env python3
"""
Llama-4プロセッサ初期化問題デバッグスクリプト
"""
import sys
import torch
import logging
from pathlib import Path

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_llama4_processor():
    """Llama-4プロセッサ初期化をデバッグ"""
    
    logger.info("=== Llama-4プロセッサ初期化デバッグ開始 ===")
    
    # 1. transformersライブラリ確認
    try:
        import transformers
        logger.info(f"✅ transformers version: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"❌ transformers import error: {e}")
        return
    
    # 2. AutoProcessorインポート確認
    try:
        from transformers import AutoProcessor, AutoModel, AutoTokenizer
        logger.info("✅ AutoProcessor, AutoModel, AutoTokenizer import成功")
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return
    
    # 3. モデルID確認
    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    logger.info(f"📋 Model ID: {model_id}")
    
    # 4. プロセッサ初期化試行（詳細エラー情報付き）
    logger.info("🔄 AutoProcessor.from_pretrained()試行中...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True  # Fast tokenizerを明示的に指定
        )
        logger.info("✅ AutoProcessor初期化成功！")
        logger.info(f"  - Processor type: {type(processor)}")
        logger.info(f"  - Attributes: {dir(processor)}")
        return processor
        
    except Exception as e:
        logger.error(f"❌ AutoProcessor初期化失敗: {type(e).__name__}: {e}")
        
        # 5. フォールバック: AutoTokenizer試行
        logger.info("🔄 フォールバック: AutoTokenizer試行中...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True
            )
            logger.info("✅ AutoTokenizer初期化成功！")
            logger.info(f"  - Tokenizer type: {type(tokenizer)}")
            
            # 6. Llama-4はマルチモーダルなのでImageProcessorも必要か確認
            logger.info("🔄 ImageProcessor確認中...")
            try:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                logger.info("✅ AutoImageProcessor初期化成功！")
                return tokenizer, image_processor
            except Exception as img_e:
                logger.warning(f"⚠️ AutoImageProcessor初期化失敗: {img_e}")
                logger.info("💡 Llama-4-Scoutはテキストのみの可能性")
                return tokenizer
                
        except Exception as tok_e:
            logger.error(f"❌ AutoTokenizer初期化も失敗: {tok_e}")
            
    # 7. 環境情報出力
    logger.info("\n=== 環境情報 ===")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # 8. HuggingFaceトークン確認
    import os
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        logger.info("✅ HuggingFaceトークン設定済み")
    else:
        logger.warning("⚠️ HuggingFaceトークン未設定（プライベートモデルの場合必要）")

if __name__ == "__main__":
    result = debug_llama4_processor()
    if result:
        logger.info("\n🎉 デバッグ成功！プロセッサ初期化可能")
    else:
        logger.info("\n❌ デバッグ失敗。上記エラーを確認してください")