#!/usr/bin/env python3
"""
Llama-4-Scout-17B-16E-Instructのトークナイザーを詳細に確認
特に<image>トークンがデフォルトで存在するかをチェック
"""

import sys
import os
import json
from transformers import AutoTokenizer, AutoProcessor

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_linux

def check_tokenizer():
    """トークナイザーの詳細確認"""
    print("=== Llama-4-Scout-17B-16E-Instructトークナイザー詳細確認 ===\n")
    
    # トークナイザーをロード
    print(f"モデルID: {config_linux.LLAMA_MODEL_ID}")
    print("トークナイザーをロード中...")
    
    try:
        # AutoProcessorを試す
        processor = AutoProcessor.from_pretrained(
            config_linux.LLAMA_MODEL_ID,
            trust_remote_code=True
        )
        tokenizer = processor.tokenizer
        print("✅ AutoProcessorからトークナイザーを取得")
    except Exception as e:
        print(f"⚠️ AutoProcessorエラー: {e}")
        # 直接トークナイザーをロード
        tokenizer = AutoTokenizer.from_pretrained(
            config_linux.LLAMA_MODEL_ID,
            trust_remote_code=True
        )
        print("✅ AutoTokenizerから直接取得")
    
    # 基本情報
    print(f"\n[基本情報]")
    print(f"トークナイザークラス: {type(tokenizer).__name__}")
    print(f"語彙サイズ: {len(tokenizer)}")
    print(f"モデル最大長: {tokenizer.model_max_length}")
    
    # 特殊トークン
    print(f"\n[特殊トークン]")
    special_tokens = tokenizer.special_tokens_map
    for key, value in special_tokens.items():
        print(f"  {key}: {value}")
    
    # 追加の特殊トークン
    if hasattr(tokenizer, 'additional_special_tokens'):
        print(f"\n[追加特殊トークン]")
        for token in tokenizer.additional_special_tokens:
            print(f"  - {token}")
    
    # <image>トークンの存在確認
    print(f"\n[<image>トークン確認]")
    image_tokens = ["<image>", "<img>", "[IMAGE]", "[IMG]", "<IMAGE>", "<IMG>"]
    vocab = tokenizer.get_vocab()
    
    found_image_tokens = []
    for token in image_tokens:
        if token in vocab:
            token_id = vocab[token]
            found_image_tokens.append((token, token_id))
            print(f"✅ {token} が存在: ID = {token_id}")
        else:
            print(f"❌ {token} は存在しない")
    
    # 全トークンから画像関連のものを検索
    print(f"\n[画像関連トークンの検索]")
    image_related = []
    for token, token_id in vocab.items():
        if any(keyword in token.lower() for keyword in ['image', 'img', 'pic', 'photo', 'vision']):
            image_related.append((token, token_id))
    
    if image_related:
        print(f"画像関連トークンが {len(image_related)} 個見つかりました:")
        for token, token_id in sorted(image_related)[:20]:  # 最初の20個のみ表示
            print(f"  - {token}: ID = {token_id}")
        if len(image_related) > 20:
            print(f"  ... 他 {len(image_related) - 20} 個")
    else:
        print("画像関連トークンは見つかりませんでした")
    
    # テスト: <image>を含むテキストのトークン化
    print(f"\n[<image>を含むテキストのトークン化テスト]")
    test_texts = [
        "<image>",
        "USER: <image>\nWhat is in this image?",
        "This is an <image> token test.",
    ]
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"\nテキスト: {repr(text)}")
        print(f"トークン: {tokens}")
        print(f"トークンID: {token_ids}")
        
        # <image>がどのようにトークン化されたか確認
        if "<image>" in text:
            decoded = tokenizer.decode(token_ids)
            print(f"デコード結果: {repr(decoded)}")
            if decoded != text:
                print("⚠️ 元のテキストと異なります！")
    
    # 設定ファイルの確認
    print(f"\n[トークナイザー設定の詳細]")
    if hasattr(tokenizer, 'name_or_path'):
        print(f"名前/パス: {tokenizer.name_or_path}")
    if hasattr(tokenizer, 'chat_template'):
        print(f"チャットテンプレート: {'あり' if tokenizer.chat_template else 'なし'}")
    
    return found_image_tokens

if __name__ == "__main__":
    found_tokens = check_tokenizer()
    
    print("\n" + "="*50)
    print("結論:")
    if found_tokens:
        print(f"✅ デフォルトで画像トークンが存在します: {found_tokens}")
    else:
        print("❌ デフォルトでは<image>トークンは存在しません")
        print("   → カスタムで追加する必要があります")

