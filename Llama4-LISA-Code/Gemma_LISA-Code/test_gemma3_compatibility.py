#!/usr/bin/env python3
"""
Test gemma-3-4b-it compatibility with current environment
"""

import transformers
import torch
import traceback

def test_gemma3_compatibility():
    print("="*60)
    print("🔍 Gemma-3-4B-IT Compatibility Test")
    print("="*60)
    
    # 1. Check current versions
    print(f"🐍 Python: {torch.__version__}")
    print(f"📦 transformers: {transformers.__version__}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print()
    
    # 2. Test tokenizer loading
    print("🧪 Testing Tokenizer Loading...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        print("✅ gemma-3-4b-it tokenizer loading: SUCCESS")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ gemma-3-4b-it tokenizer loading: FAILED")
        print(f"   Error: {str(e)}")
        print()
        print("🔍 Error Details:")
        traceback.print_exc()
        return False
    
    # 3. Test model configuration loading
    print("\n🧪 Testing Model Configuration...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
        print("✅ gemma-3-4b-it config loading: SUCCESS")
        print(f"   Model type: {config.model_type}")
        print(f"   Architecture: {config.architectures}")
    except Exception as e:
        print(f"❌ gemma-3-4b-it config loading: FAILED")
        print(f"   Error: {str(e)}")
        return False
    
    # 4. Test model class availability
    print("\n🧪 Testing Model Class...")
    try:
        from transformers import AutoModelForCausalLM
        # Just test if the model class is available, don't actually load
        model_class = AutoModelForCausalLM._model_mapping.get(config.model_type)
        if model_class:
            print("✅ gemma-3-4b-it model class: AVAILABLE")
            print(f"   Model class: {model_class}")
        else:
            print("❌ gemma-3-4b-it model class: NOT AVAILABLE")
            return False
    except Exception as e:
        print(f"❌ gemma-3-4b-it model class test: FAILED")
        print(f"   Error: {str(e)}")
        return False
    
    print("\n🎉 All compatibility tests PASSED!")
    print("✅ Current environment supports gemma-3-4b-it")
    return True

if __name__ == "__main__":
    success = test_gemma3_compatibility()
    if not success:
        print("\n💡 SOLUTION REQUIRED:")
        print("   Current transformers version does not support gemma-3-4b-it")
        print("   Minimum required: transformers >= 4.50.0")
        print("   Current version: {}".format(transformers.__version__))
        exit(1)
    else:
        print("\n🚀 Ready to use gemma-3-4b-it!")
        exit(0) 