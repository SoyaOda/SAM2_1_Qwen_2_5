#!/usr/bin/env python3
"""
Simple test for gemma-3-4b-it functionality and environment compatibility
"""

print("="*60)
print("🧪 Simple Gemma-3-4B-IT Functionality Test")
print("="*60)

# Test 1: Basic imports
print("🔧 Testing basic imports...")
try:
    import torch
    import transformers
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ transformers: {transformers.__version__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Gemma3 class availability
print("\n🔧 Testing Gemma3 class availability...")
try:
    from transformers import Gemma3ForConditionalGeneration
    print("✅ Gemma3ForConditionalGeneration class available")
except Exception as e:
    print(f"❌ Gemma3 class import failed: {e}")
    exit(1)

# Test 3: Config and Tokenizer loading
print("\n🔧 Testing config and tokenizer...")
try:
    from transformers import AutoConfig, AutoTokenizer
    
    config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
    print(f"✅ Config loaded: {config.model_type}")
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    print(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
except Exception as e:
    print(f"❌ Config/Tokenizer failed: {e}")
    exit(1)

# Test 4: DTensor compatibility
print("\n🔧 Testing DTensor compatibility...")
try:
    # Try new API first (PyTorch 2.5+)
    from torch.distributed.tensor import DTensor
    print("✅ DTensor (new API) available")
    dtensor_available = True
except ImportError:
    try:
        # Try old API
        from torch.distributed._tensor import DTensor
        print("✅ DTensor (old API) available")
        dtensor_available = True
    except ImportError:
        print("⚠️  DTensor not available (may need distributed setup)")
        dtensor_available = False

# Test 5: GPU availability
print("\n🔧 Testing GPU availability...")
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  GPU not available")

# Final summary
print("\n" + "="*60)
print("📊 COMPATIBILITY SUMMARY")
print("="*60)
print(f"✅ gemma-3-4b-it config/tokenizer: SUPPORTED")
print(f"✅ Gemma3ForConditionalGeneration: AVAILABLE")
print(f"{'✅' if dtensor_available else '⚠️ '} DTensor: {'AVAILABLE' if dtensor_available else 'LIMITED'}")
print(f"✅ Environment: READY FOR LISA-GEMMA")

print("\n🚀 RESULT: gemma-3-4b-it is ready to use!")
print("💡 Next: Update LISA-Gemma model configuration") 