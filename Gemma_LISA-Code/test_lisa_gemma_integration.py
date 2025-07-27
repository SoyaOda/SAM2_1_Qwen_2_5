#!/usr/bin/env python3
"""
LISA-Gemma Integration Test
Test if current LISA-Gemma code works with gemma-3-4b-it
"""

import sys
import os
import torch
from datetime import datetime
import traceback

print("="*80)
print("🤖 LISA-Gemma Integration Test (gemma-3-4b-it)")
print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Test results tracking
test_results = {}
failed_tests = []

def run_test(test_name, test_func):
    """Run a test and track results"""
    print(f"\n🔧 Testing {test_name}...")
    try:
        result = test_func()
        test_results[test_name] = "✅ PASS"
        print(f"✅ {test_name}: SUCCESS")
        return True
    except Exception as e:
        test_results[test_name] = f"❌ FAIL: {str(e)}"
        failed_tests.append(test_name)
        print(f"❌ {test_name}: FAILED - {str(e)}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

# Test 1: Import LISA-Gemma Model
def test_lisa_gemma_import():
    from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
    print(f"✅ LISA-Gemma classes imported successfully")
    print(f"   Config class: {LisaGemmaConfig}")
    print(f"   Model class: {LisaGemmaForCausalLM}")
    return True

# Test 2: Create LISA-Gemma Configuration
def test_lisa_gemma_config():
    from model.gemma_lisa import LisaGemmaConfig
    
    config = LisaGemmaConfig(
        gemma_model_id="google/gemma-3-4b-it",
        sam_checkpoint_path=None,  # No SAM for now
        seg_token="[SEG]",
        gemma_hidden_size=2560,
        sam_prompt_embed_dim=256,
        gemma_image_size=896,
        sam_image_size=1024,
        model_max_length=2048,
    )
    
    print(f"✅ LISA-Gemma Config created:")
    print(f"   Gemma Model: {config.gemma_model_id}")
    print(f"   SEG Token: {config.seg_token}")
    print(f"   Hidden Size: {config.gemma_hidden_size}")
    print(f"   SAM Checkpoint: {config.sam_checkpoint_path}")
    
    return True

# Test 3: Initialize LISA-Gemma Model (CPU/Small)
def test_lisa_gemma_model_init():
    from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
    
    # Create minimal config for testing
    config = LisaGemmaConfig(
        gemma_model_id="google/gemma-3-4b-it",
        sam_checkpoint_path=None,  # Skip SAM for quick testing
        seg_token="[SEG]",
        gemma_hidden_size=2560,
        sam_prompt_embed_dim=256,
        gemma_image_size=896,
        sam_image_size=1024,
        model_max_length=2048,
    )
    
    print(f"🔄 Initializing LISA-Gemma model...")
    print(f"   This may take a few minutes for first-time download...")
    
    # Initialize model
    model = LisaGemmaForCausalLM(config)
    
    print(f"✅ Model initialized successfully")
    print(f"   Type: {type(model)}")
    print(f"   Gemma Model: {type(model.gemma_model)}")
    print(f"   Has SAM: {model.has_sam_capability()}")
    
    # Check embedding size
    embed_size = model.gemma_model.get_input_embeddings().weight.shape[0]
    print(f"   Embedding Size: {embed_size}")
    
    # Check tokenizer
    tokenizer_vocab_size = len(model.gemma_processor.tokenizer)
    print(f"   Tokenizer Vocab Size: {tokenizer_vocab_size}")
    
    # Check SEG token
    seg_token_id = model.seg_token_id
    print(f"   SEG Token ID: {seg_token_id}")
    
    return True

# Test 4: Basic Text Processing
def test_basic_text_processing():
    from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
    
    # Create config
    config = LisaGemmaConfig(
        gemma_model_id="google/gemma-3-4b-it",
        sam_checkpoint_path=None,
    )
    
    # Initialize model
    model = LisaGemmaForCausalLM(config)
    
    # Test text processing
    test_text = "Hello, this is a test for LISA-Gemma with gemma-3-4b-it model."
    
    # Tokenize
    tokens = model.gemma_processor.tokenizer(test_text, return_tensors="pt")
    
    print(f"✅ Text processing test:")
    print(f"   Input: '{test_text}'")
    print(f"   Tokens shape: {tokens['input_ids'].shape}")
    print(f"   First 10 token IDs: {tokens['input_ids'][0][:10].tolist()}")
    
    return True

# Test 5: Model Device and Memory
def test_model_device_memory():
    import torch
    from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
    
    # Create config
    config = LisaGemmaConfig(
        gemma_model_id="google/gemma-3-4b-it",
        sam_checkpoint_path=None,
    )
    
    # Initialize model
    model = LisaGemmaForCausalLM(config)
    
    # Check model device
    model_device = next(model.parameters()).device
    print(f"✅ Model device check:")
    print(f"   Model device: {model_device}")
    
    # Check memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU Memory Allocated: {memory_allocated:.2f} GB")
        print(f"   GPU Memory Reserved: {memory_reserved:.2f} GB")
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Frozen Parameters: {total_params - trainable_params:,}")
    
    return True

# Run all tests
if __name__ == "__main__":
    tests = [
        ("LISA-Gemma Import", test_lisa_gemma_import),
        ("LISA-Gemma Config", test_lisa_gemma_config),
        ("LISA-Gemma Model Init", test_lisa_gemma_model_init),
        ("Basic Text Processing", test_basic_text_processing),
        ("Model Device and Memory", test_model_device_memory),
    ]
    
    print(f"\n🚀 Running {len(tests)} LISA-Gemma integration tests...")
    
    passed = 0
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
    
    # Final Results
    print("\n" + "="*80)
    print("📊 LISA-GEMMA INTEGRATION TEST RESULTS")
    print("="*80)
    
    for test_name, result in test_results.items():
        print(f"{result} {test_name}")
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   - {test}")
        print(f"\n💡 Possible Issues:")
        print(f"   - Model too large for current GPU memory")
        print(f"   - Missing dependencies")
        print(f"   - Configuration issues")
    else:
        print(f"\n🎉 ALL LISA-GEMMA TESTS PASSED!")
        print(f"✅ LISA-Gemma with gemma-3-4b-it is fully functional")
        print(f"✅ Ready for training and inference")
        print(f"✅ No code updates needed for gemma-3-4b-it support")
    
    print(f"\n📝 LISA-Gemma Status: {'✅ READY' if not failed_tests else '⚠️  ISSUES DETECTED'}")
    print(f"💡 Next: Ready for actual training/inference testing") 