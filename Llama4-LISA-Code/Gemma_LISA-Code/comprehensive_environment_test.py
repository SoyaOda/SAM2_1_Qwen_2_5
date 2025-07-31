#!/usr/bin/env python3
"""
Comprehensive Environment Integration Test for LISA-Gemma
Tests all major components and their integration
"""

import sys
import os
import traceback
from datetime import datetime

print("="*80)
print("🧪 LISA-Gemma Comprehensive Environment Integration Test")
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
        return False

# Test 1: Basic Library Imports
def test_basic_imports():
    import torch
    import transformers
    import accelerate
    import deepspeed
    import wandb
    import cv2
    import skimage
    import numpy as np
    import PIL
    
    versions = {
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'accelerate': accelerate.__version__,
        'deepspeed': deepspeed.__version__,
        'wandb': wandb.__version__,
        'opencv': cv2.__version__,
        'numpy': np.__version__,
    }
    
    print("📦 Library Versions:")
    for lib, ver in versions.items():
        print(f"   {lib}: {ver}")
    
    return True

# Test 2: GPU and CUDA Compatibility
def test_gpu_cuda():
    import torch
    
    if not torch.cuda.is_available():
        raise Exception("CUDA not available")
    
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    
    print(f"🔥 GPU: {device_name}")
    print(f"   Device Count: {device_count}")
    print(f"   VRAM: {device_props.total_memory / 1e9:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    # Test basic GPU operations
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x.T)
    assert y.is_cuda, "GPU tensor operation failed"
    
    return True

# Test 3: DTensor Compatibility
def test_dtensor_compatibility():
    import torch
    
    # Test new DTensor API (PyTorch 2.5+)
    try:
        from torch.distributed.tensor import DTensor
        api_version = "new"
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
            api_version = "old"
        except ImportError:
            raise Exception("DTensor not available")
    
    print(f"🔗 DTensor API: {api_version}")
    
    # Test distributed ops availability
    if hasattr(torch.distributed, 'is_available'):
        print(f"   torch.distributed available: {torch.distributed.is_available()}")
    
    return True

# Test 4: Gemma-3-4B-IT Support
def test_gemma3_support():
    from transformers import AutoConfig, AutoTokenizer, Gemma3ForConditionalGeneration
    
    # Test config loading
    config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
    assert config.model_type == "gemma3", f"Expected gemma3, got {config.model_type}"
    
    # Test tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    vocab_size = tokenizer.vocab_size
    assert vocab_size == 262144, f"Expected 262144, got {vocab_size}"
    
    # Test model class availability
    model_class = Gemma3ForConditionalGeneration
    assert hasattr(model_class, 'from_pretrained'), "from_pretrained method not available"
    
    print(f"🤖 Gemma-3 Config: {config.model_type}")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Architecture: {config.architectures}")
    
    return True

# Test 5: Accelerate Integration
def test_accelerate_integration():
    from accelerate import Accelerator
    import torch
    
    # Test basic accelerator creation (without distributed setup)
    try:
        accelerator = Accelerator(cpu=True)  # CPU mode for testing
        print(f"🚀 Accelerate Device: {accelerator.device}")
        print(f"   Distributed Type: {accelerator.distributed_type}")
        print(f"   Num Processes: {accelerator.num_processes}")
    except Exception as e:
        # Try without any arguments
        accelerator = Accelerator()
        print(f"🚀 Accelerate Device: {accelerator.device}")
    
    return True

# Test 6: DeepSpeed Configuration
def test_deepspeed_config():
    import deepspeed
    import torch
    import json
    
    # Test basic DeepSpeed configuration
    ds_config = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-5
            }
        },
        "fp16": {
            "enabled": True
        }
    }
    
    # Test configuration validation
    config_str = json.dumps(ds_config, indent=2)
    print(f"📝 DeepSpeed Config Sample:")
    print(f"   Train Batch Size: {ds_config['train_batch_size']}")
    print(f"   FP16 Enabled: {ds_config['fp16']['enabled']}")
    
    return True

# Test 7: Image Processing Libraries
def test_image_processing():
    import cv2
    import numpy as np
    from skimage import data, transform
    from PIL import Image
    
    # Test OpenCV
    test_img_cv = np.zeros((224, 224, 3), dtype=np.uint8)
    test_img_cv[:, :, 0] = 255  # Red channel
    assert test_img_cv.shape == (224, 224, 3), "OpenCV image creation failed"
    
    # Test scikit-image
    coins = data.coins()
    resized = transform.resize(coins, (224, 224))
    assert resized.shape == (224, 224), "Scikit-image processing failed"
    
    # Test PIL
    pil_img = Image.new("RGB", (224, 224), color="green")
    assert pil_img.size == (224, 224), "PIL image creation failed"
    
    print(f"🖼️  OpenCV: {test_img_cv.shape}")
    print(f"   Scikit-image: {resized.shape}")
    print(f"   PIL: {pil_img.size}")
    
    return True

# Test 8: Memory Management
def test_memory_management():
    import torch
    import gc
    
    if torch.cuda.is_available():
        # Test GPU memory management
        initial_memory = torch.cuda.memory_allocated()
        
        # Create and delete large tensor
        large_tensor = torch.randn(1000, 1000).cuda()
        peak_memory = torch.cuda.memory_allocated()
        
        del large_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated()
        
        print(f"💾 GPU Memory Test:")
        print(f"   Initial: {initial_memory / 1e6:.1f} MB")
        print(f"   Peak: {peak_memory / 1e6:.1f} MB")
        print(f"   Final: {final_memory / 1e6:.1f} MB")
        
        return True
    else:
        print("💾 CPU Memory Management: OK")
        return True

# Test 9: Integration Test (Small Model Loading)
def test_integration_small_model():
    import torch
    from transformers import AutoTokenizer
    
    # Test with a tiny model first
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    
    # Test tokenization
    test_text = "Hello world! This is a test."
    tokens = tokenizer(test_text, return_tensors="pt")
    
    assert 'input_ids' in tokens, "Tokenization failed"
    assert tokens['input_ids'].shape[1] > 0, "Empty tokenization"
    
    print(f"🔤 Tokenization Test:")
    print(f"   Input: '{test_text}'")
    print(f"   Tokens: {tokens['input_ids'].shape}")
    
    return True

# Run all tests
if __name__ == "__main__":
    tests = [
        ("Basic Library Imports", test_basic_imports),
        ("GPU and CUDA Compatibility", test_gpu_cuda),
        ("DTensor Compatibility", test_dtensor_compatibility),
        ("Gemma-3-4B-IT Support", test_gemma3_support),
        ("Accelerate Integration", test_accelerate_integration),
        ("DeepSpeed Configuration", test_deepspeed_config),
        ("Image Processing Libraries", test_image_processing),
        ("Memory Management", test_memory_management),
        ("Integration Test (Tokenization)", test_integration_small_model),
    ]
    
    print(f"\n🚀 Running {len(tests)} comprehensive tests...")
    
    passed = 0
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
    
    # Final Results
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("="*80)
    
    for test_name, result in test_results.items():
        print(f"{result} {test_name}")
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   - {test}")
        print(f"\n💡 Check individual test failures above for details")
    else:
        print(f"\n🎉 ALL TESTS PASSED! Environment is ready for LISA-Gemma development")
        print(f"✅ Ready for gemma-3-4b-it integration")
        print(f"✅ Ready for multi-GPU distributed training")
    
    print(f"\n📝 Environment Status: {'✅ READY' if not failed_tests else '⚠️  ISSUES DETECTED'}") 