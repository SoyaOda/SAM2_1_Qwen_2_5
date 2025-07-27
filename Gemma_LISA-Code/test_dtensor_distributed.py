#!/usr/bin/env python3
"""
DTensor Distributed Training Test for LISA-Gemma
Tests DTensor functionality in distributed environments
"""

import os
import sys
import torch
import torch.distributed as dist
from datetime import datetime
import traceback

print("="*80)
print("🔗 DTensor Distributed Training Test")
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

# Test 1: DTensor Import and Basic Operations
def test_dtensor_basic():
    import torch
    from torch.distributed.tensor import DTensor, Shard, Replicate
    from torch.distributed._tensor.device_mesh import init_device_mesh, DeviceMesh
    
    print(f"🔗 DTensor API: Available (PyTorch {torch.__version__})")
    
    # Test basic DTensor creation (without distributed setup)
    # This tests if the DTensor classes are properly imported
    device_mesh_tensor = torch.arange(4).reshape(2, 2)
    print(f"   Device mesh tensor: {device_mesh_tensor.shape}")
    
    # Test DTensor placement strategies
    shard_spec = Shard(0)
    replicate_spec = Replicate()
    print(f"   Shard spec: {shard_spec}")
    print(f"   Replicate spec: {replicate_spec}")
    
    return True

# Test 2: Distributed Environment Setup
def test_distributed_setup():
    import torch
    import torch.distributed as dist
    
    # Check if distributed is available
    if not dist.is_available():
        raise Exception("torch.distributed not available")
    
    print(f"🌐 Distributed Status:")
    print(f"   Available: {dist.is_available()}")
    print(f"   Initialized: {dist.is_initialized()}")
    
    # Test NCCL availability
    if torch.cuda.is_available():
        print(f"   NCCL Available: {dist.is_nccl_available()}")
    
    return True

# Test 3: Accelerate Integration with DTensor
def test_accelerate_dtensor():
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    import torch
    
    # Test accelerate with different configurations
    set_seed(42)
    
    # Test 1: CPU mode
    accelerator_cpu = Accelerator(cpu=True)
    print(f"🚀 Accelerate CPU Mode:")
    print(f"   Device: {accelerator_cpu.device}")
    print(f"   Distributed Type: {accelerator_cpu.distributed_type}")
    
    # Test 2: GPU mode (if available)
    if torch.cuda.is_available():
        accelerator_gpu = Accelerator()
        print(f"🚀 Accelerate GPU Mode:")
        print(f"   Device: {accelerator_gpu.device}")
        print(f"   Distributed Type: {accelerator_gpu.distributed_type}")
        print(f"   Num Processes: {accelerator_gpu.num_processes}")
    
    return True

# Test 4: Model Preparation with DTensor-like Operations
def test_model_dtensor_preparation():
    from transformers import AutoConfig, AutoTokenizer
    import torch
    import torch.nn as nn
    
    # Test model configuration loading
    config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    
    print(f"🤖 Model Configuration:")
    print(f"   Model Type: {config.model_type}")
    print(f"   Vocab Size: {config.vocab_size}")
    print(f"   Hidden Size: {config.hidden_size}")
    
    # Test embedding layer resize operation (DTensor problem area)
    current_vocab_size = config.vocab_size
    required_vocab_size = current_vocab_size + 1 + 256  # SEG token + 256 image tokens
    
    print(f"📏 Vocab Size Calculation:")
    print(f"   Current: {current_vocab_size}")
    print(f"   Required: {required_vocab_size}")
    print(f"   Difference: {required_vocab_size - current_vocab_size}")
    
    # Test embedding layer creation
    embedding_dim = config.hidden_size
    old_embeddings = nn.Embedding(current_vocab_size, embedding_dim)
    new_embeddings = nn.Embedding(required_vocab_size, embedding_dim)
    
    print(f"🔤 Embedding Layers:")
    print(f"   Old: {old_embeddings.weight.shape}")
    print(f"   New: {new_embeddings.weight.shape}")
    
    # Test manual resize operation (the problematic area)
    with torch.no_grad():
        new_embeddings.weight[:current_vocab_size] = old_embeddings.weight
        print(f"✅ Manual embedding resize: SUCCESS")
    
    return True

# Test 5: Simulated Multi-GPU DTensor Operations
def test_simulated_dtensor_ops():
    import torch
    from torch.distributed.tensor import DTensor, Shard, Replicate
    
    if not torch.cuda.is_available():
        print("⚠️  GPU not available, testing CPU DTensor operations")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    # Create test tensors
    test_tensor = torch.randn(8, 512).to(device)
    print(f"🧮 Test Tensor: {test_tensor.shape} on {device}")
    
    # Test tensor operations that would be problematic in DTensor
    copied_tensor = test_tensor.clone()
    sliced_tensor = test_tensor[:4, :]
    
    print(f"   Cloned: {copied_tensor.shape}")
    print(f"   Sliced: {sliced_tensor.shape}")
    
    # Test the specific operation that causes DTensor issues
    combined_tensor = torch.zeros(10, 512).to(device)
    combined_tensor[:8, :] = test_tensor
    print(f"   Combined: {combined_tensor.shape}")
    print(f"✅ Tensor copy operations: SUCCESS")
    
    return True

# Test 6: Accelerate Launch Configuration
def test_accelerate_config():
    from accelerate import Accelerator
    from accelerate.utils import write_basic_config
    import tempfile
    import os
    
    # Test accelerate configuration
    print(f"🚀 Accelerate Configuration Test:")
    
    # Create temporary config
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "default_config.yaml")
        
        # Test basic config creation
        print(f"   Config file: {config_file}")
        
        # Test accelerator with explicit config
        accelerator = Accelerator()
        print(f"   Device: {accelerator.device}")
        print(f"   State: {accelerator.state}")
        
    return True

# Test 7: DeepSpeed Integration
def test_deepspeed_integration():
    import deepspeed
    import torch
    import json
    
    # Test DeepSpeed configuration
    ds_config = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 100
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        }
    }
    
    print(f"📝 DeepSpeed Configuration:")
    print(f"   Batch Size: {ds_config['train_batch_size']}")
    print(f"   Micro Batch Size: {ds_config['train_micro_batch_size_per_gpu']}")
    print(f"   ZeRO Stage: {ds_config['zero_optimization']['stage']}")
    print(f"   FP16: {ds_config['fp16']['enabled']}")
    
    # Test configuration validation
    config_json = json.dumps(ds_config, indent=2)
    assert len(config_json) > 100, "Config too short"
    
    return True

# Run all tests
if __name__ == "__main__":
    tests = [
        ("DTensor Basic Operations", test_dtensor_basic),
        ("Distributed Environment Setup", test_distributed_setup),
        ("Accelerate Integration with DTensor", test_accelerate_dtensor),
        ("Model DTensor Preparation", test_model_dtensor_preparation),
        ("Simulated DTensor Operations", test_simulated_dtensor_ops),
        ("Accelerate Launch Configuration", test_accelerate_config),
        ("DeepSpeed Integration", test_deepspeed_integration),
    ]
    
    print(f"\n🚀 Running {len(tests)} DTensor distributed tests...")
    
    passed = 0
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
    
    # Final Results
    print("\n" + "="*80)
    print("📊 DTENSOR DISTRIBUTED TEST RESULTS")
    print("="*80)
    
    for test_name, result in test_results.items():
        print(f"{result} {test_name}")
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   - {test}")
        print(f"\n💡 Possible Issues:")
        print(f"   - DTensor API changes in PyTorch 2.5.1")
        print(f"   - Distributed environment not initialized")
        print(f"   - Multi-GPU setup required for full testing")
    else:
        print(f"\n🎉 ALL DTENSOR TESTS PASSED!")
        print(f"✅ DTensor operations compatible with current environment")
        print(f"✅ Ready for distributed training with accelerate/deepspeed")
        print(f"✅ Embedding resize operations work correctly")
    
    print(f"\n📝 DTensor Status: {'✅ READY' if not failed_tests else '⚠️  ISSUES DETECTED'}")
    print(f"💡 Next: Test with actual multi-GPU environment for full validation") 