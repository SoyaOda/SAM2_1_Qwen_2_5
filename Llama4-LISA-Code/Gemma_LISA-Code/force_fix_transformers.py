#!/usr/bin/env python3
"""
Force fix broken transformers installation and install latest version for gemma-3-4b-it
"""

import subprocess
import sys
import os

def run_command(cmd, description, allow_fail=False):
    """Run a command and return success status"""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            print(f"   Output: {lines[-1]}")  # Last line only
        return True
    except subprocess.CalledProcessError as e:
        if allow_fail:
            print(f"⚠️  Expected failure: {description}")
            return True
        print(f"❌ Failed: {description}")
        if e.stderr.strip():
            lines = e.stderr.strip().split('\n')
            print(f"   Error: {lines[-1]}")  # Last line only
        return False

def force_fix_transformers():
    """Force fix broken transformers installation"""
    print("🔨 Force fixing broken transformers...")
    
    # Step 1: Force reinstall current broken version to fix RECORD
    print("\n🔧 Step 1: Force reinstall to fix RECORD file...")
    if not run_command("pip install --force-reinstall --no-deps transformers==4.52.0", 
                      "Force reinstall transformers 4.52.0"):
        return False
    
    # Step 2: Reinstall dependencies  
    print("\n🔧 Step 2: Reinstall dependencies...")
    dependencies = [
        "huggingface-hub", "tokenizers", "numpy", "pyyaml", 
        "regex", "requests", "safetensors", "tqdm"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep} --upgrade --no-cache-dir", f"Install {dep}", allow_fail=True)
    
    # Step 3: Upgrade to latest transformers
    print("\n🔧 Step 3: Upgrade to latest transformers...")
    if not run_command("pip install transformers --upgrade --no-cache-dir", 
                      "Upgrade to latest transformers"):
        return False
    
    return True

def test_basic_import():
    """Test if transformers can be imported"""
    print("🧪 Testing basic transformers import...")
    try:
        import transformers
        print(f"✅ transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"❌ transformers import failed: {e}")
        return False

def test_gemma3_compatibility():
    """Test if gemma-3-4b-it is supported"""
    print("🧪 Testing gemma-3-4b-it compatibility...")
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Test config loading
        print("   Testing config loading...")
        config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
        print(f"✅ Config: model_type={config.model_type}")
        
        # Test tokenizer loading
        print("   Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        print(f"✅ Tokenizer: vocab_size={tokenizer.vocab_size}")
        
        return True
    except Exception as e:
        print(f"❌ gemma-3-4b-it test failed: {e}")
        return False

def test_dtensor_compatibility():
    """Test DTensor compatibility"""
    print("🧪 Testing DTensor compatibility...")
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        
        # Try old API first
        try:
            from torch.distributed._tensor import DTensor
            print("✅ DTensor (old API) available")
            return True
        except ImportError:
            pass
        
        # Try new API
        try:
            from torch.distributed.tensor import DTensor
            print("✅ DTensor (new API) available")
            return True
        except ImportError:
            print("⚠️  DTensor not available (may need distributed setup)")
            return False
            
    except Exception as e:
        print(f"❌ DTensor test failed: {e}")
        return False

def main():
    print("="*70)
    print("🔨 Force Fix Transformers for gemma-3-4b-it Support")
    print("="*70)
    
    # Step 1: Force fix broken transformers
    print("\n" + "="*50)
    print("Step 1: Force Fix Broken Transformers")
    print("="*50)
    
    if not force_fix_transformers():
        print("❌ Failed to fix transformers installation")
        return False
    
    # Step 2: Test basic functionality
    print("\n" + "="*50)
    print("Step 2: Test Basic Functionality")
    print("="*50)
    
    if not test_basic_import():
        print("❌ Basic transformers import still broken")
        return False
    
    # Step 3: Test gemma-3 compatibility
    print("\n" + "="*50)
    print("Step 3: Test gemma-3-4b-it Compatibility")
    print("="*50)
    
    gemma3_works = test_gemma3_compatibility()
    
    # Step 4: Test DTensor compatibility
    print("\n" + "="*50)
    print("Step 4: Test DTensor Compatibility")
    print("="*50)
    
    dtensor_works = test_dtensor_compatibility()
    
    # Final result
    if gemma3_works:
        print("\n🎉 SUCCESS: gemma-3-4b-it is now supported!")
        if dtensor_works:
            print("🎉 BONUS: DTensor compatibility maintained!")
        else:
            print("⚠️  Note: DTensor may need distributed environment")
        return True
    else:
        print("\n❌ gemma-3-4b-it still not supported")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*70)
        print("🎉 TRANSFORMERS FIX COMPLETED SUCCESSFULLY")
        print("="*70)
        print("✅ gemma-3-4b-it is now supported")
        print("✅ Ready to use google/gemma-3-4b-it in LISA-Gemma")
        print("\n💡 Next action:")
        print("   Update LISA-Gemma to use google/gemma-3-4b-it")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("❌ TRANSFORMERS FIX FAILED")
        print("="*70)
        print("💡 gemma-3-4b-it may require transformers from main branch")
        print("💡 Alternative: Use gemma-2-27b-it which is well supported")
        sys.exit(1) 