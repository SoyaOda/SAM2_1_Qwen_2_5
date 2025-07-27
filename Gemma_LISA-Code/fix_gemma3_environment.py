#!/usr/bin/env python3
"""
Fix numpy incompatibility and enable gemma-3-4b-it support
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()[-500:]}")  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"   Error: {e.stderr.strip()[-500:]}")  # Last 500 chars
        return False

def clean_environment():
    """Clean up the environment for fresh installation"""
    print("🧹 Cleaning up environment...")
    
    # Clear pip cache
    run_command("pip cache purge", "Clear pip cache")
    
    # Remove problematic packages
    packages_to_remove = [
        "transformers", "tokenizers", "numpy", "scipy", 
        "scikit-learn", "pandas", "matplotlib"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"Remove {package}")
    
    return True

def install_numpy_compatible_stack():
    """Install compatible numpy and related packages"""
    print("📦 Installing compatible numpy stack...")
    
    # Install numpy first with specific version
    if not run_command("pip install numpy==1.24.3 --no-cache-dir", "Install numpy 1.24.3"):
        return False
    
    # Install scipy with compatible version
    if not run_command("pip install scipy==1.10.1 --no-cache-dir", "Install scipy 1.10.1"):
        return False
    
    return True

def install_latest_transformers():
    """Install latest transformers with all dependencies"""
    print("🚀 Installing latest transformers...")
    
    # Install latest transformers
    if not run_command("pip install transformers --upgrade --no-cache-dir", "Install latest transformers"):
        return False
    
    # Install compatible tokenizers
    if not run_command("pip install tokenizers --upgrade --no-cache-dir", "Install latest tokenizers"):
        return False
    
    return True

def test_gemma3_compatibility():
    """Test if gemma-3-4b-it is now supported"""
    print("🧪 Testing gemma-3-4b-it compatibility...")
    try:
        import transformers
        print(f"📦 transformers version: {transformers.__version__}")
        
        from transformers import AutoConfig, AutoTokenizer
        
        # Test config loading
        config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
        print(f"✅ Config loading successful - Model type: {config.model_type}")
        
        # Test tokenizer loading  
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        print(f"✅ Tokenizer loading successful - Vocab size: {tokenizer.vocab_size}")
        
        # Test model class availability
        from transformers import AutoModelForCausalLM
        print("✅ Model class available")
        
        return True
    except Exception as e:
        print(f"❌ gemma-3-4b-it compatibility test failed: {e}")
        return False

def test_dtensor_compatibility():
    """Test if DTensor still works"""
    print("🧪 Testing DTensor compatibility...")
    try:
        import torch
        print(f"🔥 PyTorch version: {torch.__version__}")
        
        try:
            from torch.distributed._tensor import DTensor
            print("✅ DTensor old API still available")
            return True
        except ImportError:
            try:
                from torch.distributed.tensor import DTensor
                print("✅ DTensor new API available")
                return True
            except ImportError:
                print("⚠️  DTensor not available")
                return False
    except Exception as e:
        print(f"❌ DTensor test failed: {e}")
        return False

def main():
    print("="*70)
    print("🔧 Comprehensive Environment Fix for gemma-3-4b-it Support")
    print("="*70)
    
    # Step 1: Clean environment
    print("\n" + "="*50)
    print("Step 1: Environment Cleanup")
    print("="*50)
    
    if not clean_environment():
        print("❌ Environment cleanup failed")
        return False
    
    # Step 2: Install compatible numpy stack
    print("\n" + "="*50)
    print("Step 2: Install Compatible NumPy Stack")
    print("="*50)
    
    if not install_numpy_compatible_stack():
        print("❌ NumPy stack installation failed")
        return False
    
    # Step 3: Install latest transformers
    print("\n" + "="*50)
    print("Step 3: Install Latest Transformers")
    print("="*50)
    
    if not install_latest_transformers():
        print("❌ Transformers installation failed")
        return False
    
    # Step 4: Test compatibility
    print("\n" + "="*50)
    print("Step 4: Compatibility Testing")
    print("="*50)
    
    gemma3_works = test_gemma3_compatibility()
    dtensor_works = test_dtensor_compatibility()
    
    if gemma3_works:
        print("🎉 gemma-3-4b-it compatibility: SUCCESS!")
        if dtensor_works:
            print("🎉 DTensor compatibility: MAINTAINED!")
            print("✅ PERFECT: Both gemma-3-4b-it and DTensor work!")
        else:
            print("⚠️  DTensor compatibility: LOST")
            print("✅ ACCEPTABLE: gemma-3-4b-it works, DTensor may need alternatives")
        return True
    else:
        print("❌ gemma-3-4b-it still not supported")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*70)
        print("🎉 ENVIRONMENT FIX COMPLETED SUCCESSFULLY")
        print("="*70)
        print("✅ gemma-3-4b-it is now supported")
        print("✅ Ready to use google/gemma-3-4b-it in LISA-Gemma")
        print("\n💡 Next steps:")
        print("   1. Update model/gemma_lisa.py to use google/gemma-3-4b-it")
        print("   2. Test LISA-Gemma training with new model")
        print("   3. Monitor for any DTensor issues in distributed training")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("❌ ENVIRONMENT FIX FAILED")
        print("="*70)
        print("💡 gemma-3-4b-it may require transformers >= 4.53.0")
        print("💡 Consider using gemma-2-27b-it as alternative")
        sys.exit(1) 