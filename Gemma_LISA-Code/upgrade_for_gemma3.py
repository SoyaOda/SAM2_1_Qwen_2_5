#!/usr/bin/env python3
"""
Upgrade transformers to support gemma-3-4b-it while monitoring DTensor compatibility
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
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"   Error: {e.stderr.strip()}")
        return False

def test_gemma3_compatibility():
    """Test if gemma-3-4b-it is now supported"""
    print("🧪 Testing gemma-3-4b-it compatibility...")
    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
        
        print(f"📦 transformers version: {transformers.__version__}")
        
        # Test config loading
        config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
        print(f"✅ Config loading successful - Model type: {config.model_type}")
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        print(f"✅ Tokenizer loading successful - Vocab size: {tokenizer.vocab_size}")
        
        return True
    except Exception as e:
        print(f"❌ gemma-3-4b-it compatibility test failed: {e}")
        return False

def test_dtensor_compatibility():
    """Test if DTensor still works after upgrade"""
    print("🧪 Testing DTensor compatibility...")
    try:
        import torch
        from torch.distributed._tensor import DTensor
        print(f"✅ DTensor old API still available in PyTorch {torch.__version__}")
        return True
    except ImportError as e:
        print(f"❌ DTensor old API not available: {e}")
        return False

def main():
    print("="*60)
    print("🚀 Upgrading transformers for gemma-3-4b-it support")
    print("="*60)
    
    # Check current versions
    try:
        import transformers
        import torch
        print(f"📦 Current transformers: {transformers.__version__}")
        print(f"🔥 Current PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ Could not import current versions")
        return False
    
    # Step 1: Test current gemma-3 support
    print("\n" + "="*50)
    print("Step 1: Testing current gemma-3-4b-it support")
    print("="*50)
    
    if test_gemma3_compatibility():
        print("🎉 gemma-3-4b-it already supported! No upgrade needed.")
        return True
    
    # Step 2: Upgrade transformers to support gemma-3
    print("\n" + "="*50)
    print("Step 2: Upgrading transformers for gemma-3 support")
    print("="*50)
    
    # Try incremental upgrades to find minimum working version
    upgrade_targets = [
        "4.50.0",  # Minimum for gemma-3 support
        "4.51.0",  # Stable version
        "4.52.0",  # Latest stable before current issues
    ]
    
    for version in upgrade_targets:
        print(f"\n🔧 Trying transformers {version}...")
        
        if run_command(f"pip install transformers=={version} --no-cache-dir", 
                      f"Install transformers {version}"):
            
            # Test gemma-3 compatibility
            if test_gemma3_compatibility():
                print(f"✅ transformers {version} supports gemma-3-4b-it!")
                
                # Test DTensor compatibility
                if test_dtensor_compatibility():
                    print(f"✅ DTensor compatibility maintained with transformers {version}")
                    print(f"🎉 SUCCESS: transformers {version} is optimal!")
                    return True
                else:
                    print(f"⚠️  DTensor compatibility lost with transformers {version}")
                    print("   Will continue with this version for gemma-3 support")
                    print("   DTensor issues may need alternative solutions")
                    return True
            else:
                print(f"❌ transformers {version} still doesn't support gemma-3-4b-it")
    
    print("❌ Could not find compatible transformers version")
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("🎉 UPGRADE COMPLETED SUCCESSFULLY")
        print("="*60)
        print("✅ gemma-3-4b-it is now supported")
        print("✅ Ready to use google/gemma-3-4b-it model")
        print("\n💡 Next steps:")
        print("   1. Test LISA-Gemma with gemma-3-4b-it")
        print("   2. Monitor DTensor behavior in distributed training")
        print("   3. Use fallback strategies if DTensor issues reoccur")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ UPGRADE FAILED")
        print("="*60)
        print("💡 Manual intervention may be required")
        sys.exit(1) 