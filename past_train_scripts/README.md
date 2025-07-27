# Past Training Scripts

This folder contains training scripts that were used during development but are no longer actively maintained or used.

## Current Active Scripts (Root Directory)
- `train_llama4_lisa_single_process.py` - ✅ **Main training script** (verified working)
- `launch_simple_training.py` - Launcher for simple training
- `overfit_llama4_lisa_batch.py` - Verification/testing script

## Archived Scripts (This Directory)

### Legacy Training Scripts
- `train.py` - Original training script
- `train_llama4.py` - Early Llama4-only training implementation
- `train_llama4_deepspeed.py` - Early DeepSpeed implementation (standalone Llama4)

### Development/Debug Scripts  
- `train_debug.py` - Debug version of training script
- `train_llama4_lisa_simple.py` - Simplified LISA training attempt
- `train_llama4_lisa_deepspeed.py` - LISA + DeepSpeed integration attempt

### Legacy Launchers
- `launch_training.py` - Old training launcher

## Note
These scripts were moved here after successful completion of `train_llama4_lisa_single_process.py` implementation to clean up the project structure. They contain valuable development history but are not needed for current training workflows.