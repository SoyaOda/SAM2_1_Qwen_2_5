#!/usr/bin/env python3
"""
LISA-Llama4 Training Script
Training script for LISA with Llama-4-Scout-17B-16E-Instruct + SAM
"""

import os
import sys
import argparse
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoProcessor,
    TrainingArguments, 
    Trainer,
    get_scheduler
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configurations
try:
    import config_llama4 as config
    print("✅ Using config_llama4.py")
except ImportError:
    try:
        import config_linux as config
        print("⚠️ Falling back to config_linux.py")
    except ImportError:
        print("❌ No valid configuration found")
        sys.exit(1)

from model.llama4_lisa import (
    LISALlama4ForCausalLM,
    Llama4LisaConfig,
    create_llama4_lisa_model
)
from utils.llama4_processing import Llama4DualStreamProcessor
from utils.llama4_dataset import Llama4HybridDataset, llama4_collate_fn
import deepspeed


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LISA-Llama4 Training")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default=config.MODEL_ID,
                       help="Llama-4 model identifier")
    parser.add_argument("--sam_checkpoint", type=str, default=config.SAM_CHECKPOINT_PATH,
                       help="SAM checkpoint path")
    parser.add_argument("--train_mask_decoder", action="store_true",
                       help="Whether to train SAM mask decoder")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/llama4_lisa",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY,
                       help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=config.WARMUP_RATIO,
                       help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=config.SAVE_STEPS,
                       help="Save checkpoint every N steps")
    
    # Dataset arguments
    parser.add_argument("--dataset_dir", type=str, default=config.DATASET_BASE_DIR,
                       help="Dataset directory")
    parser.add_argument("--samples_per_epoch", type=int, default=config.SAMPLES_PER_EPOCH,
                       help="Samples per epoch")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS,
                       help="Number of data loading workers")
    
    # Loss weights
    parser.add_argument("--ce_loss_weight", type=float, default=config.CE_LOSS_WEIGHT,
                       help="Cross-entropy loss weight")
    parser.add_argument("--dice_loss_weight", type=float, default=config.DICE_LOSS_WEIGHT,
                       help="Dice loss weight")
    parser.add_argument("--bce_loss_weight", type=float, default=config.BCE_LOSS_WEIGHT,
                       help="BCE loss weight")
    
    # Hardware arguments
    parser.add_argument("--precision", type=str, default=config.PRECISION,
                       choices=["fp16", "bf16", "fp32"],
                       help="Training precision")
    parser.add_argument("--use_deepspeed", action="store_true",
                       help="Use DeepSpeed for training")
    parser.add_argument("--deepspeed_config", type=str, default="./configs/deepspeed_config.json",
                       help="DeepSpeed configuration file")
    
    # Logging and evaluation
    parser.add_argument("--logging_steps", type=int, default=config.LOGGING_STEPS,
                       help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=config.EVAL_STEPS,
                       help="Evaluate every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    return args


def initialize_model_and_processor(args):
    """Initialize model and processor"""
    print("\n" + "=" * 70)
    print("🔧 Initializing Model and Processor")
    print("=" * 70)
    
    # Initialize processor
    print(f"📱 Loading processor: {args.model_id}")
    processor = Llama4DualStreamProcessor(model_id=args.model_id)
    
    # Create model configuration
    model_config = Llama4LisaConfig(
        model_id=args.model_id,
        hidden_size=config.HIDDEN_SIZE,
        out_dim=config.OUT_DIM,
        train_mask_decoder=args.train_mask_decoder,
        sam_checkpoint_path=args.sam_checkpoint,
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
        max_images=config.MAX_IMAGES_PER_INPUT,
        use_native_vision=True,
    )
    
    # Initialize model
    print(f"🤖 Loading model: {args.model_id}")
    model = create_llama4_lisa_model(
        model_id=args.model_id,
        sam_checkpoint_path=args.sam_checkpoint,
        train_mask_decoder=args.train_mask_decoder,
        hidden_size=config.HIDDEN_SIZE,
        out_dim=config.OUT_DIM,
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
    )
    
    # Set SEG token ID
    model.set_seg_token_idx(processor.seg_token_id)
    
    # Resize token embeddings if SEG token was added
    model.resize_token_embeddings(len(processor.llama4_processor.tokenizer))
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")
    print(f"   SEG token ID: {processor.seg_token_id}")
    
    return model, processor, model_config


def create_datasets(args, processor):
    """Create training and validation datasets"""
    print("\n" + "=" * 70)
    print("📚 Creating Datasets")
    print("=" * 70)
    
    # Training dataset
    print("🏋️ Creating training dataset...")
    train_dataset = Llama4HybridDataset(
        base_image_dir=args.dataset_dir,
        model_id=args.model_id,
        samples_per_epoch=args.samples_per_epoch,
        precision=args.precision,
        image_size=1024,
        dataset_dir=args.dataset_dir,
        sample_rate=config.SAMPLE_RATE,
    )
    
    print(f"   Training samples per epoch: {len(train_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=llama4_collate_fn,
        pin_memory=True,
    )
    
    print(f"   Training batches per epoch: {len(train_loader)}")
    
    return train_dataset, train_loader


def create_optimizer_and_scheduler(model, args, num_training_steps):
    """Create optimizer and learning rate scheduler"""
    print("\n" + "=" * 70)
    print("⚙️ Creating Optimizer and Scheduler")
    print("=" * 70)
    
    # Filter trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Create learning rate scheduler
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    print(f"📈 Optimizer and Scheduler:")
    print(f"   Optimizer: AdamW")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Total training steps: {num_training_steps}")
    print(f"   Warmup steps: {num_warmup_steps}")
    print(f"   Scheduler: cosine")
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, scheduler, epoch, args, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_ce_loss = 0
    total_seg_loss = 0
    total_dice_loss = 0
    total_bce_loss = 0
    
    num_batches = len(train_loader)
    
    print(f"\n🏋️ Training Epoch {epoch + 1}/{args.num_epochs}")
    print("=" * 70)
    
    for step, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            sam_images=batch['sam_images'],
            seg_token_positions=batch['seg_token_positions'],
            ground_truth_masks=batch['ground_truth_masks'],
            has_masks=batch['has_masks'],
        )
        
        # Calculate loss
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update metrics
        total_loss += loss.item() * args.gradient_accumulation_steps
        total_ce_loss += outputs.ce_loss.item() if hasattr(outputs, 'ce_loss') else 0
        total_seg_loss += outputs.seg_loss.item() if hasattr(outputs, 'seg_loss') else 0
        total_dice_loss += outputs.dice_loss.item() if hasattr(outputs, 'dice_loss') else 0
        total_bce_loss += outputs.bce_loss.item() if hasattr(outputs, 'bce_loss') else 0
        
        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if (step + 1) % args.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"   Step {step + 1}/{num_batches} | "
                      f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f} | "
                      f"LR: {current_lr:.2e}")
        
        # Save checkpoint
        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, step, args)
    
    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_seg_loss = total_seg_loss / num_batches
    avg_dice_loss = total_dice_loss / num_batches
    avg_bce_loss = total_bce_loss / num_batches
    
    print(f"\n📊 Epoch {epoch + 1} Results:")
    print(f"   Average Total Loss: {avg_loss:.4f}")
    print(f"   Average CE Loss: {avg_ce_loss:.4f}")
    print(f"   Average Seg Loss: {avg_seg_loss:.4f}")
    print(f"   Average Dice Loss: {avg_dice_loss:.4f}")
    print(f"   Average BCE Loss: {avg_bce_loss:.4f}")
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, step, args):
    """Save training checkpoint"""
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}-step-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(checkpoint_dir)
    
    # Save training state
    checkpoint_state = {
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args,
    }
    
    torch.save(checkpoint_state, os.path.join(checkpoint_dir, "training_state.pt"))
    
    print(f"💾 Checkpoint saved: {checkpoint_dir}")


def main():
    """Main training function"""
    print("🚀 LISA-Llama4 Training")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model and processor
    model, processor, model_config = initialize_model_and_processor(args)
    model = model.to(device)
    
    # Create datasets
    train_dataset, train_loader = create_datasets(args, processor)
    
    # Calculate training steps
    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, num_training_steps)
    
    # Training loop
    print("\n" + "=" * 70)
    print("🏋️ Starting Training")
    print("=" * 70)
    
    for epoch in range(args.num_epochs):
        # Train epoch
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, args, device)
        
        # Save epoch checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, len(train_loader) - 1, args)
        
        print(f"✅ Epoch {epoch + 1} completed with average loss: {avg_loss:.4f}")
    
    # Final save
    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    processor.llama4_processor.save_pretrained(final_dir)
    
    print(f"\n🎉 Training completed!")
    print(f"   Final model saved: {final_dir}")
    print(f"   Total epochs: {args.num_epochs}")
    print(f"   Total steps: {num_training_steps}")


if __name__ == "__main__":
    main()
