#!/usr/bin/env python3
"""
Optimized training script specifically for MPS (Apple Silicon).
This script implements several optimizations for better performance:
1. Uses float16 precision for model parameters
2. Employs a smaller batch size (1) which often works better on MPS
3. Minimizes device transfers
4. Regularly clears memory cache
5. Uses simpler tensor operations where possible
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import argparse
import gc

from src.models.dead_model import DeadShowDatingModel
from src.models import CombinedDeadLoss
from src.data.dataset import PreprocessedDeadShowDataset, optimized_collate_fn


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Optimized MPS training")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1, best for MPS)")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the dataset to this many samples for faster training")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with extra logging")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set the data type based on arguments
    dtype = torch.float16 if args.fp16 else torch.float32
    print(f"Using precision: {dtype}")
    
    # Create model
    print("Creating model...")
    model = DeadShowDatingModel(sample_rate=24000)
    
    # Convert model to float16 if requested
    if args.fp16:
        model = model.to(dtype=dtype)
    
    # Move model to device
    model = model.to(device)
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.data_dir}...")
    dataset = PreprocessedDeadShowDataset(
        preprocessed_dir=args.data_dir,
        device=None,  # Always load data on CPU first
        memory_efficient=True,  # Use memory-efficient mode
    )
    
    # Print first few files to verify dataset
    print(f"First few files in dataset:")
    for i in range(min(3, len(dataset))):
        print(f"  {dataset.files[i]}")
        
    # Limit dataset size if requested
    if args.max_samples is not None and args.max_samples < len(dataset.files):
        print(f"Limiting dataset to {args.max_samples} samples (from {len(dataset.files)})")
        import random
        random.seed(42)  # For reproducibility
        limited_files = random.sample(dataset.files, args.max_samples)
        dataset.files = limited_files
    
    print(f"Using {len(dataset)} files for training")
    
    # Create dataloader with optimized settings for MPS
    print(f"Creating dataloader with batch size {args.batch_size}...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # For MPS compatibility
        collate_fn=lambda batch: optimized_collate_fn(batch, device=None),
        drop_last=True,
        pin_memory=False  # Don't pin memory for MPS
    )
    
    # Create optimizer with small initial learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,  # Start with smaller learning rate
        weight_decay=0.01,
        eps=1e-5,  # Slightly larger epsilon for better numerical stability
    )
    
    # Set gradient clipping value much lower due to our scaling fix
    grad_clip_value = 1.0
    
    # Create loss function
    criterion = CombinedDeadLoss()
    
    # Set model to training mode
    model.train()
    
    # Initialize progress tracking
    total_steps = args.steps
    start_time = time.time()
    losses = []
    
    # Clear memory before starting
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    # Training loop
    print(f"Starting training for {total_steps} steps...")
    
    for step in range(total_steps):
        step_start = time.time()
        
        # Get a batch
        try:
            batch = next(dataloader_iter)
        except (StopIteration, NameError):
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Skip empty batches
        if batch.get('empty_batch', False):
            print(f"Step {step+1}/{total_steps}: Skipping empty batch")
            continue
        
        # Move data to device and convert to float16 if needed
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if args.fp16:
                    # Use float16 for most tensors
                    if key not in ['era', 'year'] and value.dtype != torch.int64 and value.dtype != torch.int32:
                        batch[key] = value.to(device=device, dtype=dtype)
                    else:
                        batch[key] = value.to(device)
                else:
                    batch[key] = value.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        try:
            # Debug info if requested
            if args.debug:
                print(f"  Forward pass - batch keys: {list(batch.keys())}")
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        print(f"    {key}: {val.shape}, {val.dtype}, device={val.device}")
            
            outputs = model(batch, global_step=step)
            
            # Debug info if requested
            if args.debug:
                print(f"  Forward pass complete - output keys: {list(outputs.keys())}")
                for key, val in outputs.items():
                    if isinstance(val, torch.Tensor):
                        print(f"    {key}: {val.shape}, {val.dtype}, device={val.device}")
            
            # Loss calculation
            loss_dict = criterion(
                outputs,
                {'days': batch.get('label'), 'era': batch.get('era')},
                step,
                total_steps
            )
        except Exception as e:
            print(f"Error in forward/loss calculation: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
        loss = loss_dict['loss']
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {step+1}/{total_steps}: Invalid loss detected, skipping step")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        # Update weights
        optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        losses.append(loss_value)
        
        # Clear memory every few steps
        if step % 5 == 0:
            # Delete tensors to free memory
            if 'outputs' in locals(): del outputs
            if 'loss_dict' in locals(): del loss_dict
            if 'loss' in locals(): del loss
            
            # Explicitly clear memory
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        # Compute elapsed time and ETA
        step_time = time.time() - step_start
        elapsed = time.time() - start_time
        steps_per_sec = (step + 1) / elapsed
        eta_seconds = (total_steps - (step + 1)) / steps_per_sec if steps_per_sec > 0 else 0
        
        # Calculate average loss over last 10 steps
        avg_loss = np.mean(losses[-10:]) if losses else 0
        
        # Print progress
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1}/{total_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Step time: {step_time:.2f}s | "
                  f"ETA: {eta_seconds/60:.1f}m")
        
        # Save checkpoint
        if (step + 1) % args.checkpoint_interval == 0 or step + 1 == total_steps:
            checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"checkpoint_step_{step+1}.pt")
            print(f"Saving checkpoint to {checkpoint_path}")
            
            # Move model to CPU for saving (prevents memory issues)
            model_cpu = model.to("cpu")
            torch.save({
                "model_state_dict": model_cpu.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step + 1,
                "losses": losses
            }, checkpoint_path)
            
            # Move model back to device
            model = model_cpu.to(device)
            
            # Clear memory after checkpoint
            del model_cpu
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Average loss: {np.mean(losses):.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "checkpoints", "final_model.pt")
    torch.save({
        "model_state_dict": model.to("cpu").state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": total_steps,
        "losses": losses
    }, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()