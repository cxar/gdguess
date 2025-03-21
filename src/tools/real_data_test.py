#!/usr/bin/env python3
"""
Test script using real data to profile training performance.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from src.models.dead_model import DeadShowDatingModel
from src.models import CombinedDeadLoss
from src.data.dataset import PreprocessedDeadShowDataset, optimized_collate_fn


def profile_training_step(model, batch, device, criterion, optimizer, step=0, total_steps=1000):
    """Profile a full training step with real data."""
    # Time moving batch to device
    device_start = time.time()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    device_end = time.time()
    device_time = device_end - device_start
    
    # Forward pass
    forward_start = time.time()
    outputs = model(batch)
    forward_end = time.time()
    forward_time = forward_end - forward_start
    
    # Loss calculation
    loss_start = time.time()
    loss_dict = criterion(
        outputs,
        {'days': batch.get('label'), 'era': batch.get('era')},
        step,
        total_steps
    )
    loss = loss_dict['loss']
    loss_end = time.time()
    loss_time = loss_end - loss_start
    
    # Backward pass
    backward_start = time.time()
    loss.backward()
    backward_end = time.time()
    backward_time = backward_end - backward_start
    
    # Optimizer step
    optim_start = time.time()
    optimizer.step()
    optimizer.zero_grad()
    optim_end = time.time()
    optim_time = optim_end - optim_start
    
    # Memory cleanup for MPS
    gc_start = time.time()
    if device.type == 'mps':
        if 'outputs' in locals(): del outputs
        if 'loss' in locals(): del loss
        if 'loss_dict' in locals(): del loss_dict
        
        import gc
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    gc_end = time.time()
    gc_time = gc_end - gc_start
    
    # Total time
    total_time = device_time + forward_time + loss_time + backward_time + optim_time + gc_time
    
    # Return timing breakdown
    return {
        'device': device_time,
        'forward': forward_time,
        'loss': loss_time,
        'backward': backward_time,
        'optim': optim_time,
        'gc': gc_time,
        'total': total_time,
        'loss_value': loss.item() if 'loss' in locals() else float('nan')
    }


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Profile training with real data")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps to profile")
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.data_dir}...")
    start_time = time.time()
    dataset = PreprocessedDeadShowDataset(
        preprocessed_dir=args.data_dir,
        device=torch.device("cpu"),  # Always load data on CPU first
        memory_efficient=True,  # Use memory-efficient mode
    )
    dataset_time = time.time() - start_time
    print(f"Dataset loaded in {dataset_time:.2f} seconds, found {len(dataset)} files")
    
    # Create dataloader
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
    
    # Create model
    print("Creating model...")
    model = DeadShowDatingModel(sample_rate=24000)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Create loss function
    criterion = CombinedDeadLoss()
    
    # Set model to training mode
    model.train()
    
    # Fetch a batch
    print("Fetching a batch...")
    batch_fetch_start = time.time()
    batch = None
    for i, b in enumerate(dataloader):
        if not b.get('empty_batch', False):
            batch = b
            break
        if i >= 5:  # Try a few batches
            break
    
    batch_fetch_time = time.time() - batch_fetch_start
    print(f"Batch fetched in {batch_fetch_time:.2f} seconds")
    
    if batch is None:
        print("ERROR: Could not get a valid batch for profiling")
        return
    
    # Print batch info
    print("\nBatch Contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Warmup
    print("\nWarming up...")
    warmup_start = time.time()
    
    # Move batch to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Forward & backward pass
    outputs = model(batch)
    loss_dict = criterion(
        outputs,
        {'days': batch.get('label'), 'era': batch.get('era')},
        0,
        1000
    )
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Clear memory
    del outputs, loss, loss_dict
    import gc
    gc.collect()
    if device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        
    warmup_time = time.time() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f} seconds")
    
    # Training loop with timing
    print(f"\nRunning {args.steps} training steps with detailed timing...")
    
    # Fetch a new batch for each step to better simulate real training
    use_new_batch_each_time = False
    
    timings = []
    dataloader_iter = iter(dataloader)
    
    for step in range(args.steps):
        print(f"\nStep {step+1}/{args.steps}:")
        
        # Get a new batch if needed
        if use_new_batch_each_time or step == 0:
            batch_fetch_start = time.time()
            try:
                new_batch = next(dataloader_iter)
                if not new_batch.get('empty_batch', False):
                    batch = new_batch
                    print(f"  New batch fetched in {time.time() - batch_fetch_start:.4f}s")
                else:
                    print(f"  Skipping empty batch, reusing previous")
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
                print(f"  Reset dataloader, fetched new batch in {time.time() - batch_fetch_start:.4f}s")
        
        # Profile this step
        timing = profile_training_step(model, batch, device, criterion, optimizer, step, args.steps)
        timings.append(timing)
        
        # Print timing details
        print(f"  Device:   {timing['device']:.4f}s ({timing['device']/timing['total']*100:.1f}%)")
        print(f"  Forward:  {timing['forward']:.4f}s ({timing['forward']/timing['total']*100:.1f}%)")
        print(f"  Loss:     {timing['loss']:.4f}s ({timing['loss']/timing['total']*100:.1f}%)")
        print(f"  Backward: {timing['backward']:.4f}s ({timing['backward']/timing['total']*100:.1f}%)")
        print(f"  Optim:    {timing['optim']:.4f}s ({timing['optim']/timing['total']*100:.1f}%)")
        print(f"  GC:       {timing['gc']:.4f}s ({timing['gc']/timing['total']*100:.1f}%)")
        print(f"  Total:    {timing['total']:.4f}s")
        print(f"  Loss value: {timing['loss_value']:.4f}")
    
    # Print summary
    print("\nPerformance Summary:")
    print("=" * 50)
    
    # Calculate averages
    avg_timings = {
        'device': sum(t['device'] for t in timings) / len(timings),
        'forward': sum(t['forward'] for t in timings) / len(timings),
        'loss': sum(t['loss'] for t in timings) / len(timings),
        'backward': sum(t['backward'] for t in timings) / len(timings),
        'optim': sum(t['optim'] for t in timings) / len(timings),
        'gc': sum(t['gc'] for t in timings) / len(timings),
        'total': sum(t['total'] for t in timings) / len(timings),
    }
    
    # Compute percentages
    total_time = avg_timings['total']
    
    print(f"Average step time: {total_time:.4f} seconds")
    print(f"  Device:   {avg_timings['device']:.4f}s ({avg_timings['device']/total_time*100:.1f}%)")
    print(f"  Forward:  {avg_timings['forward']:.4f}s ({avg_timings['forward']/total_time*100:.1f}%)")
    print(f"  Loss:     {avg_timings['loss']:.4f}s ({avg_timings['loss']/total_time*100:.1f}%)")
    print(f"  Backward: {avg_timings['backward']:.4f}s ({avg_timings['backward']/total_time*100:.1f}%)")
    print(f"  Optim:    {avg_timings['optim']:.4f}s ({avg_timings['optim']/total_time*100:.1f}%)")
    print(f"  GC:       {avg_timings['gc']:.4f}s ({avg_timings['gc']/total_time*100:.1f}%)")
    
    print(f"\nEstimated time for 1000 steps: {total_time * 1000 / 60:.1f} minutes")
    
    # Add dataloader time to total estimate
    batch_time = batch_fetch_time / args.batch_size  # Time per sample
    print(f"Data loading time (per batch): {batch_fetch_time:.4f}s")
    
    # Compare with synthetic data
    print("\nComparison with synthetic data performance (0.5s per step):")
    speed_ratio = total_time / 0.5
    print(f"Real data is approximately {speed_ratio:.1f}x slower than synthetic data")
    
    # Recommendations
    print("\nRecommendations:")
    print("=" * 50)
    
    # Identify the bottleneck
    bottleneck = max(avg_timings.items(), key=lambda x: x[1] if x[0] != 'total' else 0)
    
    if bottleneck[0] == 'device':
        print("Bottleneck: Data transfer to device")
        print("- Consider preprocessing data to reduce transfer size")
        print("- Try using float16 precision to reduce memory transfers")
        print("- Batch data more efficiently")
    elif bottleneck[0] == 'forward':
        print("Bottleneck: Forward pass computation")
        print("- Optimize model architecture for MPS")
        print("- Consider using smaller tensors or simpler architecture")
        print("- Try mixed precision training (float16)")
    elif bottleneck[0] == 'backward':
        print("Bottleneck: Backward pass computation")
        print("- Use gradient checkpointing to reduce memory usage")
        print("- Reduce model complexity")
        print("- Try different optimizer (e.g., SGD instead of Adam)")
    elif bottleneck[0] == 'gc':
        print("Bottleneck: Memory management")
        print("- Reduce batch size to minimize memory pressure")
        print("- Optimize tensor operations to reduce fragmentation")
    
    # MPS-specific recommendations
    if device.type == 'mps':
        print("\nMPS-specific optimization recommendations:")
        print("1. Use batch size of 1 (MPS often performs better with smaller batches)")
        print("2. Try float16 precision (torch.float16) for significant speedup")
        print("3. Clear MPS cache regularly between steps")
        print("4. Simplify model architecture if possible")


if __name__ == "__main__":
    main()