#!/usr/bin/env python3
"""
Performance profiling script for training loop.
"""

import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
import torch.profiler

from src.models.dead_model import DeadShowDatingModel
from src.data.dataset import PreprocessedDeadShowDataset, optimized_collate_fn
from src.training.trainer import Trainer
from src.config import Config


def profile_forward_pass(model, batch, device, iterations=10):
    """Profile the forward pass of the model."""
    # Move data to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Warmup
    model.train()
    _ = model(batch)
    
    # Profile
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = model(batch)
        end = time.time()
        times.append(end - start)
    
    # Report results
    avg_time = sum(times) / len(times)
    print(f"Forward pass: {avg_time:.4f} seconds (avg of {iterations} runs)")
    print(f"Range: {min(times):.4f} - {max(times):.4f} seconds")
    
    return avg_time


def profile_backwards_pass(model, batch, device, criterion, iterations=10):
    """Profile the backward pass of the model."""
    # Move data to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Prepare labels
    label = batch.get('label')
    era = batch.get('era')
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Warmup
    model.train()
    outputs = model(batch)
    loss_dict = criterion(outputs, {'days': label, 'era': era}, 0, 1000)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Profile
    times = []
    for step in range(iterations):
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss_dict = criterion(outputs, {'days': label, 'era': era}, step, 1000)
        loss = loss_dict['loss']
        
        # Record backward pass time
        start = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time.time()
        
        times.append(end - start)
    
    # Report results
    avg_time = sum(times) / len(times)
    print(f"Backward pass: {avg_time:.4f} seconds (avg of {iterations} runs)")
    print(f"Range: {min(times):.4f} - {max(times):.4f} seconds")
    
    return avg_time


def profile_data_loading(dataloader, iterations=5):
    """Profile data loading time."""
    times = []
    batch_sizes = []
    
    # Profile batches
    for i, batch in enumerate(dataloader):
        start = time.time()
        # Process batch (just accessing keys to simulate use)
        keys = list(batch.keys())
        if 'empty_batch' in batch and batch['empty_batch']:
            print(f"Batch {i} is empty, skipping")
            continue
        
        # Process each tensor (simulate use)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                _ = value.shape  # Just access the tensor
        
        end = time.time()
        times.append(end - start)
        
        # Record batch size
        has_batch_size = False
        for key in ['harmonic', 'percussive', 'label']:
            if key in batch and isinstance(batch[key], torch.Tensor):
                has_batch_size = True
                batch_sizes.append(batch[key].shape[0])
                break
        
        if not has_batch_size:
            batch_sizes.append(0)
        
        print(f"Batch {i}: loading took {end-start:.4f}s, size: {batch_sizes[-1]}")
        
        if i >= iterations:
            break
    
    # Report results
    if times:
        avg_time = sum(times) / len(times)
        avg_batch = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        print(f"Data loading: {avg_time:.4f} seconds per batch (avg of {len(times)} batches)")
        print(f"Average batch size: {avg_batch:.1f}")
        print(f"Range: {min(times):.4f} - {max(times):.4f} seconds")
        return avg_time
    else:
        print("No valid batches loaded")
        return 0


def profile_full_step(model, batch, device, criterion, iterations=5):
    """Profile a complete training step (forward + backward)."""
    # Move data to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Prepare labels
    label = batch.get('label')
    era = batch.get('era')
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Warmup
    model.train()
    outputs = model(batch)
    loss_dict = criterion(outputs, {'days': label, 'era': era}, 0, 1000)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Profile complete steps
    times = []
    for step in range(iterations):
        start = time.time()
        
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss_dict = criterion(outputs, {'days': label, 'era': era}, step, 1000)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        end = time.time()
        times.append(end - start)
        print(f"Step {step+1}/{iterations}: {times[-1]:.4f}s")
    
    # Report results
    avg_time = sum(times) / len(times)
    print(f"Complete step: {avg_time:.4f} seconds (avg of {iterations} runs)")
    print(f"Range: {min(times):.4f} - {max(times):.4f} seconds")
    
    return avg_time


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Profile training performance")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for testing")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for profiling")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Create a model
    print("Creating model...")
    model = DeadShowDatingModel(sample_rate=24000)
    model = model.to(device)
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.data_dir}...")
    dataset = PreprocessedDeadShowDataset(
        preprocessed_dir=args.data_dir,
        device=torch.device("cpu"),  # Load data on CPU first
    )
    
    # Create dataloader
    print(f"Creating dataloader with batch size {args.batch_size}...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # For MPS compatibility
        collate_fn=lambda batch: optimized_collate_fn(batch, device=None),
        drop_last=True
    )
    
    print("\n1. PROFILING DATA LOADING")
    print("=" * 50)
    data_time = profile_data_loading(dataloader, iterations=args.iterations)
    
    # Get a batch for profiling model operations
    print("\nFetching a batch for model profiling...")
    batch = None
    for i, b in enumerate(dataloader):
        if not b.get('empty_batch', False):
            batch = b
            break
        if i >= 5:  # Try a few batches
            break
    
    if batch is None:
        print("ERROR: Could not get a valid batch for profiling")
        return
    
    # Create loss function
    from src.models import CombinedDeadLoss
    criterion = CombinedDeadLoss()
    
    print("\n2. PROFILING FORWARD PASS")
    print("=" * 50)
    forward_time = profile_forward_pass(model, batch, device, iterations=args.iterations)
    
    print("\n3. PROFILING BACKWARD PASS")
    print("=" * 50)
    backward_time = profile_backwards_pass(model, batch, device, criterion, iterations=args.iterations)
    
    print("\n4. PROFILING COMPLETE TRAINING STEP")
    print("=" * 50)
    step_time = profile_full_step(model, batch, device, criterion, iterations=args.iterations)
    
    print("\n5. PROFILING WITH PYTORCH PROFILER")
    print("=" * 50)
    # Move batch to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Prepare labels
    label = batch.get('label')
    era = batch.get('era')
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Profiling
    print("Running PyTorch profiler...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if device.type == 'cuda' else None
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss_dict = criterion(outputs, {'days': label, 'era': era}, 0, 1000)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Print profiler results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Summary
    print("\nPERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Data loading:     {data_time:.4f} seconds per batch")
    print(f"Forward pass:     {forward_time:.4f} seconds")
    print(f"Backward pass:    {backward_time:.4f} seconds")
    print(f"Complete step:    {step_time:.4f} seconds")
    print(f"Theoretical training time for 1000 steps: {step_time * 1000 / 3600:.2f} hours")
    
    # Recommendations
    print("\nRECOMMENDATIONS")
    print("=" * 50)
    
    if data_time > (forward_time + backward_time) * 0.5:
        print("- Data loading appears to be a bottleneck. Consider:")
        print("  * Reducing preprocessing complexity")
        print("  * Using a smaller subset of data for faster iterations")
        print("  * Adding more workers to the DataLoader (if not using MPS)")
        print("  * Preprocessing data more efficiently")
    
    if device.type == 'mps':
        print("- MPS optimization recommendations:")
        print("  * Avoid frequent CPU-GPU transfers")
        print("  * Batch operations where possible")
        print("  * Consider using smaller precision (float16) if accuracy allows")
        print("  * Regularly clear memory cache with torch.mps.empty_cache()")
    

if __name__ == "__main__":
    main()