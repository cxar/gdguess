#!/usr/bin/env python3
"""
Minimal training script with performance tracing.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime

from src.models.dead_model import DeadShowDatingModel
from src.models import CombinedDeadLoss


def create_synthetic_batch(batch_size=4, device=None):
    """Create a synthetic batch for testing."""
    harmonic = torch.randn(batch_size, 1, 128, 50, device=device)
    percussive = torch.randn(batch_size, 1, 128, 50, device=device)
    chroma = torch.randn(batch_size, 12, 235, device=device)
    spectral_contrast = torch.randn(batch_size, 6, 235, device=device)
    
    # Labels
    label = torch.randint(0, 10000, (batch_size,), device=device).float()
    era = torch.randint(0, 5, (batch_size,), device=device).long()
    year = torch.randint(1965, 1995, (batch_size,), device=device).long()
    
    return {
        'harmonic': harmonic,
        'percussive': percussive,
        'chroma': chroma,
        'spectral_contrast': spectral_contrast,
        'label': label,
        'era': era,
        'year': year
    }


def main():
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
    
    # Create model
    print("Creating model...")
    model = DeadShowDatingModel(sample_rate=24000)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Create loss function
    criterion = CombinedDeadLoss()
    
    # Create synthetic batch
    print("Creating synthetic batch...")
    batch = create_synthetic_batch(batch_size=4, device=device)
    
    # Set model to training mode
    model.train()
    
    # Warmup
    print("Warming up...")
    outputs = model(batch)
    loss_dict = criterion(
        outputs,
        {'days': batch['label'], 'era': batch['era']},
        0,
        1000
    )
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Clear memory
    if device.type == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            
    # Training loop with timing
    num_steps = 10
    print(f"Running {num_steps} training steps with detailed timing...")
    
    total_forward_time = 0
    total_loss_time = 0
    total_backward_time = 0
    total_step_time = 0
    
    for step in range(num_steps):
        step_start = time.time()
        
        # Forward pass
        forward_start = time.time()
        outputs = model(batch)
        forward_end = time.time()
        
        # Loss calculation
        loss_start = time.time()
        loss_dict = criterion(
            outputs,
            {'days': batch['label'], 'era': batch['era']},
            step,
            1000
        )
        loss = loss_dict['loss']
        loss_end = time.time()
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        backward_end = time.time()
        
        # Record times
        step_time = time.time() - step_start
        forward_time = forward_end - forward_start
        loss_time = loss_end - loss_start
        backward_time = backward_end - backward_start
        
        # Accumulate times
        total_forward_time += forward_time
        total_loss_time += loss_time
        total_backward_time += backward_time
        total_step_time += step_time
        
        # Print step info
        print(f"Step {step+1}/{num_steps}:")
        print(f"  Forward: {forward_time:.4f}s ({forward_time/step_time*100:.1f}%)")
        print(f"  Loss:    {loss_time:.4f}s ({loss_time/step_time*100:.1f}%)")
        print(f"  Backward: {backward_time:.4f}s ({backward_time/step_time*100:.1f}%)")
        print(f"  Total:   {step_time:.4f}s")
        print(f"  Loss value: {loss.item():.4f}")
        
        # Clear memory
        if device.type == 'mps' and step % 3 == 0:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    # Print summary
    print("\nPerformance Summary:")
    print("=" * 50)
    avg_step = total_step_time / num_steps
    print(f"Average step time: {avg_step:.4f} seconds")
    print(f"  Forward: {total_forward_time/num_steps:.4f}s ({total_forward_time/total_step_time*100:.1f}%)")
    print(f"  Loss:    {total_loss_time/num_steps:.4f}s ({total_loss_time/total_step_time*100:.1f}%)")
    print(f"  Backward: {total_backward_time/num_steps:.4f}s ({total_backward_time/total_step_time*100:.1f}%)")
    print(f"Estimated time for 1000 steps: {avg_step * 1000 / 60:.1f} minutes")
    
    if device.type == 'mps':
        print("\nMPS Optimization Tips:")
        print("1. Consider using faster data loading")
        print("2. Try batch size of 1 if memory is an issue")
        print("3. Consider using float16 precision")
        print("4. Clear MPS cache regularly during training")


if __name__ == "__main__":
    main()