#!/usr/bin/env python3
"""
Debug script to diagnose NaN loss in the Grateful Dead show dating model.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import argparse

from src.models.dead_model import DeadShowDatingModel
from src.models import CombinedDeadLoss
from src.data.dataset import PreprocessedDeadShowDataset, optimized_collate_fn


def debug_loss(batch, model, criterion, device):
    """Debug the loss calculation and find NaN causes."""
    # Print batch info
    print("\nBatch details:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"  ❌ {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}, CONTAINS NaN")
            elif torch.isinf(value).any():
                print(f"  ❌ {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}, CONTAINS Inf")
            else:
                print(f"  ✅ {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}, range=[{value.min().item():.4f}, {value.max().item():.4f}]")
        else:
            print(f"  {key}: {type(value)}")
    
    # Move to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Check specific fields used by loss calculation
    print("\nChecking essential fields for loss calculation:")
    if 'label' in batch:
        label = batch['label']
        print(f"  Label (target days): shape={label.shape}, range=[{label.min().item():.2f}, {label.max().item():.2f}]")
    else:
        print("  ❌ 'label' is missing from batch")
    
    if 'era' in batch:
        era = batch['era']
        print(f"  Era: shape={era.shape}, values={era.detach().cpu().numpy()}")
    else:
        print("  ❌ 'era' is missing from batch")
    
    # Forward pass
    print("\nExecuting model forward pass...")
    try:
        outputs = model(batch)
        print("  ✅ Forward pass successful")
        
        # Check outputs
        print("\nModel outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    print(f"  ❌ {key}: shape={value.shape}, CONTAINS NaN")
                elif torch.isinf(value).any():
                    print(f"  ❌ {key}: shape={value.shape}, CONTAINS Inf")
                else:
                    print(f"  ✅ {key}: shape={value.shape}, range=[{value.min().item():.4f}, {value.max().item():.4f}]")
        
        # Check specific fields
        if 'days' in outputs:
            days = outputs['days']
            print(f"  Predicted days: shape={days.shape}, range=[{days.min().item():.2f}, {days.max().item():.2f}]")
        
        if 'log_variance' in outputs:
            log_var = outputs['log_variance']
            print(f"  Log variance: shape={log_var.shape}, range=[{log_var.min().item():.2f}, {log_var.max().item():.2f}]")
            
            # Check if log_variance is extremely large or small
            if log_var.max().item() > 20:
                print("  ⚠️ log_variance has very large values (> 20), may cause numerical issues")
            if log_var.min().item() < -20:
                print("  ⚠️ log_variance has very small values (< -20), may cause numerical issues")
        
        # Loss calculation
        print("\nCalculating loss...")
        try:
            loss_dict = criterion(
                outputs,
                {'days': batch.get('label'), 'era': batch.get('era')},
                0,
                1000
            )
            
            print("Loss components:")
            for key, value in loss_dict.items():
                if torch.isnan(value).any():
                    print(f"  ❌ {key}: {value.item() if value.numel() == 1 else value.detach().cpu().numpy()} (NaN)")
                elif torch.isinf(value).any():
                    print(f"  ❌ {key}: {value.item() if value.numel() == 1 else value.detach().cpu().numpy()} (Inf)")
                else:
                    print(f"  ✅ {key}: {value.item() if value.numel() == 1 else value.detach().cpu().numpy()}")
            
            total_loss = loss_dict['loss']
            print(f"\nTotal loss: {total_loss.item()}")
            
            # Check if loss is valid for backpropagation
            if torch.isnan(total_loss):
                print("  ❌ NaN detected in total loss")
            elif torch.isinf(total_loss):
                print("  ❌ Infinity detected in total loss")
            else:
                print("  ✅ Loss is valid for backpropagation")
                
                # Test backward pass
                print("\nTesting backward pass...")
                total_loss.backward()
                print("  ✅ Backward pass successful")
                
                # Check gradients
                print("\nChecking gradients:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad
                        if torch.isnan(grad).any():
                            print(f"  ❌ {name}: gradient contains NaN")
                        elif torch.isinf(grad).any():
                            print(f"  ❌ {name}: gradient contains Inf")
                        elif grad.abs().max().item() > 100:
                            print(f"  ⚠️ {name}: gradient has large magnitude {grad.abs().max().item():.2f}")
                        # Skip printing for successful gradients to avoid cluttering output
            
        except Exception as e:
            print(f"  ❌ Error in loss calculation: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"  ❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Debug NaN loss")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.data_dir}...")
    dataset = PreprocessedDeadShowDataset(
        preprocessed_dir=args.data_dir,
        device=torch.device("cpu"),  # Always load data on CPU first
        memory_efficient=True,  # Use memory-efficient mode
    )
    print(f"Found {len(dataset)} files")
    
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
    
    # Create model
    print("Creating model...")
    model = DeadShowDatingModel(sample_rate=24000)
    model = model.to(device)
    model.train()
    
    # Create loss function
    criterion = CombinedDeadLoss()
    
    # Try debugging with different batches
    num_batches_to_try = 3
    print(f"\nTrying {num_batches_to_try} different batches to diagnose the problem:")
    
    for i, batch in enumerate(dataloader):
        print(f"\n{'='*50}")
        print(f"BATCH {i+1}/{num_batches_to_try}")
        print(f"{'='*50}")
        
        if batch.get('empty_batch', False):
            print("Empty batch, skipping")
            continue
            
        debug_loss(batch, model, criterion, device)
        
        if i >= num_batches_to_try - 1:
            break
    
    print("\nLoss debugging complete")


if __name__ == "__main__":
    main()