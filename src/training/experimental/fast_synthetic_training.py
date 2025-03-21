#!/usr/bin/env python3
"""
Fast training script for testing purposes.
Uses a smaller dataset and fewer steps to quickly test training loop.
"""

import os
import sys
import torch
import random
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from src
from src.models.dead_model import DeadShowDatingModel
from src.models import CombinedDeadLoss

def main():
    print("=== Fast Training Test ===")
    
    # Set device
    device = torch.device('mps' if hasattr(torch.backends, 'mps') and 
                         torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DeadShowDatingModel().to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    criterion = CombinedDeadLoss()
    
    # Set model to training mode
    model.train()
    
    # Generate some synthetic data
    batch_size = 4
    num_steps = 5
    
    print("\nRunning training loop...")
    for step in tqdm(range(num_steps)):
        # Create synthetic batch that mimics real data
        batch = {
            'harmonic': torch.randn(batch_size, 1, 128, 50).to(device),
            'percussive': torch.randn(batch_size, 1, 128, 50).to(device),
            'chroma': torch.randn(batch_size, 12, 235).to(device),  # Different time dimension
            'spectral_contrast': torch.randn(batch_size, 6, 235).to(device),  # Different time dimension
            'label': torch.randint(0, 10000, (batch_size, 1)).float().to(device),
            'era': torch.randint(0, 5, (batch_size,)).to(device),
            'year': torch.randint(1970, 1995, (batch_size,)).to(device)
        }
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Calculate loss
        loss_dict = criterion(
            outputs,
            {'days': batch['label'], 'era': batch['era']},
            step,
            num_steps
        )
        loss = loss_dict['loss']
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in loss - skipping batch")
            continue
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress
        print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # Clean up for MPS
        if device.type == 'mps':
            del outputs, loss, loss_dict
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    print("\nTraining completed successfully!")
    print("The model can now handle different time dimensions in feature inputs.")

if __name__ == "__main__":
    main()