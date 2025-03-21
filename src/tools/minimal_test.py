#!/usr/bin/env python3
"""
Minimal test script that uses synthetic data to test model and feature extractor.
"""

import os
import sys
import torch
import time

# Add to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import only what we need
from src.models.feature_extractors import ParallelFeatureNetwork
from src.models.dead_model import DeadShowDatingModel
from src.models.losses import CombinedDeadLoss

def main():
    print("=== Model Test with Synthetic Data ===")
    
    # Set device
    device = torch.device('mps' if hasattr(torch.backends, 'mps') and 
                        torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data - deliberately use different time dimensions
    batch_size = 2
    batch = {
        'harmonic': torch.randn(batch_size, 1, 128, 50).to(device),
        'percussive': torch.randn(batch_size, 1, 128, 50).to(device),
        'chroma': torch.randn(batch_size, 12, 235).to(device),
        'spectral_contrast': torch.randn(batch_size, 6, 235).to(device),
        'label': torch.randint(0, 10000, (batch_size, 1)).float().to(device),
        'era': torch.randint(0, 5, (batch_size,)).to(device),
        'year': torch.randint(1970, 1995, (batch_size,)).to(device)
    }
    
    # Create model and components
    print("\nInitializing model...")
    feature_network = ParallelFeatureNetwork().to(device)
    model = DeadShowDatingModel().to(device)
    criterion = CombinedDeadLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test feature network independently
    print("\nTesting feature network...")
    start_time = time.time()
    features_output = feature_network(batch)
    print(f"Feature network output shape: {features_output.shape}")
    print(f"Feature extraction took {time.time() - start_time:.4f} seconds")
    
    # Test model with updated feature extractor
    print("\nTesting full model with synthetic data...")
    
    # Time the forward pass
    start_time = time.time()
    outputs = model(batch)
    forward_time = time.time() - start_time
    print(f"Forward pass took {forward_time:.4f} seconds")
    
    # Time the loss calculation 
    start_time = time.time()
    loss_dict = criterion(
        outputs,
        {'days': batch['label'], 'era': batch['era']}, 
        0, 1
    )
    loss = loss_dict['loss']
    loss_time = time.time() - start_time
    print(f"Loss calculation took {loss_time:.4f} seconds")
    
    # Time backward pass
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    print(f"Backward pass took {backward_time:.4f} seconds")
    
    # Time optimizer step
    start_time = time.time()
    optimizer.step()
    optimizer.zero_grad()
    optim_time = time.time() - start_time
    print(f"Optimizer step took {optim_time:.4f} seconds")
    
    # Total time
    total_time = forward_time + loss_time + backward_time + optim_time
    print(f"\nTotal training step time: {total_time:.4f} seconds")
    
    # Check loss value
    print(f"Loss value: {loss.item():.4f}")
    
    print("\nModel test completed successfully!")
    print("Feature extraction can now handle different time dimensions correctly.")

if __name__ == "__main__":
    main()