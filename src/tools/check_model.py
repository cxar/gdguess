#!/usr/bin/env python
import sys
import os
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.feature_extractors import ParallelFeatureNetwork

def main():
    # Test with simple mock tensors
    batch_size = 2
    
    # Create a test batch
    batch = {
        'harmonic': torch.randn(batch_size, 1, 128, 50),
        'percussive': torch.randn(batch_size, 1, 128, 50),
        'chroma': torch.randn(batch_size, 12, 235),  # Different time dimensions
        'spectral_contrast': torch.randn(batch_size, 6, 235)  # Different time dimensions
    }
    
    # Device setup
    device = torch.device('mps' if hasattr(torch.backends, 'mps') and 
                       torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Move to device
    for k, v in batch.items():
        batch[k] = v.to(device)
        
    # Initialize model
    model = ParallelFeatureNetwork().to(device)
    
    # Print input shapes
    print("\nInput shapes:")
    for k, v in batch.items():
        print(f"{k}: {v.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    output = model(batch)
    
    # Print output shape
    print(f"Output shape: {output.shape}")
    print("Model works correctly!")

if __name__ == "__main__":
    main()