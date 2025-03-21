#!/usr/bin/env python3
"""Minimal script to check a single batch and the feature extractor."""

import os
import sys
import torch
import torch.nn.functional as F

# Add the root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.feature_extractors import ParallelFeatureNetwork

def main():
    """Test the feature extractor with a minimal example."""
    print("Testing feature extractor...")
    
    # Create a small batch with the issue
    batch = {
        'harmonic': torch.randn(2, 1, 128, 50),
        'percussive': torch.randn(2, 1, 128, 50),
        'chroma': torch.randn(2, 12, 235),
        'spectral_contrast': torch.randn(2, 6, 235)
    }
    
    # Put on MPS if available
    device = torch.device('mps' if hasattr(torch.backends, 'mps') and 
                         torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for k, v in batch.items():
        batch[k] = v.to(device)
    
    # Create fixed feature extractor
    feature_network = ParallelFeatureNetwork().to(device)
    
    # Print batch shapes
    print("\nBatch shapes:")
    for key, tensor in batch.items():
        print(f"{key}: {tensor.shape}")
    
    # Test with fixed feature extractor
    print("\nTesting fixed feature extractor...")
    output = feature_network(batch)
    print(f"Output shape: {output.shape}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()