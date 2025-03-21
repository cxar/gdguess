#!/usr/bin/env python3
"""Debug script to check dataset and dataloader."""

import os
import sys
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.feature_extractors import ParallelFeatureNetwork

# Create a simple config class
class SimpleConfig:
    def __init__(self):
        self.data_dir = "./data/audsnippets-all/preprocessed"
        self.batch_size = 2
        self.device = "mps"

def main():
    """Main function to debug dataloader and model."""
    print("=== Feature Extractor Debug Script ===")
    
    # Initialize config
    config = SimpleConfig()
    
    # Get device
    device = torch.device('mps' if hasattr(torch.backends, 'mps') and 
                       torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a sample batch with the known problematic shapes
    batch = {
        'harmonic': torch.randn(2, 1, 128, 50),
        'percussive': torch.randn(2, 1, 128, 50),
        'chroma': torch.randn(2, 12, 235),  # Deliberately different time dimension
        'spectral_contrast': torch.randn(2, 6, 235)  # Deliberately different time dimension
    }
    
    # Move to device
    for k, v in batch.items():
        batch[k] = v.to(device)
    
    # Create feature network (the part that had the issue)
    feature_network = ParallelFeatureNetwork().to(device)
    
    # Print input shapes
    print("\n=== Sample batch shapes ===")
    for key, tensor in batch.items():
        print(f"{key}: {tensor.shape}")
    
    # Try feature extraction with the fixed model
    try:
        print("\nRunning feature extraction...")
        out = feature_network(batch)
        print(f"Feature extraction success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDebug completed.")

if __name__ == "__main__":
    main()