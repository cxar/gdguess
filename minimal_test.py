#!/usr/bin/env python3
"""
Minimal test script to debug the model error.
"""

import torch
from src.models.dead_model import ParallelFeatureNetwork

def debug_feature_network():
    """Debug the ParallelFeatureNetwork to identify the issue."""
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create the feature network only
    feature_network = ParallelFeatureNetwork(device=device)
    
    # Create dummy features with proper shapes
    batch_size = 2
    time_steps = 50
    
    # Based on error message, the shapes need to be:
    # 'harmonic' and 'percussive': [batch, channel=1, freq, time]
    # 'chroma': [batch, channel=12, time] - needs 12 channels, not 1 channel with 12 features
    # 'spectral_contrast': [batch, channel=6, time] - needs 6 channels
    dummy_features = {
        'harmonic': torch.rand(batch_size, 1, 128, time_steps, device=device),
        'percussive': torch.rand(batch_size, 1, 128, time_steps, device=device),
        'chroma': torch.rand(batch_size, 12, time_steps, device=device),  # 12 channels directly
        'spectral_contrast': torch.rand(batch_size, 6, time_steps, device=device),  # 6 channels directly
    }
    
    # Print the shapes of inputs for debugging
    print("Input shapes:")
    for key, tensor in dummy_features.items():
        print(f"  {key}: {tensor.shape}")
    
    # Try to process through the feature network
    try:
        with torch.no_grad():
            output = feature_network(dummy_features)
        print(f"Output shape: {output.shape}")
        print("Feature network forward pass successful")
    except Exception as e:
        print(f"Error in feature network: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_network() 