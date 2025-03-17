#!/usr/bin/env python3
"""
Simple test for the Transformer encoder implementation.
This script directly tests the Transformer encoder component.
"""

import os
import sys
import torch
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dead_model import DeadShowDatingModel, PositionalEncoding

def test_transformer_encoder():
    """Test the Transformer encoder component directly."""
    print("\n" + "="*50)
    print("TRANSFORMER ENCODER COMPONENT TEST")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    print("\nInitializing model...")
    model = DeadShowDatingModel(sample_rate=24000)
    
    # Extract the transformer encoder and positional encoding components
    transformer_encoder = model.transformer_encoder
    pos_encoder = model.pos_encoder
    
    # Move components to device
    transformer_encoder.to(device)
    pos_encoder.to(device)
    
    # Create random input tensor (batch_size, seq_len, hidden_dim)
    batch_size = 2
    seq_len = 100
    hidden_dim = 256
    
    print(f"\nCreating random input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]")
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
    
    # Test positional encoding
    print("\nTesting positional encoding...")
    try:
        pos_encoded = pos_encoder(x)
        pos_encoding_success = True
        print("✅ Positional encoding passed")
        
        # Check that positional encoding changed the input
        diff = torch.abs(x - pos_encoded).mean().item()
        print(f"Positional encoding difference: {diff:.6f}")
        if diff > 0:
            print("✅ Positional encoding is modifying the input")
        else:
            print("❌ Positional encoding is not having any effect")
    except Exception as e:
        pos_encoding_success = False
        print(f"❌ Positional encoding failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test transformer encoder
    print("\nTesting transformer encoder...")
    if pos_encoding_success:
        try:
            transformer_output = transformer_encoder(pos_encoded)
            transformer_success = True
            print("✅ Transformer encoder passed")
            
            # Check output shape
            expected_shape = (batch_size, seq_len, hidden_dim)
            actual_shape = tuple(transformer_output.shape)
            
            if actual_shape == expected_shape:
                print(f"✅ Output shape is correct: {actual_shape}")
            else:
                print(f"❌ Output shape is incorrect: {actual_shape}, expected: {expected_shape}")
            
            # Check that transformer changed the input
            diff = torch.abs(pos_encoded - transformer_output).mean().item()
            print(f"Transformer difference: {diff:.6f}")
            if diff > 0:
                print("✅ Transformer encoder is modifying the input")
            else:
                print("❌ Transformer encoder is not having any effect")
            
            # Check for NaN values
            if torch.isnan(transformer_output).any():
                print("❌ Output contains NaN values")
            else:
                print("✅ No NaN values in output")
        except Exception as e:
            transformer_success = False
            print(f"❌ Transformer encoder failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        transformer_success = False
        print("❌ Skipping transformer encoder test due to positional encoding failure")
    
    # Overall test result
    print("\n" + "="*50)
    if pos_encoding_success and transformer_success:
        print("✅ TRANSFORMER ENCODER COMPONENT TEST PASSED")
    else:
        print("❌ TRANSFORMER ENCODER COMPONENT TEST FAILED")
    print("="*50)
    
    return pos_encoding_success and transformer_success

if __name__ == "__main__":
    test_transformer_encoder() 