#!/usr/bin/env python3
"""
Sanity test for the Transformer encoder implementation in the DeadShowDatingModel.
This script creates a model instance, feeds it random data, and verifies the outputs.
"""

import os
import sys
import torch
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dead_model import DeadShowDatingModel
from src.utils.helpers import reset_parameters

def run_transformer_sanity_test():
    """Run a sanity test on the Transformer encoder implementation."""
    print("\n" + "="*50)
    print("TRANSFORMER ENCODER SANITY TEST")
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
    model.to(device)
    
    # Apply enhanced initialization
    reset_parameters(model)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create random audio input (batch_size, audio_length)
    batch_size = 2
    audio_length = 24000 * 15  # 15 seconds at 24kHz
    
    print(f"\nCreating random audio input: [batch_size={batch_size}, audio_length={audio_length}]")
    audio = torch.randn(batch_size, audio_length, device=device, dtype=torch.float32)
    
    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        try:
            outputs = model(audio)
            success = True
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    # Check outputs
    if success:
        print("\nChecking outputs...")
        
        # Check if we have the expected output keys
        expected_keys = ['days', 'era_logits']
        missing_keys = [key for key in expected_keys if key not in outputs]
        
        if missing_keys:
            print(f"❌ Missing expected keys in output: {missing_keys}")
        else:
            print(f"✅ All expected keys present in output")
        
        # Check output shapes
        days_shape = outputs['days'].shape
        era_logits_shape = outputs['era_logits'].shape
        
        expected_days_shape = (batch_size,)
        expected_era_logits_shape = (batch_size, 5)
        
        if days_shape == expected_days_shape:
            print(f"✅ 'days' output has correct shape: {days_shape}")
        else:
            print(f"❌ 'days' output has wrong shape: {days_shape}, expected: {expected_days_shape}")
        
        if era_logits_shape == expected_era_logits_shape:
            print(f"✅ 'era_logits' output has correct shape: {era_logits_shape}")
        else:
            print(f"❌ 'era_logits' output has wrong shape: {era_logits_shape}, expected: {expected_era_logits_shape}")
        
        # Check for NaN values
        has_nans = torch.isnan(outputs['days']).any() or torch.isnan(outputs['era_logits']).any()
        
        if has_nans:
            print("❌ Output contains NaN values")
        else:
            print("✅ No NaN values in output")
        
        # Print output values
        print("\nOutput values:")
        print(f"  days: {outputs['days']}")
        print(f"  era_logits: {outputs['era_logits']}")
        
        # Overall test result
        test_passed = (
            not missing_keys and
            days_shape == expected_days_shape and
            era_logits_shape == expected_era_logits_shape and
            not has_nans
        )
        
        print("\n" + "="*50)
        if test_passed:
            print("✅ TRANSFORMER ENCODER SANITY TEST PASSED")
        else:
            print("❌ TRANSFORMER ENCODER SANITY TEST FAILED")
        print("="*50)
    
    return success

if __name__ == "__main__":
    run_transformer_sanity_test() 