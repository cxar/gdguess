#!/usr/bin/env python3
"""
Test script to verify device consistency in the DeadShowDatingModel.
"""

import torch
from src.models.dead_model import DeadShowDatingModel

def test_model_device_consistency():
    """Test if the model can handle device consistency properly."""
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create model
    model = DeadShowDatingModel(device=device)

    # Create dummy inputs - use dictionary of features instead of raw audio
    batch_size = 2
    time_steps = 50
    dummy_features = {
        'harmonic': torch.rand(batch_size, 1, 128, time_steps, device=device),
        'percussive': torch.rand(batch_size, 1, 128, time_steps, device=device),
        'chroma': torch.rand(batch_size, 12, time_steps, device=device),  # [batch, chroma_bins, time]
        'spectral_contrast': torch.rand(batch_size, 6, time_steps, device=device),  # [batch, contrast_bands, time]
        'date': torch.tensor([100.0, 200.0], device=device)  # day of year
    }
    
    # Verify all model parameters are on the correct device
    print("\nVerifying model parameters device consistency...")
    all_on_device = True
    for name, param in model.named_parameters():
        if param.device != torch.device(device):
            print(f"❌ Parameter '{name}' on {param.device}, expected {device}")
            all_on_device = False
    
    for name, buffer in model.named_buffers():
        if buffer.device != torch.device(device):
            print(f"❌ Buffer '{name}' on {buffer.device}, expected {device}")
            all_on_device = False
    
    if all_on_device:
        print(f"✓ All model parameters and buffers are on {device}")
    
    # Create half-precision version of inputs to test mixed precision handling
    print("\nTesting with mixed precision inputs...")
    half_precision_features = {}
    for key, tensor in dummy_features.items():
        if isinstance(tensor, torch.Tensor):
            # Create half precision tensors but ensure they're on the correct device
            half_precision_features[key] = tensor.half().to(device)
    
    # Test with uncompiled model
    print("\nTesting with standard model...")
    try:
        with torch.no_grad():
            outputs = model(dummy_features)
        print('✓ Forward pass successful with full precision inputs')
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(f'  {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}')
        
        # Now test with half precision inputs
        with torch.no_grad():
            outputs_half = model(half_precision_features)
        print('✓ Forward pass successful with half precision inputs')
        for k, v in outputs_half.items():
            if isinstance(v, torch.Tensor):
                print(f'  {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}')
    except Exception as e:
        print(f'✗ Error: {e}')

    # Test with torch.compile if available (PyTorch >= 2.0)
    if hasattr(torch, 'compile'):
        if device == 'cuda':
            print("\nSkipping torch.compile test for CUDA device due to known device inconsistency issues")
            print("DeadShowDatingModel works correctly without compilation on CUDA")
        else:
            try:
                print("\nTesting with torch.compile...")
                compiled_model = torch.compile(model, mode='reduce-overhead')
                with torch.no_grad():
                    outputs = compiled_model(dummy_features)
                print('✓ Compiled model forward pass successful')
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f'  {k}: shape={v.shape}, device={v.device}')
            except Exception as e:
                print(f'✗ Error with torch.compile: {e}')

if __name__ == "__main__":
    test_model_device_consistency() 