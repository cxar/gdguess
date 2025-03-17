#!/usr/bin/env python3
"""
Utilities for handling data type consistency in models.
"""

import torch
import torch.nn as nn
import sys

def force_dtype(model, dtype=torch.float32, device=None):
    """
    Recursively force all parameters and buffers in a model to specified dtype.
    This function will directly modify the model parameters.
    
    Args:
        model: PyTorch model or module
        dtype: Target data type (default: torch.float32)
        device: Optional device to move parameters to
        
    Returns:
        The model with all parameters converted to the specified dtype
    """
    # Process parameters
    for param in model.parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)
        if device is not None and param.device != torch.device(device):
            param.data = param.data.to(device)
    
    # Process buffers
    for buffer_name, buffer in model.named_buffers():
        if buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype)
        if device is not None and buffer.device != torch.device(device):
            buffer.data = buffer.data.to(device)
    
    return model

def verify_model_dtype(model, target_dtype=torch.float32):
    """
    Verify that all parameters and buffers in a model are of the specified dtype.
    
    Args:
        model: PyTorch model or module
        target_dtype: Expected data type
        
    Returns:
        Tuple of (all_match, num_mismatched, total_params, mismatched_params)
    """
    mismatched = []
    
    # Check parameters
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            mismatched.append((name, param.dtype, param.shape))
    
    # Check buffers
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype:
            mismatched.append((name, buffer.dtype, buffer.shape))
    
    total = len(list(model.parameters())) + len(list(model.buffers()))
    return len(mismatched) == 0, len(mismatched), total, mismatched

def disable_half_precision(model):
    """
    Ensure the model doesn't use half precision at all.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with half precision disabled
    """
    # Disable amp/autocast if being used
    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        torch.amp.autocast('cuda', enabled=False).__enter__()
    
    # Convert model to full precision
    model = force_dtype(model, dtype=torch.float32)
    
    # Disable mixed precision in cudnn
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = True
    
    # Disable NVIDIA apex if it's present
    if 'apex' in sys.modules:
        try:
            from apex import amp
            amp.init(enabled=False)
        except (ImportError, ModuleNotFoundError):
            pass
            
    return model 