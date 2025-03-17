#!/usr/bin/env python3
"""
Utility functions for the Grateful Dead show dating model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Any, Optional, Tuple


def reset_parameters(model: nn.Module) -> None:
    """
    Enhanced parameter initialization to prevent training issues.

    Args:
        model: Neural network module to initialize
    """
    print("Applying enhanced initialization...")

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming initialization for convolutions based on activation function
            if isinstance(m.weight, nn.Parameter):
                if hasattr(m, "gelu"):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            # Initialize linear layers with a combination of Xavier and small variance
            gain = 0.6
            if m.out_features == 1:  # For regression output
                gain = 0.1  # Even smaller for final regression layer

            nn.init.xavier_normal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LSTM):
            # Orthogonal initialization for recurrent layers
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param, gain=0.6)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param, gain=0.6)
                elif "bias" in name:
                    # Initialize forget gate biases to 1.0 as per best practices
                    # This helps prevent vanishing gradients
                    bias = param
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

        elif isinstance(m, nn.TransformerEncoder) or isinstance(m, nn.TransformerEncoderLayer):
            # Initialize transformer layers
            for name, param in m.named_parameters():
                if param.dim() > 1:
                    # Initialize weight matrices with Xavier
                    nn.init.xavier_uniform_(param, gain=0.6)
                elif "bias" in name:
                    # Initialize bias vectors to zero
                    nn.init.zeros_(param)

        elif isinstance(m, nn.MultiheadAttention):
            # Initialize attention layers
            for name, param in m.named_parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.6)

        # Check for ResidualBlock class
        elif hasattr(model, "ResidualBlock") and isinstance(m, model.ResidualBlock):
            # Special initialization for residual blocks
            for name, module in m.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Initialize last layer of each residual block with zeros
                    # This makes residual blocks behave like identity at start
                    if "conv2" in name:
                        nn.init.zeros_(module.weight)
                    else:
                        nn.init.kaiming_normal_(
                            module.weight, mode="fan_out", nonlinearity="relu"
                        )

    print("Enhanced initialization applied")

    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} total parameters")


def ensure_device_consistency(model: nn.Module, config: Any) -> nn.Module:
    """
    Ensure all model parameters and buffers are on the correct device.
    
    Args:
        model: PyTorch model
        config: Configuration object with device settings
        
    Returns:
        Model with all parameters and buffers on the target device
    """
    target_device = torch.device(config.device)
    model = model.to(target_device)
    
    # Verify all parameters and buffers are on the correct device
    for name, param in model.named_parameters():
        if param.device != target_device:
            print(f"Warning: Parameter '{name}' on {param.device}, moving to {target_device}")
            param.data = param.data.to(target_device)
    
    for name, buffer in model.named_buffers():
        if buffer.device != target_device:
            print(f"Warning: Buffer '{name}' on {buffer.device}, moving to {target_device}")
            buffer.data = buffer.data.to(target_device)
    
    return model


def ensure_compile_device_consistency(model: nn.Module, target_device: torch.device) -> None:
    """
    Special device consistency function for torch.compile compatibility.
    
    Args:
        model: PyTorch model that might be wrapped by torch.compile
        target_device: Target device for all parameters and buffers
    """
    try:
        # For torch.compile models, we need to access the original module
        if hasattr(model, '_orig_mod'):
            # Handle compiled model specifically
            orig_mod = model._orig_mod
            
            # Force all parameters to the target device
            for name, param in orig_mod.named_parameters():
                if param.device != target_device:
                    param.data = param.data.to(target_device)
            
            # Force all buffers to the target device
            for name, buffer in orig_mod.named_buffers():
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)
                    
        elif hasattr(model, 'module'):  # Handle DataParallel
            for name, param in model.module.named_parameters():
                if param.device != target_device:
                    param.data = param.data.to(target_device)
                    
            for name, buffer in model.module.named_buffers():
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)
        else:
            # Regular model, just call the standard function
            for name, param in model.named_parameters():
                if param.device != target_device:
                    param.data = param.data.to(target_device)
                    
            for name, buffer in model.named_buffers():
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)
    except Exception as e:
        print(f"Warning: Device consistency check encountered an error: {e}")


def prepare_batch(batch: Any, config: Any) -> Any:
    """
    Move batch data to the correct device.
    
    Args:
        batch: A batch of data (typically a dictionary of tensors)
        config: Configuration object with device settings
        
    Returns:
        Batch with all tensors moved to the target device
    """
    if not getattr(config, "force_inputs_to_device", True):
        return batch
    
    target_device = torch.device(config.device)
    
    if isinstance(batch, torch.Tensor):
        return batch.to(target_device)
    elif isinstance(batch, dict):
        return {k: prepare_batch(v, config) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(prepare_batch(x, config) for x in batch)
    else:
        return batch


def cleanup_file_handles() -> int:
    """
    Force garbage collection and attempt to close any leaked file handles.
    Returns the number of objects collected.
    """
    import gc
    
    # Force a full garbage collection
    collected = gc.collect()
    
    # Release memory
    torch.cuda.empty_cache()
    
    return collected


def debug_shape(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """
    Debug helper to print tensor shapes.

    Args:
        tensor: Tensor to print shape of
        name: Name to identify the tensor in the output

    Returns:
        The input tensor (for chaining operations)
    """
    print(f"Shape of {name}: {tensor.shape}")
    return tensor
