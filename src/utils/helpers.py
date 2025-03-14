#!/usr/bin/env python3
"""
Utility functions for the Grateful Dead show dating model.
"""

import torch
import torch.nn as nn


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
