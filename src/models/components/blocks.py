"""
Neural network building blocks for the Grateful Dead show dating model.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and a skip connection.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)  # Spatial dropout

        # Skip connection projection if needed
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        # Apply skip connection
        identity = self.skip(x)
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)

        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection and apply activation
        out = out + identity
        out = self.gelu(out)

        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding module that adds position information to features.
    
    Args:
        d_model: Feature dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding buffer
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)