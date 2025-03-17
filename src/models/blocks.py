"""
Neural network building blocks for the Grateful Dead show dating model.
"""

import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and a skip connection.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        device: Device to place the module on
    """

    def __init__(self, in_channels: int, out_channels: int, device='cpu'):
        super().__init__()
        self.device = device
        
        # Create all components on the correct device from the start
        self.to(device)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(out_channels).to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(out_channels).to(device)
        self.gelu = nn.GELU().to(device)
        self.dropout = nn.Dropout2d(0.1).to(device)  # Spatial dropout

        # Skip connection projection if needed
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(device)
        else:
            self.skip = nn.Identity().to(device)
        
        # Ensure all parameters are on the target device
        for name, param in self.named_parameters():
            if param.device != torch.device(device):
                param.data = param.data.to(device)
                
        for name, buffer in self.named_buffers():
            if buffer.device != torch.device(device):
                buffer.data = buffer.data.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Apply skip connection immediately to keep device consistent
        identity = self.skip(x)
        
        # First convolution block with strict device checking
        out = self.conv1(x)
        
        # Critical section for batch norm - ensure everything is on the same device
        if out.device != self.device:
            out = out.to(self.device)
            
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)

        # Second convolution block with strict device checking
        out = self.conv2(out)
        
        if out.device != self.device:
            out = out.to(self.device)
            
        out = self.bn2(out)

        # Ensure identity and out are on the same device before addition
        if identity.device != out.device:
            identity = identity.to(out.device)
            
        out = out + identity  # Using + instead of += to avoid in-place operation
        out = self.gelu(out)
        
        # Final device check before returning
        if out.device != self.device:
            out = out.to(self.device)

        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding module that adds position information to features.
    
    Args:
        d_model: Feature dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
        device: Device to place the module on
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model, device=device)
        position = torch.arange(0, max_seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Move to specified device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Tensor with positional encoding added
        """
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Make sure the pe buffer is on the same device
        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)
            
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 