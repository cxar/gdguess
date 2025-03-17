"""
Attention mechanisms for the Grateful Dead show dating model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional


class FrequencyTimeAttention(nn.Module):
    """
    Memory-efficient multi-head self-attention module that processes frequency and time dimensions separately.
    Uses layer normalization and chunked attention computation for better stability and memory usage.
    
    Args:
        dim: Input feature dimension
        heads: Number of attention heads
        dropout: Dropout probability
        chunk_size: Size of chunks for memory-efficient attention computation
    """
    
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, chunk_size: int = 128):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.chunk_size = chunk_size
        
        # Input projection
        self.to_qkv = nn.Linear(dim, dim * 3)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _reshape_qkv_tensor(self, tensor: Tensor, b: int, f: int, t: int) -> Tensor:
        """
        Reshape a tensor for attention computation.
        """
        return tensor.reshape(b, f, t, self.heads, -1).permute(0, 3, 1, 2, 4)

    def _chunk_attention(self, q: Tensor, k: Tensor, v: Tensor, chunk_size: int) -> Tensor:
        """
        Compute attention in chunks to save memory.
        """
        # Ensure q, k, v are 4D tensors (batch, heads, seq_len, dim)
        if q.dim() == 3:  # [batch*heads, seq_len, dim]
            batch_heads = q.size(0)
            seq_len = q.size(1)
        elif q.dim() == 4:  # [batch, heads, seq_len, dim]
            _, _, seq_len, _ = q.shape
        else:
            raise ValueError(f"Expected 3D or 4D query tensor, got shape {q.shape}")
            
        out = []
        
        # Process queries in chunks
        for i in range(0, seq_len, chunk_size):
            if q.dim() == 3:
                chunk_q = q[:, i:i + chunk_size]
            else:  # q.dim() == 4
                chunk_q = q[:, :, i:i + chunk_size]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(chunk_q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            chunk_out = torch.matmul(attn, v)
            out.append(chunk_out)
        
        # Concatenate along sequence dimension
        return torch.cat(out, dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply frequency-time attention to the input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, freq, time)
            
        Returns:
            Tensor of same shape as input with attention applied
        """
        b, c, f, t = x.shape
        
        # Reshape input for attention
        x_orig = x
        x = x.permute(0, 2, 3, 1)  # (b, f, t, c)
        x_norm = self.norm1(x)
        
        # Project to q, k, v
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention
        q = self._reshape_qkv_tensor(q, b, f, t)
        k = self._reshape_qkv_tensor(k, b, f, t)
        v = self._reshape_qkv_tensor(v, b, f, t)
        
        # Reshape for efficient attention
        q = q.reshape(b * self.heads, f * t, -1)
        k = k.reshape(b * self.heads, f * t, -1)
        v = v.reshape(b * self.heads, f * t, -1)
        
        # Chunked attention computation
        out = self._chunk_attention(q, k, v, self.chunk_size)
        
        # Reshape output
        out = out.reshape(b, self.heads, f, t, -1)
        out = out.permute(0, 2, 3, 1, 4).reshape(b, f, t, -1)
        
        # Output projection and residual connection
        out = self.norm2(out)
        out = self.to_out(out)
        out = out + x  # Residual connection
        
        # Restore original shape
        out = out.permute(0, 3, 1, 2)  # (b, c, f, t)
        
        return out