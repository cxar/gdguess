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
        device: Device to place the module on
    """
    
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, chunk_size: int = 128, device='cpu'):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.chunk_size = chunk_size
        self.device = device
        
        # Explicitly move to device first
        self.to(device)

        # Input projection
        self.to_qkv = nn.Linear(dim, dim * 3).to(device)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ).to(device)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim).to(device)
        self.norm2 = nn.LayerNorm(dim).to(device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout).to(device)
        
        # Ensure all parameters are on the right device
        for name, param in self.named_parameters():
            if param.device != torch.device(device):
                param.data = param.data.to(device)
                
        for name, buffer in self.named_buffers():
            if buffer.device != torch.device(device):
                buffer.data = buffer.data.to(device)

    def _reshape_qkv_tensor(self, tensor: Tensor, b: int, f: int, t: int) -> Tensor:
        """
        Reshape a tensor for attention computation.
        This replaces the lambda function to be compatible with torch.compile.
        
        Args:
            tensor: Input tensor
            b: Batch size
            f: Frequency dimension
            t: Time dimension
            
        Returns:
            Reshaped tensor
        """
        # Ensure tensor is on the correct device
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
            
        return tensor.reshape(b, f, t, self.heads, -1).permute(0, 3, 1, 2, 4)

    def _chunk_attention(self, q: Tensor, k: Tensor, v: Tensor, chunk_size: int) -> Tensor:
        """
        Compute attention in chunks to save memory.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            chunk_size: Size of chunks to process
            
        Returns:
            Output tensor after chunked attention
        """
        # Ensure inputs are on the correct device
        if q.device != self.device:
            q = q.to(self.device)
        if k.device != self.device:
            k = k.to(self.device)
        if v.device != self.device:
            v = v.to(self.device)
            
        # Ensure q, k, v are 4D tensors (batch, heads, seq_len, dim)
        if q.dim() == 3:  # [batch*heads, seq_len, dim]
            batch_heads = q.size(0)
            heads = self.heads
            batch = batch_heads // heads
            seq_len = q.size(1)
            dim = q.size(2)
        elif q.dim() == 4:  # [batch, heads, seq_len, dim]
            batch, heads, seq_len, dim = q.shape
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
        
        result = torch.cat(out, dim=-2)  # Concatenate along sequence dimension
        
        # Ensure output is on the correct device
        if result.device != self.device:
            result = result.to(self.device)
            
        return result

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply frequency-time attention to the input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, freq, time)
            
        Returns:
            Tensor of same shape as input with attention applied
        """
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        b, c, f, t = x.shape
        
        # Reshape and normalize input
        x = x.permute(0, 2, 3, 1)  # (b, f, t, c)
        
        # Ensure x is on correct device before normalization
        if x.device != self.device:
            x = x.to(self.device)
        
        x = self.norm1(x)
        
        # Project to q, k, v with device checking
        qkv = self.to_qkv(x)
        if qkv.device != self.device:
            qkv = qkv.to(self.device)
            
        qkv_chunks = qkv.chunk(3, dim=-1)
        
        # Reshape with explicit device checks
        q = self._reshape_qkv_tensor(qkv_chunks[0], b, f, t)
        k = self._reshape_qkv_tensor(qkv_chunks[1], b, f, t)
        v = self._reshape_qkv_tensor(qkv_chunks[2], b, f, t)
        
        # Reshape for attention with device checking
        q = q.reshape(b * self.heads, f * t, -1)
        if q.device != self.device:
            q = q.to(self.device)
            
        k = k.reshape(b * self.heads, f * t, -1)
        if k.device != self.device:
            k = k.to(self.device)
            
        v = v.reshape(b * self.heads, f * t, -1)
        if v.device != self.device:
            v = v.to(self.device)
        
        # Chunked attention computation
        out = self._chunk_attention(q, k, v, self.chunk_size)
        
        # Reshape output with device checking
        out = out.reshape(b, self.heads, f, t, -1)
        if out.device != self.device:
            out = out.to(self.device)
            
        out = out.permute(0, 2, 3, 1, 4).reshape(b, f, t, -1)
        if out.device != self.device:
            out = out.to(self.device)
        
        # Output projection and residual connection
        out = self.norm2(out)
        out = self.to_out(out)
        
        # Ensure residual connection has matching devices
        if out.device != x.device:
            if x.device == self.device:
                out = out.to(self.device)
            else:
                x = x.to(self.device)
            
        out = out + x  # Residual connection
        
        # Restore original shape
        out = out.permute(0, 3, 1, 2)  # (b, c, f, t)
        
        # Final device check
        if out.device != self.device:
            out = out.to(self.device)
            
        return out 