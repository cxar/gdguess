"""
Feature extraction networks for the Grateful Dead show dating model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .components import ResidualBlock, FrequencyTimeAttention


class ParallelFeatureNetwork(nn.Module):
    """
    Network that processes multiple audio features in parallel branches.
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature processing branches
        self.harmonic_branch = nn.Sequential(
            ResidualBlock(1, 32),
            ResidualBlock(32, 64),
            FrequencyTimeAttention(64)
        )
        
        self.percussive_branch = nn.Sequential(
            ResidualBlock(1, 32),
            ResidualBlock(32, 64),
            FrequencyTimeAttention(64)
        )
        
        self.chroma_branch = nn.Sequential(
            ResidualBlock(12, 32),
            ResidualBlock(32, 64),
            FrequencyTimeAttention(64)
        )
        
        self.contrast_branch = nn.Sequential(
            ResidualBlock(6, 32),
            ResidualBlock(32, 64),
            FrequencyTimeAttention(64)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 64, 1),
            nn.GELU()
        )
        
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process multiple feature streams and fuse them.
        
        Args:
            features_dict: Dictionary of input features
            
        Returns:
            Fused features tensor
        """
        # Get features with fallbacks for different naming conventions
        harmonic = features_dict.get('harmonic', features_dict.get('mel_spec'))
        percussive = features_dict.get('percussive', features_dict.get('mel_spec_percussive'))
        chroma = features_dict.get('chroma')
        spectral_contrast = features_dict.get('spectral_contrast', features_dict.get('spectral_contrast_harmonic'))
        
        # Reshape chroma to 4D for ResidualBlock processing if needed
        if chroma is not None and chroma.dim() == 3:
            chroma = chroma.unsqueeze(2)  # Add freq dimension
            
        # Reshape spectral_contrast to 4D for ResidualBlock processing if needed
        if spectral_contrast is not None and spectral_contrast.dim() == 3:
            spectral_contrast = spectral_contrast.unsqueeze(2)  # Add freq dimension
        
        # Process each feature stream
        h_out = self.harmonic_branch(harmonic)
        p_out = self.percussive_branch(percussive)
        c_out = self.chroma_branch(chroma)
        sc_out = self.contrast_branch(spectral_contrast)
        
        # Frequency dimension matching for concatenation
        freq_dim = h_out.shape[2]
        
        # Resize c_out and sc_out to match the frequency dimension
        c_out_resized = F.interpolate(
            c_out, 
            size=(freq_dim, c_out.shape[3]),
            mode='nearest'
        )
        
        sc_out_resized = F.interpolate(
            sc_out, 
            size=(freq_dim, sc_out.shape[3]),
            mode='nearest'
        )
        
        # Concatenate features with matched dimensions
        concat = torch.cat([h_out, p_out, c_out_resized, sc_out_resized], dim=1)
        
        # Fuse features
        return self.fusion(concat)


class SeasonalPatternModule(nn.Module):
    """
    Module that models seasonal patterns/periodicity.
    
    Args:
        input_dim: Dimension of input features (default: 1 for a scalar day value)
        output_dim: Dimension of output seasonal features (default: 4)
    """
    
    def __init__(self, input_dim: int = 1, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert day of year to seasonal features.
        
        Args:
            x: Day of year tensor, shape (batch_size,)
            
        Returns:
            Seasonal features, shape (batch_size, output_dim)
        """
        # Reshape to proper dimensions if needed
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Add feature dimension
            
        return self.net(x)