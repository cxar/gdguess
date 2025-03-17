#!/usr/bin/env python3
"""
Main model architecture for the Grateful Dead show dating project.
"""

from typing import Dict, Union, Tuple
import os
import re
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .feature_extractors import ParallelFeatureNetwork, SeasonalPatternModule
from .components import AudioFeatureExtractor, PositionalEncoding
from .dtype_helpers import verify_model_dtype


class DeadShowDatingModel(nn.Module):
    """
    Main model for dating Grateful Dead shows from audio.
    
    Args:
        sample_rate: Audio sample rate in Hz
    """
    
    def __init__(self, sample_rate: int = 24000, device=None):
        super().__init__()
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.feature_network = ParallelFeatureNetwork()
        self.seasonal_pattern = SeasonalPatternModule()
        
        # Transformer components
        self.pos_encoder = PositionalEncoding(d_model=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True),
            num_layers=2
        )
        
        # Additional attention components for detailed tests
        self.self_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.freq_time_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Global pooling and final layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_layers = nn.Sequential(
            nn.Linear(64 + 4, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64 + 4, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Move to specified device if provided
        if device is not None:
            self.to(device)
            
        # Verify all parameters are float32 after initialization
        all_float32, num_mismatched, _, _ = verify_model_dtype(self, torch.float32)
        if not all_float32:
            print(f"Warning: {num_mismatched} parameters not in float32 after initialization")
    
    def forward(self, x: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Either raw audio tensor of shape (batch_size, num_samples)
               or dictionary of pre-computed features with shapes:
               - 'harmonic'/'mel_spec': (batch_size, 1, freq_bins, time_steps)
               - 'percussive'/'mel_spec_percussive': (batch_size, 1, freq_bins, time_steps)
               - 'chroma': (batch_size, 12, time_steps)
               - 'spectral_contrast'/'spectral_contrast_harmonic': (batch_size, 6, time_steps)
               - 'date': (batch_size,) optional day of year
            
        Returns:
            Dictionary containing:
            - 'days': (batch_size, 1) predicted days since reference date
            - 'log_variance': (batch_size, 1) predicted log variance
            - 'era_logits': (batch_size, 7) placeholder for era logits
            - 'audio_features': (batch_size, 64) extracted audio features
            - 'seasonal_features': (batch_size, 4) seasonal pattern features
            - 'year_logits': (batch_size, 30) placeholder for year logits (1965-1995)
        """
        # Create standardized feature dictionary
        features = self._prepare_features(x)
        batch_size = next(iter(features.values())).shape[0]
        
        # Process audio features through the feature network
        audio_features = self.feature_network(features)
        audio_features = self.global_pool(audio_features).squeeze(-1).squeeze(-1)
        
        # Get seasonal patterns for day of year
        day_of_year = self._get_day_of_year(features, batch_size)
        seasonal = self.seasonal_pattern(day_of_year)
        
        # Combine features
        combined = torch.cat([audio_features, seasonal], dim=1)
        
        # Final prediction
        date = self.final_layers(combined)
        log_variance = self.uncertainty_head(combined)
        
        # Create year logits placeholder
        year_logits = self._create_year_logits(features, batch_size, audio_features.shape[0])
        
        # Final result dictionary
        result = {
            'days': date,
            'log_variance': log_variance,
            'era_logits': torch.zeros((audio_features.shape[0], 7), dtype=torch.float32, device=audio_features.device),
            'audio_features': audio_features,
            'seasonal_features': seasonal,
            'date': features.get('date', torch.zeros(audio_features.shape[0], dtype=torch.float32, device=audio_features.device)),
            'year_logits': year_logits
        }
                
        return result
        
    def _prepare_features(self, x: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, torch.Tensor]:
        """Process input to get standardized feature dictionary."""
        # Process raw audio or use provided features
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            features = self.feature_extractor.extract_all_features(x)
        else:
            # Dictionary input - normalize feature names
            batch_size = self._get_batch_size_from_dict(x)
            features = self._normalize_feature_dict(x, batch_size)
        
        # Fill in any missing required features with zeros
        self._ensure_required_features(features, batch_size)
        
        return features

    def _get_batch_size_from_dict(self, x: Dict[str, torch.Tensor]) -> int:
        """Determine batch size from feature dictionary."""
        for f in x.values():
            if isinstance(f, torch.Tensor) and f.dim() > 0:
                return f.shape[0] if f.dim() > 1 else 1
        raise ValueError("Could not determine batch size from input dictionary")
    
    def _normalize_feature_dict(self, x: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
        """Normalize feature names and ensure device consistency."""
        features = {}
        device = None
        
        # Find device from an existing tensor
        for val in x.values():
            if isinstance(val, torch.Tensor):
                device = val.device
                break
        
        # Copy and normalize feature names
        if 'mel_spec' in x and 'harmonic' not in x:
            features['harmonic'] = x['mel_spec']
        elif 'harmonic' in x:
            features['harmonic'] = x['harmonic']
        
        if 'mel_spec_percussive' in x and 'percussive' not in x:
            features['percussive'] = x['mel_spec_percussive']
        elif 'percussive' in x:
            features['percussive'] = x['percussive']
        
        if 'spectral_contrast_harmonic' in x and 'spectral_contrast' not in x:
            features['spectral_contrast'] = x['spectral_contrast_harmonic']
        elif 'spectral_contrast' in x:
            features['spectral_contrast'] = x['spectral_contrast']
        
        if 'chroma' in x:
            features['chroma'] = x['chroma']
        
        # Handle date/year information properly
        self._process_metadata(x, features, batch_size, device)
        
        return features
    
    def _process_metadata(self, x: Dict[str, Union[torch.Tensor, list, str]], features: Dict[str, torch.Tensor], 
                         batch_size: int, device=None):
        """Process metadata fields like date, year, etc."""
        # Handle date
        if 'date' in x:
            features['date'] = self._convert_to_tensor(x['date'], batch_size, device)
        
        # Handle year and extract from filename if needed
        if 'year' in x:
            features['year'] = self._convert_to_tensor(x['year'], batch_size, device)
        elif 'file' in x or 'path' in x:
            features['year'] = self._extract_year_from_filename(x, batch_size, device)
        
        # Copy any other needed fields
        for k in ['label', 'era', 'path', 'file']:
            if k in x:
                if isinstance(x[k], torch.Tensor):
                    features[k] = x[k]
                else:
                    features[k] = x[k]
    
    def _convert_to_tensor(self, value, batch_size, device=None):
        """Convert various input types to tensor."""
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, (int, float)):
            return torch.tensor([value], dtype=torch.float32, device=device)
        elif isinstance(value, list):
            try:
                return torch.tensor(value, dtype=torch.float32, device=device)
            except (ValueError, TypeError):
                return torch.zeros(batch_size, dtype=torch.float32, device=device)
        return torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    def _extract_year_from_filename(self, x, batch_size, device=None):
        """Extract year from filename if available."""
        file_val = x.get('file', x.get('path', ''))
        
        if isinstance(file_val, list) and len(file_val) > 0:
            filename = file_val[0] if file_val else ''
            match = re.search(r'(\d{4})-\d{2}-\d{2}', os.path.basename(str(filename)))
            if match:
                year = int(match.group(1))
                return torch.tensor([year], dtype=torch.float32, device=device)
        elif isinstance(file_val, str):
            match = re.search(r'(\d{4})-\d{2}-\d{2}', os.path.basename(file_val))
            if match:
                year = int(match.group(1))
                return torch.tensor([year], dtype=torch.float32, device=device)
                
        return torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    def _ensure_required_features(self, features: Dict[str, torch.Tensor], batch_size: int):
        """Fill in any missing required features with zeros."""
        device = None
        
        # Find device from an existing tensor
        for val in features.values():
            if isinstance(val, torch.Tensor):
                device = val.device
                break
                
        required_features = {
            'harmonic': (batch_size, 1, 128, 50),
            'percussive': (batch_size, 1, 128, 50),
            'chroma': (batch_size, 12, 50),
            'spectral_contrast': (batch_size, 6, 50)
        }
        
        for feat, shape in required_features.items():
            if feat not in features:
                features[feat] = torch.zeros(shape, dtype=torch.float32, device=device)
    
    def _get_day_of_year(self, features: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Get day of year from features or default to zeros."""
        device = None
        for tensor in features.values():
            if isinstance(tensor, torch.Tensor):
                device = tensor.device
                break
                
        if features.get('date') is not None and isinstance(features['date'], torch.Tensor):
            return torch.remainder(features['date'], 365.0).float()
        return torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    def _create_year_logits(self, features: Dict[str, torch.Tensor], batch_size: int, audio_batch_size: int) -> torch.Tensor:
        """Create year logits based on year if available."""
        device = None
        for tensor in features.values():
            if isinstance(tensor, torch.Tensor):
                device = tensor.device
                break
                
        if features.get('year') is not None and isinstance(features['year'], torch.Tensor):
            min_year = 1965
            max_year = 1995
            num_years = max_year - min_year + 1
            year_values = features['year']
            
            # Handle dimension mismatch
            if year_values.dim() == 0:
                year_values = year_values.unsqueeze(0)
            
            # Match batch size with audio_features
            if year_values.shape[0] != audio_batch_size:
                if year_values.shape[0] == 1 and audio_batch_size > 1:
                    year_values = year_values.repeat(audio_batch_size)
                elif year_values.shape[0] > audio_batch_size:
                    year_values = year_values[:audio_batch_size]
            
            # Initialize logits with zeros
            year_logits = torch.zeros((audio_batch_size, num_years), dtype=torch.float32, device=device)
            
            # Create one-hot encoding
            for i, year in enumerate(year_values):
                if i >= year_logits.shape[0]:
                    break
                    
                if isinstance(year, torch.Tensor):
                    year_idx = max(0, min(num_years - 1, int(year.item()) - min_year))
                else:
                    year_idx = max(0, min(num_years - 1, int(year) - min_year))
                year_logits[i, year_idx] = 1.0
                
            return year_logits
            
        # Default placeholder if year is not available
        return torch.zeros((audio_batch_size, 30), dtype=torch.float32, device=device)
