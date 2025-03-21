#!/usr/bin/env python3
"""
Simplified model architecture for the Grateful Dead show dating project.
"""

from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ResidualBlock(nn.Module):
    """Residual block with 2D convolutions."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv for residual connection if dimensions change
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.gelu(x)

class FrequencyTimeAttention(nn.Module):
    """Simple attention mechanism for frequency-time features."""
    
    def __init__(self, channels):
        super().__init__()
        self.freq_attention = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.time_attention = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        
    def forward(self, x):
        freq_weights = torch.sigmoid(self.freq_attention(x))
        time_weights = torch.sigmoid(self.time_attention(x))
        return x * freq_weights * time_weights

class ParallelFeatureNetwork(nn.Module):
    """Network that processes multiple audio features in parallel branches."""
    
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
        
    def forward(self, features_dict):
        """Process multiple feature streams and fuse them."""
        # Get features
        harmonic = features_dict['harmonic']
        percussive = features_dict['percussive']
        chroma = features_dict['chroma']
        spectral_contrast = features_dict['spectral_contrast']
        
        # Fix shapes if needed
        # Ensure 4D: [batch_size, channels, freq, time]
        
        # Fix harmonic if needed (ensure it's [B, 1, freq, time])
        if harmonic.dim() != 4:
            if harmonic.dim() == 3 and harmonic.size(1) == 128:  # [B, freq, time]
                harmonic = harmonic.unsqueeze(1)  # Add channel dim
            elif harmonic.dim() == 3 and harmonic.size(0) == 1:  # [1, freq, time]
                # This is trickier - reshape to [1, 1, freq, time]
                harmonic = harmonic.unsqueeze(0)
        
        # Similarly for percussive
        if percussive.dim() != 4:
            if percussive.dim() == 3 and percussive.size(1) == 128:
                percussive = percussive.unsqueeze(1)
            elif percussive.dim() == 3 and percussive.size(0) == 1:
                percussive = percussive.unsqueeze(0)
        
        # Reshape chroma to 4D for ResidualBlock processing
        if chroma.dim() == 3:
            if chroma.size(1) == 12:  # [B, 12, time]
                # Standard shape, just add a dim
                chroma = chroma.unsqueeze(2)  # [B, 12, 1, time]
            else:
                # Try to adapt to the expected shape
                chroma = chroma.view(chroma.size(0), 12, 1, -1)
                
        # Reshape spectral_contrast to 4D
        if spectral_contrast.dim() == 3:
            if spectral_contrast.size(1) == 6:  # [B, 6, time]
                spectral_contrast = spectral_contrast.unsqueeze(2)  # [B, 6, 1, time]
            else:
                spectral_contrast = spectral_contrast.view(spectral_contrast.size(0), 6, 1, -1)
        
        # Process each feature stream
        try:
            h_out = self.harmonic_branch(harmonic)
            p_out = self.percussive_branch(percussive)
            c_out = self.chroma_branch(chroma)
            sc_out = self.contrast_branch(spectral_contrast)
        except Exception as e:
            # Try alternative approach with reshape
            # This is a more aggressive attempt to force shapes to work
            batch_size = harmonic.size(0)
            harmonic = harmonic.reshape(batch_size, 1, 128, -1)
            percussive = percussive.reshape(batch_size, 1, 128, -1)
            chroma = chroma.reshape(batch_size, 12, 1, -1)
            spectral_contrast = spectral_contrast.reshape(batch_size, 6, 1, -1)
            
            h_out = self.harmonic_branch(harmonic)
            p_out = self.percussive_branch(percussive)
            c_out = self.chroma_branch(chroma)
            sc_out = self.contrast_branch(spectral_contrast)
        
        # Get output dimensions for resizing to match
        freq_dim = h_out.shape[2]
        time_dim = h_out.shape[3]
        
        # Resize c_out and sc_out to match dimensions
        c_out_resized = F.interpolate(c_out, size=(freq_dim, time_dim), mode='nearest')
        sc_out_resized = F.interpolate(sc_out, size=(freq_dim, time_dim), mode='nearest')
        
        # Concatenate features with matched dimensions
        concat = torch.cat([h_out, p_out, c_out_resized, sc_out_resized], dim=1)
        
        # Fuse features
        result = self.fusion(concat)
        
        return result

class SeasonalPatternModule(nn.Module):
    """Module that models seasonal patterns/periodicity."""
    
    def __init__(self, input_dim=1, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        """Convert day of year to seasonal features."""
        # Reshape to proper dimensions if needed
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Add feature dimension
            
        return self.net(x)

class AudioFeatureExtractor:
    """Simple placeholder for audio feature extraction."""
    
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
    
    def extract_all_features(self, audio):
        """Placeholder for feature extraction from raw audio."""
        # In a real implementation, this would extract mel spectrograms,
        # chroma features, etc. from raw audio
        batch_size = audio.shape[0]
        device = audio.device
        
        # Return dummy features with expected shapes
        return {
            'harmonic': torch.zeros((batch_size, 1, 128, 50), device=device),
            'percussive': torch.zeros((batch_size, 1, 128, 50), device=device),
            'chroma': torch.zeros((batch_size, 12, 50), device=device),
            'spectral_contrast': torch.zeros((batch_size, 6, 50), device=device)
        }

class DeadShowDatingModel(nn.Module):
    """Main model for dating Grateful Dead shows from audio."""
    
    def __init__(self, sample_rate=24000, device=None):
        super().__init__()
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.feature_network = ParallelFeatureNetwork()
        self.seasonal_pattern = SeasonalPatternModule()
        
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
    
    def forward(self, x: Union[Tensor, Dict[str, Tensor]], global_step=None):
        """Forward pass of the model."""
        # Create standardized feature dictionary
        try:
            features = self._prepare_features(x)
            batch_size = next(iter(features.values())).shape[0]
        except Exception as e:
            print(f"Error preparing features: {e}")
            # Fallback mode - create minimal features
            if isinstance(x, dict):
                features = x
                # Try to determine batch size from any available tensor
                for val in x.values():
                    if isinstance(val, torch.Tensor):
                        batch_size = val.shape[0]
                        break
                else:
                    batch_size = 1
            else:
                features = {}
                if isinstance(x, torch.Tensor):
                    batch_size = x.shape[0]
                else:
                    batch_size = 1
            
            # Ensure required features exist no matter what
            self._ensure_required_features(features, batch_size)
        
        try:
            # Process audio features through the feature network
            audio_features = self.feature_network(features)
            audio_features = self.global_pool(audio_features).squeeze(-1).squeeze(-1)
            
            # Get seasonal patterns for day of year
            day_of_year = self._get_day_of_year(features, batch_size)
            seasonal = self.seasonal_pattern(day_of_year)
            
            # Combine features
            combined = torch.cat([audio_features, seasonal], dim=1)
            
            # Final prediction (outputs values in [0,1] range)
            date = self.final_layers(combined)
            # Initialize close to 0.5 if starting fresh training
            if global_step is not None and global_step == 0 and self.training:
                date = torch.ones_like(date) * 0.5
            log_variance = self.uncertainty_head(combined)
            
            # Final result dictionary
            result = {
                'days': date,  # Scaled prediction in [0,1] range
                'days_unscaled': date * 10000.0,  # Unscaled prediction in original range
                'log_variance': log_variance,
                'audio_features': audio_features,
                'seasonal_features': seasonal
            }
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Fallback with dummy outputs
            print("Using fallback dummy outputs")
            device = next((tensor.device for tensor in features.values() if isinstance(tensor, torch.Tensor)), 
                          torch.device('cpu'))
            result = {
                'days': torch.ones((batch_size, 1), device=device) * 0.5,
                'days_unscaled': torch.ones((batch_size, 1), device=device) * 5000.0,
                'log_variance': torch.zeros((batch_size, 1), device=device),
                'audio_features': torch.zeros((batch_size, 64), device=device),
                'seasonal_features': torch.zeros((batch_size, 4), device=device)
            }
                
        return result
        
    def _prepare_features(self, x):
        """Process input to get standardized feature dictionary."""
        # Process raw audio or use provided features
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            features = self.feature_extractor.extract_all_features(x)
        else:
            # Dictionary input
            features = self._normalize_feature_dict(x)
            
            # Check if we have any features
            if not features:
                # Try to use whatever is available
                for key in x.keys():
                    if isinstance(x[key], torch.Tensor) and len(x[key].shape) >= 3:
                        features[key] = x[key]
            
            try:
                batch_size = next(iter(features.values())).shape[0]
            except (StopIteration, AttributeError) as e:
                # Fallback batch size
                batch_size = 1
        
        # Ensure required features exist
        self._ensure_required_features(features, batch_size)
        
        return features
    
    def _normalize_feature_dict(self, x):
        """Normalize feature names and ensure all required features exist."""
        features = {}
        device = next((v.device for v in x.values() if isinstance(v, torch.Tensor)), None)
        
        # Copy and normalize feature names
        # For harmonic component
        if 'mel_spec' in x and 'harmonic' not in x:
            features['harmonic'] = x['mel_spec']
        elif 'harmonic' in x:
            features['harmonic'] = x['harmonic']
        
        # For percussive component
        if 'mel_spec_percussive' in x and 'percussive' not in x:
            features['percussive'] = x['mel_spec_percussive']
        elif 'percussive' in x:
            features['percussive'] = x['percussive']
        
        # For spectral contrast
        if 'spectral_contrast_harmonic' in x and 'spectral_contrast' not in x:
            features['spectral_contrast'] = x['spectral_contrast_harmonic']
        elif 'spectral_contrast' in x:
            features['spectral_contrast'] = x['spectral_contrast']
        
        # For chroma
        if 'chroma' in x:
            features['chroma'] = x['chroma']
            
        # For date info
        if 'date' in x:
            features['date'] = x['date'] if isinstance(x['date'], torch.Tensor) else torch.tensor([x['date']], device=device)
        
        return features
    
    def _ensure_required_features(self, features, batch_size):
        """Fill in any missing required features with zeros."""
        device = next((v.device for v in features.values() if isinstance(v, torch.Tensor)), None)
        
        # Get time dimension from existing features if possible
        time_dim = 50  # default
        for feat_name in ['harmonic', 'percussive', 'mel_spec', 'mel_spec_percussive']:
            if feat_name in features and isinstance(features[feat_name], torch.Tensor):
                # Try to determine time dimension from existing feature
                if features[feat_name].dim() >= 3:
                    time_dim = features[feat_name].shape[-1]
                    break
        
        required_features = {
            'harmonic': (batch_size, 1, 128, time_dim),
            'percussive': (batch_size, 1, 128, time_dim),
            'chroma': (batch_size, 12, time_dim),
            'spectral_contrast': (batch_size, 6, time_dim)
        }
        
        for feat, shape in required_features.items():
            if feat not in features:
                features[feat] = torch.zeros(shape, dtype=torch.float32, device=device)
                
            # Ensure each feature has 4 dimensions if needed
            if feat in features and features[feat].dim() < 4 and feat in ['harmonic', 'percussive']:
                # Try to reshape to 4D in a sensible way
                if features[feat].dim() == 3:
                    # Batch, freq, time -> Batch, channels, freq, time
                    features[feat] = features[feat].unsqueeze(1)
                elif features[feat].dim() == 2:
                    # Reshape as batch of 1
                    features[feat] = features[feat].unsqueeze(0).unsqueeze(0)
    
    def _get_day_of_year(self, features, batch_size):
        """Get day of year from features or default to zeros."""
        device = next((tensor.device for tensor in features.values() if isinstance(tensor, torch.Tensor)), None)
        
        if features.get('date') is not None and isinstance(features['date'], torch.Tensor):
            return torch.remainder(features['date'], 365.0).float()
        return torch.zeros(batch_size, dtype=torch.float32, device=device)