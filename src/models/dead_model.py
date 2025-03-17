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
import torchaudio
import torchaudio.transforms as transforms
import librosa
import numpy as np

from .attention import FrequencyTimeAttention
from .blocks import ResidualBlock, PositionalEncoding
from .dtype_helpers import force_dtype, verify_model_dtype


class ParallelFeatureNetwork(nn.Module):
    """
    Network that processes multiple audio features in parallel branches.
    """
    
    def __init__(self, device='cpu'):
        super().__init__()
        
        self.device = device
        
        # Feature processing branches
        self.harmonic_branch = nn.Sequential(
            ResidualBlock(1, 32, device=device),
            ResidualBlock(32, 64, device=device),
            FrequencyTimeAttention(64, device=device)
        )
        
        self.percussive_branch = nn.Sequential(
            ResidualBlock(1, 32, device=device),
            ResidualBlock(32, 64, device=device),
            FrequencyTimeAttention(64, device=device)
        )
        
        self.chroma_branch = nn.Sequential(
            ResidualBlock(12, 32, device=device),
            ResidualBlock(32, 64, device=device),
            FrequencyTimeAttention(64, device=device)
        )
        
        self.contrast_branch = nn.Sequential(
            ResidualBlock(6, 32, device=device),
            ResidualBlock(32, 64, device=device),
            FrequencyTimeAttention(64, device=device)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 64, 1),
            nn.GELU()
        )
        
        # Move to specified device and ensure float32
        self.to(device)
        force_dtype(self, dtype=torch.float32, device=device)
        
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process multiple feature streams and fuse them.
        
        Args:
            features_dict: Dictionary of input features
            
        Returns:
            Fused features tensor
        """
        # Ensure all inputs are on the correct device and in float32
        for key in features_dict:
            if isinstance(features_dict[key], torch.Tensor):
                features_dict[key] = features_dict[key].float().to(self.device)
        
        # Get features
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
        device: Device to place the module on
    """
    
    def __init__(self, input_dim: int = 1, output_dim: int = 4, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )
        
        # Move to specified device and ensure float32
        self.to(device)
        force_dtype(self, dtype=torch.float32, device=device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert day of year to seasonal features.
        
        Args:
            x: Day of year tensor, shape (batch_size,)
            
        Returns:
            Seasonal features, shape (batch_size, output_dim)
        """
        # Ensure input is float32 and on the correct device
        x = x.float().to(self.device)
            
        # Reshape to proper dimensions if needed
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Add feature dimension
            
        return self.net(x)


class AudioFeatureExtractor(nn.Module):
    """
    Module for extracting audio features from raw waveforms.
    
    Args:
        sample_rate: Audio sample rate in Hz
        n_fft: FFT size
        n_mels: Number of mel bins
        hop_length: Hop length for STFT
        device: Device to place the module on
    """
    
    def __init__(self, sample_rate: int = 24000, n_fft: int = 2048, n_mels: int = 128, 
                 hop_length: int = 512, device: str = 'cpu'):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.device = device
        self.is_mps = str(device).startswith('mps') if isinstance(device, torch.device) else device.startswith('mps')
        
        # Create mel spectrogram transform
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            center=True,
            power=2.0,
        ).to(device)
        
        # Move to specified device and ensure float32
        self.to(device)
        force_dtype(self, dtype=torch.float32, device=device)
    
    def extract_harmonic_percussive(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract harmonic and percussive components from waveform.
        
        Args:
            waveform: Input waveform tensor (batch_size, num_samples)
            
        Returns:
            Tuple of (harmonic_mel_spectrogram, percussive_mel_spectrogram)
            Each with shape (batch_size, 1, n_mels, time)
        """
        # Ensure input is float32
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
            
        # Ensure input is on the correct device
        if waveform.device != torch.device(self.device):
            waveform = waveform.to(self.device)
            
        batch_size = waveform.shape[0]
        results_harmonic = []
        results_percussive = []
        
        # Process in smaller batches if on MPS to avoid memory issues
        batch_size_limit = 8 if self.is_mps else batch_size
        
        for batch_idx in range(0, batch_size, batch_size_limit):
            end_idx = min(batch_idx + batch_size_limit, batch_size)
            batch_x = waveform[batch_idx:end_idx]
            
            # Move data to CPU for librosa processing
            cpu_x = batch_x.cpu() if batch_x.device.type != 'cpu' else batch_x
            
            # Process each audio in the batch
            for i in range(end_idx - batch_idx):
                audio = cpu_x[i].detach().numpy()
                
                try:
                    harmonic, percussive = librosa.effects.hpss(audio)
                    
                    # Convert to torch tensors on CPU first, then move to device
                    harmonic_tensor = torch.from_numpy(harmonic).float()
                    percussive_tensor = torch.from_numpy(percussive).float()
                    
                    # Move to device for mel processing
                    harmonic_tensor = harmonic_tensor.to(self.device)
                    percussive_tensor = percussive_tensor.to(self.device)
                    
                    # Get mel spectrograms
                    harmonic_mel = self.mel_transform(harmonic_tensor)
                    percussive_mel = self.mel_transform(percussive_tensor)
                    
                    # Apply log transform
                    harmonic_mel = torch.log10(harmonic_mel + 1e-5)
                    percussive_mel = torch.log10(percussive_mel + 1e-5)
                    
                    results_harmonic.append(harmonic_mel.unsqueeze(0))
                    results_percussive.append(percussive_mel.unsqueeze(0))
                except Exception as e:
                    # Handle errors (e.g., if the audio is too short)
                    dummy_mel = torch.zeros((1, self.n_mels, 50), dtype=torch.float32, device=self.device)
                    results_harmonic.append(dummy_mel)
                    results_percussive.append(dummy_mel)
            
            # Force immediate garbage collection to free memory on MPS
            if self.is_mps:
                import gc
                gc.collect()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        
        # Stack the batch back together
        harmonic_batch = torch.cat(results_harmonic, dim=0).unsqueeze(1)
        percussive_batch = torch.cat(results_percussive, dim=0).unsqueeze(1)
        
        return harmonic_batch, percussive_batch
    
    def extract_chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract chroma features from waveform.
        
        Args:
            waveform: Input waveform tensor (batch_size, num_samples)
            
        Returns:
            Chroma features tensor (batch_size, 12, time)
        """
        # Ensure input is float32
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
            
        # Ensure input is on the correct device
        if waveform.device != torch.device(self.device):
            waveform = waveform.to(self.device)
            
        batch_size = waveform.shape[0]
        results = []
        
        # Process in smaller batches if on MPS to avoid memory issues
        batch_size_limit = 8 if self.is_mps else batch_size
        
        for batch_idx in range(0, batch_size, batch_size_limit):
            end_idx = min(batch_idx + batch_size_limit, batch_size)
            batch_x = waveform[batch_idx:end_idx]
            
            # Move data to CPU for librosa processing
            cpu_x = batch_x.cpu() if batch_x.device.type != 'cpu' else batch_x
            
            # Process each audio in the batch
            for i in range(end_idx - batch_idx):
                audio = cpu_x[i].detach().numpy()
                
                try:
                    # Extract chroma
                    chroma = librosa.feature.chroma_stft(
                        y=audio, 
                        sr=self.sample_rate,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length
                    )
                    
                    # Convert to torch tensor on device
                    chroma_tensor = torch.from_numpy(chroma).float().to(self.device)
                    results.append(chroma_tensor.unsqueeze(0))
                except Exception as e:
                    dummy_chroma = torch.zeros((1, 12, 50), dtype=torch.float32, device=self.device)
                    results.append(dummy_chroma)
            
            # Force immediate garbage collection to free memory on MPS
            if self.is_mps:
                import gc
                gc.collect()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        
        # Stack the batch
        chroma_batch = torch.cat(results, dim=0)
        
        return chroma_batch
    
    def extract_spectral_contrast(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral contrast features from waveform.
        
        Args:
            waveform: Input waveform tensor (batch_size, num_samples)
            
        Returns:
            Spectral contrast features tensor (batch_size, 6, time)
        """
        # Ensure input is float32
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
            
        # Ensure input is on the correct device
        if waveform.device != torch.device(self.device):
            waveform = waveform.to(self.device)
            
        batch_size = waveform.shape[0]
        results = []
        
        # Process in smaller batches if on MPS to avoid memory issues
        batch_size_limit = 8 if self.is_mps else batch_size
        
        for batch_idx in range(0, batch_size, batch_size_limit):
            end_idx = min(batch_idx + batch_size_limit, batch_size)
            batch_x = waveform[batch_idx:end_idx]
            
            # Move data to CPU for librosa processing
            cpu_x = batch_x.cpu() if batch_x.device.type != 'cpu' else batch_x
            
            # Process each audio in the batch
            for i in range(end_idx - batch_idx):
                audio = cpu_x[i].detach().numpy()
                
                try:
                    # Extract spectral contrast
                    contrast = librosa.feature.spectral_contrast(
                        y=audio,
                        sr=self.sample_rate,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        n_bands=6
                    )
                    
                    # Convert to torch tensor on device
                    contrast_tensor = torch.from_numpy(contrast).float().to(self.device)
                    results.append(contrast_tensor.unsqueeze(0))
                except Exception as e:
                    dummy_contrast = torch.zeros((1, 6, 50), dtype=torch.float32, device=self.device)
                    results.append(dummy_contrast)
            
            # Force immediate garbage collection to free memory on MPS
            if self.is_mps:
                import gc
                gc.collect()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        
        # Stack the batch
        contrast_batch = torch.cat(results, dim=0)
        
        return contrast_batch

    def _clear_mps_memory(self):
        """
        Explicitly clear MPS memory cache.
        This helps prevent memory buildup on Apple Silicon GPUs.
        """
        if self.is_mps:
            import gc
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    def _extract_audio_features(self, x: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Extract audio features from raw waveform."""
        features = {}
        
        if x.dim() > 0 and x.numel() > 0:
            try:
                # Use the optimized methods from AudioFeatureExtractor
                harmonic, percussive = self.feature_extractor.extract_harmonic_percussive(x)
                features['harmonic'] = harmonic
                features['percussive'] = percussive
                
                # Clear MPS memory cache if needed
                self._clear_mps_memory()
                
                features['chroma'] = self.feature_extractor.extract_chroma(x)
                
                # Clear MPS memory cache if needed
                self._clear_mps_memory()
                
                features['spectral_contrast'] = self.feature_extractor.extract_spectral_contrast(x)
                
                # Clear MPS memory cache if needed
                self._clear_mps_memory()
                
            except Exception as e:
                # Fall back to zeros if processing fails
                features = {
                    'harmonic': torch.zeros((batch_size, 1, 128, 50), dtype=torch.float32, device=self.device),
                    'percussive': torch.zeros((batch_size, 1, 128, 50), dtype=torch.float32, device=self.device),
                    'chroma': torch.zeros((batch_size, 12, 50), dtype=torch.float32, device=self.device),
                    'spectral_contrast': torch.zeros((batch_size, 6, 50), dtype=torch.float32, device=self.device)
                }
        else:
            # Create zero tensors for empty input
            features = {
                'harmonic': torch.zeros((batch_size, 1, 128, 50), dtype=torch.float32, device=self.device),
                'percussive': torch.zeros((batch_size, 1, 128, 50), dtype=torch.float32, device=self.device),
                'chroma': torch.zeros((batch_size, 12, 50), dtype=torch.float32, device=self.device),
                'spectral_contrast': torch.zeros((batch_size, 6, 50), dtype=torch.float32, device=self.device)
            }
            
        return features


class DeadShowDatingModel(nn.Module):
    """
    Main model for dating Grateful Dead shows from audio.
    
    Args:
        sample_rate: Audio sample rate in Hz
        device: Device to place the model on ('cpu', 'cuda', 'cuda:0', etc.)
    """
    
    def __init__(self, sample_rate: int = 24000, device: str = 'cpu'):
        super().__init__()
        
        # Store device for consistent use throughout the model
        self.device = device
        self.is_mps = str(device).startswith('mps') if isinstance(device, torch.device) else device.startswith('mps')
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor(sample_rate, device=device)
        self.feature_network = ParallelFeatureNetwork(device=device)
        self.seasonal_pattern = SeasonalPatternModule(device=device)
        
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
        
        # Move to specified device and ensure float32
        self.to(device)
        force_dtype(self, dtype=torch.float32, device=device)
        
        # Verify all parameters are float32 after initialization
        all_float32, num_mismatched, _, _ = verify_model_dtype(self, torch.float32)
        if not all_float32:
            raise RuntimeError(f"{num_mismatched} parameters not in float32 after initialization")
        
    def _clear_mps_memory(self):
        """
        Explicitly clear MPS memory cache.
        This helps prevent memory buildup on Apple Silicon GPUs.
        """
        if self.is_mps:
            import gc
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
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
            - 'date': (batch_size, 1) date from input or default
            - 'year_logits': (batch_size, 30) placeholder for year logits (1965-1995)
        """
        # Create standardized feature dictionary
        features = self._prepare_features(x)
        batch_size = next(iter(features.values())).shape[0]
        
        # Process audio features through the feature network
        audio_features = self.feature_network(features)
        audio_features = self.global_pool(audio_features).squeeze(-1).squeeze(-1)
        
        # Clear MPS memory cache if needed
        self._clear_mps_memory()
        
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
            'era_logits': torch.zeros((audio_features.shape[0], 7), dtype=torch.float32, device=self.device),
            'audio_features': audio_features,
            'seasonal_features': seasonal,
            'date': features.get('date', torch.zeros(audio_features.shape[0], dtype=torch.float32, device=self.device)),
            'year_logits': year_logits
        }
        
        # Clear MPS memory cache if needed
        self._clear_mps_memory()
                
        return result
        
    def _prepare_features(self, x: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, torch.Tensor]:
        """Process input to get standardized feature dictionary."""
        # Move input to device and convert to float32
        if isinstance(x, dict):
            # Handle dictionary input
            x_processed = {}
            for key, value in x.items():
                if isinstance(value, torch.Tensor):
                    # Only convert if necessary
                    if value.device != self.device or value.dtype != torch.float32:
                        x_processed[key] = value.float().to(self.device)
                    else:
                        x_processed[key] = value
                elif isinstance(value, (list, int, float)):
                    try:
                        x_processed[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
                    except (ValueError, TypeError):
                        x_processed[key] = value
                else:
                    x_processed[key] = value
            x = x_processed
        elif isinstance(x, torch.Tensor):
            # Only convert if necessary
            if x.device != self.device or x.dtype != torch.float32:
                x = x.float().to(self.device)
        
        # Process raw audio or use provided features
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            features = self._extract_audio_features(x, batch_size)
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
        
        # Copy and normalize feature names, ensuring all are float32 on the correct device
        if 'mel_spec' in x and 'harmonic' not in x:
            features['harmonic'] = self._ensure_tensor(x['mel_spec'])
        elif 'harmonic' in x:
            features['harmonic'] = self._ensure_tensor(x['harmonic'])
        
        if 'mel_spec_percussive' in x and 'percussive' not in x:
            features['percussive'] = self._ensure_tensor(x['mel_spec_percussive'])
        elif 'percussive' in x:
            features['percussive'] = self._ensure_tensor(x['percussive'])
        
        if 'spectral_contrast_harmonic' in x and 'spectral_contrast' not in x:
            features['spectral_contrast'] = self._ensure_tensor(x['spectral_contrast_harmonic'])
        elif 'spectral_contrast' in x:
            features['spectral_contrast'] = self._ensure_tensor(x['spectral_contrast'])
        
        if 'chroma' in x:
            features['chroma'] = self._ensure_tensor(x['chroma'])
        
        # Handle date/year information properly
        self._process_metadata(x, features, batch_size)
        
        return features
    
    def _ensure_tensor(self, value):
        """Ensure value is a tensor with correct device and dtype."""
        if isinstance(value, torch.Tensor):
            # Only convert if needed to avoid unnecessary tensor creation
            if value.device != self.device or value.dtype != torch.float32:
                return value.float().to(self.device)
            return value
        return value
    
    def _process_metadata(self, x: Dict[str, Union[torch.Tensor, list, str]], features: Dict[str, torch.Tensor], batch_size: int):
        """Process metadata fields like date, year, etc."""
        # Handle date
        if 'date' in x:
            features['date'] = self._convert_to_tensor(x['date'], batch_size)
        
        # Handle year and extract from filename if needed
        if 'year' in x:
            features['year'] = self._convert_to_tensor(x['year'], batch_size)
        elif 'file' in x or 'path' in x:
            features['year'] = self._extract_year_from_filename(x, batch_size)
        
        # Copy any other needed fields
        for k in ['label', 'era', 'path', 'file']:
            if k in x:
                if isinstance(x[k], torch.Tensor):
                    features[k] = x[k].float().to(self.device)
                else:
                    features[k] = x[k]
    
    def _convert_to_tensor(self, value, batch_size):
        """Convert various input types to tensor."""
        if isinstance(value, torch.Tensor):
            # Only convert if necessary
            if value.device != self.device or value.dtype != torch.float32:
                return value.float().to(self.device)
            return value
        elif isinstance(value, (int, float)):
            return torch.tensor([value], dtype=torch.float32, device=self.device)
        elif isinstance(value, list):
            try:
                return torch.tensor(value, dtype=torch.float32, device=self.device)
            except (ValueError, TypeError):
                return torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        return torch.zeros(batch_size, dtype=torch.float32, device=self.device)
    
    def _extract_year_from_filename(self, x, batch_size):
        """Extract year from filename if available."""
        file_val = x.get('file', x.get('path', ''))
        
        if isinstance(file_val, list) and len(file_val) > 0:
            filename = file_val[0] if file_val else ''
            match = re.search(r'(\d{4})-\d{2}-\d{2}', os.path.basename(str(filename)))
            if match:
                year = int(match.group(1))
                return torch.tensor([year], dtype=torch.float32, device=self.device)
        elif isinstance(file_val, str):
            match = re.search(r'(\d{4})-\d{2}-\d{2}', os.path.basename(file_val))
            if match:
                year = int(match.group(1))
                return torch.tensor([year], dtype=torch.float32, device=self.device)
                
        return torch.zeros(batch_size, dtype=torch.float32, device=self.device)
    
    def _ensure_required_features(self, features: Dict[str, torch.Tensor], batch_size: int):
        """Fill in any missing required features with zeros."""
        required_features = {
            'harmonic': (batch_size, 1, 128, 50),
            'percussive': (batch_size, 1, 128, 50),
            'chroma': (batch_size, 12, 50),
            'spectral_contrast': (batch_size, 6, 50)
        }
        
        for feat, shape in required_features.items():
            if feat not in features:
                features[feat] = torch.zeros(shape, dtype=torch.float32, device=self.device)
            elif isinstance(features[feat], torch.Tensor):
                # Only convert if necessary to avoid wasting memory
                if features[feat].device != self.device or features[feat].dtype != torch.float32:
                    features[feat] = features[feat].float().to(self.device)
    
    def _get_day_of_year(self, features: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Get day of year from features or default to zeros."""
        if features.get('date') is not None and isinstance(features['date'], torch.Tensor):
            return torch.remainder(features['date'], 365.0).float()
        return torch.zeros(batch_size, dtype=torch.float32, device=self.device)
    
    def _create_year_logits(self, features: Dict[str, torch.Tensor], batch_size: int, audio_batch_size: int) -> torch.Tensor:
        """Create year logits based on year if available."""
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
            year_logits = torch.zeros((audio_batch_size, num_years), dtype=torch.float32, device=self.device)
            
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
        return torch.zeros((audio_batch_size, 30), dtype=torch.float32, device=self.device)
