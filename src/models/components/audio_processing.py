"""
Audio feature extraction components for the Grateful Dead show dating model.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import librosa
import numpy as np
from typing import Tuple, Dict, Union


class AudioFeatureExtractor(nn.Module):
    """
    Module for extracting audio features from raw waveforms.
    
    Args:
        sample_rate: Audio sample rate in Hz
        n_fft: FFT size
        n_mels: Number of mel bins
        hop_length: Hop length for STFT
    """
    
    def __init__(self, sample_rate: int = 24000, n_fft: int = 2048, n_mels: int = 128, 
                 hop_length: int = 512):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        
        # Create mel spectrogram transform
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            center=True,
            power=2.0,
        )
    
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
        waveform = waveform.float()
            
        batch_size = waveform.shape[0]
        results_harmonic = []
        results_percussive = []
        
        # Process each audio in the batch
        for i in range(batch_size):
            # Move data to CPU for librosa processing
            cpu_x = waveform[i].cpu().detach().numpy()
            
            try:
                harmonic, percussive = librosa.effects.hpss(cpu_x)
                
                # Convert to torch tensors
                harmonic_tensor = torch.from_numpy(harmonic).float()
                percussive_tensor = torch.from_numpy(percussive).float()
                
                # Get mel spectrograms
                harmonic_mel = self.mel_transform(harmonic_tensor)
                percussive_mel = self.mel_transform(percussive_tensor)
                
                # Apply log transform
                harmonic_mel = torch.log10(harmonic_mel + 1e-5)
                percussive_mel = torch.log10(percussive_mel + 1e-5)
                
                results_harmonic.append(harmonic_mel.unsqueeze(0))
                results_percussive.append(percussive_mel.unsqueeze(0))
            except Exception:
                # Handle errors (e.g., if the audio is too short)
                dummy_mel = torch.zeros((1, self.n_mels, 50), dtype=torch.float32)
                results_harmonic.append(dummy_mel)
                results_percussive.append(dummy_mel)
        
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
        waveform = waveform.float()
            
        batch_size = waveform.shape[0]
        results = []
        
        # Process each audio in the batch
        for i in range(batch_size):
            # Move data to CPU for librosa processing
            cpu_x = waveform[i].cpu().detach().numpy()
            
            try:
                # Extract chroma
                chroma = librosa.feature.chroma_stft(
                    y=cpu_x, 
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                
                # Convert to torch tensor
                chroma_tensor = torch.from_numpy(chroma).float()
                results.append(chroma_tensor.unsqueeze(0))
            except Exception:
                dummy_chroma = torch.zeros((1, 12, 50), dtype=torch.float32)
                results.append(dummy_chroma)
        
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
        waveform = waveform.float()
            
        batch_size = waveform.shape[0]
        results = []
        
        # Process each audio in the batch
        for i in range(batch_size):
            # Move data to CPU for librosa processing
            cpu_x = waveform[i].cpu().detach().numpy()
            
            try:
                # Extract spectral contrast
                contrast = librosa.feature.spectral_contrast(
                    y=cpu_x,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_bands=6
                )
                
                # Convert to torch tensor
                contrast_tensor = torch.from_numpy(contrast).float()
                results.append(contrast_tensor.unsqueeze(0))
            except Exception:
                dummy_contrast = torch.zeros((1, 6, 50), dtype=torch.float32)
                results.append(dummy_contrast)
        
        # Stack the batch
        contrast_batch = torch.cat(results, dim=0)
        
        return contrast_batch
        
    def extract_all_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all audio features from a waveform.
        
        Args:
            waveform: Input waveform tensor (batch_size, num_samples)
            
        Returns:
            Dictionary of extracted features
        """
        # Validate input
        if waveform.dim() == 0 or waveform.numel() == 0:
            batch_size = 1 if waveform.dim() == 0 else waveform.shape[0]
            # Create zero tensors for empty input
            return {
                'harmonic': torch.zeros((batch_size, 1, self.n_mels, 50), dtype=torch.float32),
                'percussive': torch.zeros((batch_size, 1, self.n_mels, 50), dtype=torch.float32),
                'chroma': torch.zeros((batch_size, 12, 50), dtype=torch.float32),
                'spectral_contrast': torch.zeros((batch_size, 6, 50), dtype=torch.float32)
            }
            
        try:
            harmonic, percussive = self.extract_harmonic_percussive(waveform)
            chroma = self.extract_chroma(waveform)
            spectral_contrast = self.extract_spectral_contrast(waveform)
            
            return {
                'harmonic': harmonic,
                'percussive': percussive, 
                'chroma': chroma,
                'spectral_contrast': spectral_contrast
            }
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            batch_size = waveform.shape[0]
            
            # Fall back to zeros if processing fails
            return {
                'harmonic': torch.zeros((batch_size, 1, self.n_mels, 50), dtype=torch.float32),
                'percussive': torch.zeros((batch_size, 1, self.n_mels, 50), dtype=torch.float32),
                'chroma': torch.zeros((batch_size, 12, 50), dtype=torch.float32),
                'spectral_contrast': torch.zeros((batch_size, 6, 50), dtype=torch.float32)
            }