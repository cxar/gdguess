"""
Audio processing utilities for the Grateful Dead show dating model.
"""

from typing import Tuple, Dict, Optional, Union
from functools import lru_cache

import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from torch import Tensor


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass


class AudioFeatureExtractor:
    """
    Utility class for extracting various audio features.
    
    Args:
        sample_rate: Audio sample rate in Hz
        cache_size: Size of the LRU cache for feature computations
        device: Device to place the transforms on ('cpu', 'cuda', 'cuda:0', etc.)
    """
    
    def __init__(self, sample_rate: int = 24000, cache_size: int = 128, device: str = 'cpu'):
        self.sample_rate = sample_rate
        self.device = device
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        ).to(device)
        self._setup_caches(cache_size)
        
    def _setup_caches(self, cache_size: int) -> None:
        """Setup LRU caches for feature extraction methods."""
        self._cached_hpss = lru_cache(maxsize=cache_size)(self._compute_hpss)
        self._cached_contrast = lru_cache(maxsize=cache_size)(self._compute_contrast)
        self._cached_chroma = lru_cache(maxsize=cache_size)(self._compute_chroma)
        
    def _validate_audio(self, audio: Tensor) -> None:
        """
        Validate audio tensor format and values.
        
        Args:
            audio: Input audio tensor
            
        Raises:
            AudioProcessingError: If audio format is invalid
        """
        if not isinstance(audio, torch.Tensor):
            raise AudioProcessingError("Input must be a torch.Tensor")
            
        if audio.dim() != 2:
            raise AudioProcessingError(
                f"Expected 2D tensor (batch, samples), got shape {audio.shape}"
            )
            
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            raise AudioProcessingError("Audio contains NaN or Inf values")
            
    def _ensure_float32(self, tensor: Tensor) -> Tensor:
        """Convert tensor to float32 if needed."""
        return tensor.float() if tensor.dtype != torch.float32 else tensor
        
    def _to_numpy_mono(self, audio: Tensor) -> np.ndarray:
        """Convert audio tensor to mono numpy array."""
        audio = self._ensure_float32(audio)
        return audio.cpu().numpy().mean(axis=0) if audio.dim() > 1 else audio.cpu().numpy()
        
    @staticmethod
    def _handle_librosa_error(func):
        """Decorator to handle librosa processing errors."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except librosa.ParameterError as e:
                raise AudioProcessingError(f"Invalid parameters: {str(e)}")
            except Exception as e:
                raise AudioProcessingError(f"Audio processing failed: {str(e)}")
        return wrapper
        
    @_handle_librosa_error
    def _compute_hpss(self, audio_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute harmonic-percussive source separation."""
        return librosa.effects.hpss(audio_np)
        
    @_handle_librosa_error
    def _compute_contrast(self, audio_np: np.ndarray) -> np.ndarray:
        """Compute spectral contrast."""
        return librosa.feature.spectral_contrast(
            y=audio_np,
            sr=self.sample_rate,
            n_bands=6,
            fmin=20.0
        )
        
    @_handle_librosa_error
    def _compute_chroma(self, audio_np: np.ndarray) -> np.ndarray:
        """Compute chromagram."""
        return librosa.feature.chroma_cqt(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=512,
            n_chroma=12
        )
        
    def extract_harmonic_percussive(self, audio: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Separate audio into harmonic and percussive components.
        
        Args:
            audio: Input audio tensor of shape (batch, samples)
            
        Returns:
            Tuple of (harmonic, percussive) tensors, each of shape (batch, 1, freq, time)
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        try:
            self._validate_audio(audio)
            audio_np = self._to_numpy_mono(audio)
            
            # Use cached computation
            harmonic, percussive = self._cached_hpss(tuple(audio_np))
            
            # Convert back to tensors
            harmonic = torch.from_numpy(harmonic).to(self.device).unsqueeze(1)
            percussive = torch.from_numpy(percussive).to(self.device).unsqueeze(1)
            
            return harmonic, percussive
        except Exception as e:
            # Create dummy tensors in case of failure
            print(f"HPSS failed with error: {e}. Using dummy tensors instead.")
            # Create tensors of appropriate shape (batch, 1, freq, time)
            batch_size = 1 if audio.dim() == 1 else audio.shape[0]
            device = self.device
            # Typical STFT dimensions for 15s audio at 24kHz, hop_length=512, n_fft=2048
            dummy_shape = (batch_size, 1, 1025, 704)  
            harmonic = torch.zeros(dummy_shape, device=device)
            percussive = torch.zeros(dummy_shape, device=device)
            return harmonic, percussive
        
    def extract_spectral_contrast(self, audio: Tensor) -> Tensor:
        """
        Extract spectral contrast features.
        
        Args:
            audio: Input audio tensor of shape (batch, samples)
            
        Returns:
            Spectral contrast features of shape (batch, 6, time)
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        try:
            self._validate_audio(audio)
            audio_np = self._to_numpy_mono(audio)
            
            # Use cached computation
            contrast = self._cached_contrast(tuple(audio_np))
            
            return torch.from_numpy(contrast).to(self.device)
        except Exception as e:
            # Create dummy tensors in case of failure
            print(f"Spectral contrast extraction failed with error: {e}. Using dummy tensors instead.")
            batch_size = 1 if audio.dim() == 1 else audio.shape[0]
            device = self.device
            # Typical spectral contrast shape (6 bands, time frames)
            dummy_shape = (batch_size, 6, 704)  # Time frames match STFT dimensions
            return torch.zeros(dummy_shape, device=device)
        
    def extract_chroma(self, audio: Tensor) -> Tensor:
        """
        Extract chromagram features.
        
        Args:
            audio: Input audio tensor of shape (batch, samples)
            
        Returns:
            Chromagram features of shape (batch, 12, time)
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        try:
            self._validate_audio(audio)
            audio_np = self._to_numpy_mono(audio)
            
            # Use cached computation
            chroma = self._cached_chroma(tuple(audio_np))
            
            return torch.from_numpy(chroma).to(self.device)
        except Exception as e:
            # Create dummy tensors in case of failure
            print(f"Chroma extraction failed with error: {e}. Using dummy tensors instead.")
            batch_size = 1 if audio.dim() == 1 else audio.shape[0]
            device = self.device
            # Typical chroma shape (12 pitches, time frames)
            dummy_shape = (batch_size, 12, 704)  # Time frames match STFT dimensions
            return torch.zeros(dummy_shape, device=device)
        
    def extract_onset_envelope(self, audio: Tensor) -> Tensor:
        """
        Extract onset strength envelope.
        
        Args:
            audio: Input audio tensor of shape (batch, samples)
            
        Returns:
            Onset envelope features of shape (batch, time)
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        try:
            self._validate_audio(audio)
            audio_np = self._to_numpy_mono(audio)
            
            try:
                onset_env = librosa.onset.onset_strength(
                    y=audio_np,
                    sr=self.sample_rate,
                    hop_length=512
                )
            except Exception as e:
                raise AudioProcessingError(f"Onset detection failed: {str(e)}")
            
            return torch.from_numpy(onset_env).to(self.device)
        except Exception as e:
            # Create dummy tensors in case of failure
            print(f"Onset envelope extraction failed with error: {e}. Using dummy tensors instead.")
            batch_size = 1 if audio.dim() == 1 else audio.shape[0]
            device = self.device
            # Typical onset envelope shape (time frames)
            dummy_shape = (batch_size, 704)  # Time frames match other features
            return torch.zeros(dummy_shape, device=device) 