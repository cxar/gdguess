#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) module for more robust predictions.

This module provides functions for applying test-time augmentation during inference,
which can improve prediction robustness by averaging results across multiple
augmentations of the same input.
"""

import numpy as np
import torch
import librosa
import scipy.signal
from typing import Dict, List, Union, Callable, Tuple, Optional


def apply_pitch_shift(
    audio: np.ndarray, sr: int, n_steps: float = 0.5
) -> np.ndarray:
    """
    Apply pitch shifting to audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_steps: Number of semitones to shift
        
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def apply_time_stretch(
    audio: np.ndarray, rate: float = 0.95
) -> np.ndarray:
    """
    Apply time stretching to audio.
    
    Args:
        audio: Input audio signal
        rate: Time stretch factor (0.95 = slow down by 5%)
        
    Returns:
        Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def apply_noise(
    audio: np.ndarray, noise_level: float = 0.005
) -> np.ndarray:
    """
    Add shaped noise to audio.
    
    Args:
        audio: Input audio signal
        noise_level: Noise amplitude
        
    Returns:
        Noisy audio
    """
    noise = np.random.randn(len(audio)) * noise_level
    
    # Shape noise to have more high frequency content
    b, a = scipy.signal.butter(1, 0.1, 'highpass', analog=False)
    shaped_noise = scipy.signal.filtfilt(b, a, noise)
    
    return audio + shaped_noise


def apply_eq(
    audio: np.ndarray, 
    sr: int,
    low_boost: float = 0.0, 
    mid_boost: float = 0.0, 
    high_boost: float = 0.0
) -> np.ndarray:
    """
    Apply equalization to audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        low_boost: Boost factor for low frequencies
        mid_boost: Boost factor for mid frequencies
        high_boost: Boost factor for high frequencies
        
    Returns:
        Equalized audio
    """
    X = np.fft.rfft(audio)
    freq = np.fft.rfftfreq(len(audio), 1 / sr)
    
    mask = np.ones_like(X, dtype=float)
    
    # Apply boosts
    low_mask = freq < 250
    mid_mask = np.logical_and(freq >= 250, freq <= 4000)
    high_mask = freq > 4000
    
    mask[low_mask] *= (1.0 + low_boost)
    mask[mid_mask] *= (1.0 + mid_boost)
    mask[high_mask] *= (1.0 + high_boost)
    
    X_filtered = X * mask
    audio_filtered = np.fft.irfft(X_filtered, len(audio))
    
    return audio_filtered


def get_standard_tta_transforms(
    sr: int = 24000, intensity: str = "medium"
) -> List[Callable]:
    """
    Get standard test-time augmentation transforms.
    
    Args:
        sr: Sample rate
        intensity: Augmentation intensity ("light", "medium", or "heavy")
        
    Returns:
        List of augmentation functions
    """
    # Define intensity levels
    if intensity == "light":
        pitch_shifts = [-0.5, 0.5]
        time_stretches = [0.98, 1.02]
        noise_levels = [0.001]
        eq_settings = [
            (0.05, 0, 0),    # Low boost
            (0, 0.05, 0),    # Mid boost
            (0, 0, 0.05)     # High boost
        ]
    elif intensity == "heavy":
        pitch_shifts = [-1.0, -0.5, 0.5, 1.0]
        time_stretches = [0.95, 0.98, 1.02, 1.05]
        noise_levels = [0.001, 0.003]
        eq_settings = [
            (0.1, 0, 0),     # Low boost
            (0, 0.1, 0),     # Mid boost
            (0, 0, 0.1),     # High boost
            (0.05, 0.05, 0), # Low+Mid boost
            (0, 0.05, 0.05)  # Mid+High boost
        ]
    else:  # medium (default)
        pitch_shifts = [-0.7, 0.7]
        time_stretches = [0.97, 1.03]
        noise_levels = [0.002]
        eq_settings = [
            (0.07, 0, 0),    # Low boost
            (0, 0.07, 0),    # Mid boost
            (0, 0, 0.07)     # High boost
        ]
    
    # Create transform list
    transforms = []
    
    # Identity transform (no augmentation)
    transforms.append(lambda audio: audio)
    
    # Pitch shifts
    for shift in pitch_shifts:
        transforms.append(lambda audio, shift=shift: apply_pitch_shift(audio, sr, shift))
    
    # Time stretches
    for stretch in time_stretches:
        transforms.append(lambda audio, stretch=stretch: apply_time_stretch(audio, stretch))
    
    # Noise addition
    for noise_level in noise_levels:
        transforms.append(lambda audio, nl=noise_level: apply_noise(audio, nl))
    
    # EQ transforms
    for low, mid, high in eq_settings:
        transforms.append(lambda audio, l=low, m=mid, h=high: apply_eq(audio, sr, l, m, h))
    
    return transforms


def test_time_augmentation(
    model: torch.nn.Module,
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 24000,
    num_transforms: int = 5,
    intensity: str = "medium",
    device: Union[str, torch.device] = "cuda",
    extract_features_fn: Optional[Callable] = None,
    transforms: Optional[List[Callable]] = None,
) -> Dict:
    """
    Apply test-time augmentation to get more robust prediction.
    
    Args:
        model: PyTorch model
        audio: Input audio data (numpy array or tensor)
        sr: Sample rate
        num_transforms: Number of transforms to apply (randomly selected)
        intensity: Augmentation intensity
        device: PyTorch device
        extract_features_fn: Function to extract features from audio
        transforms: List of transform functions (if None, use standard transforms)
        
    Returns:
        Dictionary with averaged prediction results
    """
    # Convert torch tensor to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio
    
    # Get transforms
    if transforms is None:
        all_transforms = get_standard_tta_transforms(sr, intensity)
        
        # Always include identity (no transform)
        transforms = [all_transforms[0]]
        
        # Randomly select additional transforms if requested more than we have
        if num_transforms > len(all_transforms):
            num_transforms = len(all_transforms)
            
        # Add random selection of transforms
        if num_transforms > 1:
            transforms.extend(np.random.choice(all_transforms[1:], num_transforms - 1, replace=False))
    
    # Store predictions
    predictions = []
    
    # Apply each transform and run inference
    for transform in transforms:
        # Apply transform
        augmented_audio = transform(audio_np)
        
        # Extract features if function provided
        if extract_features_fn is not None:
            features = extract_features_fn(augmented_audio, sr)
            # Convert features to tensors if they aren't already
            features_tensor = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v.to(device) 
                              for k, v in features.items()}
            
            # Run model inference
            with torch.no_grad():
                prediction = model(features_tensor)
        else:
            # Convert to tensor
            audio_tensor = torch.tensor(augmented_audio, dtype=torch.float32).to(device)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Run model inference
            with torch.no_grad():
                prediction = model(audio_tensor)
        
        # Store prediction
        predictions.append(prediction)
    
    # Average predictions
    if isinstance(predictions[0], dict):
        # Handle dictionary predictions (e.g., multi-output models)
        result = {}
        for key in predictions[0].keys():
            if isinstance(predictions[0][key], torch.Tensor):
                result[key] = torch.mean(torch.stack([p[key] for p in predictions]), dim=0)
            elif isinstance(predictions[0][key], (int, float)):
                result[key] = sum(p[key] for p in predictions) / len(predictions)
            else:
                # Use the most common value for non-numeric outputs
                values = [p[key] for p in predictions]
                result[key] = max(set(values), key=values.count)
    elif isinstance(predictions[0], torch.Tensor):
        # Handle tensor predictions
        result = torch.mean(torch.stack(predictions), dim=0)
    else:
        # Handle numeric predictions
        result = sum(predictions) / len(predictions)
    
    return result 