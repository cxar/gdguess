#!/usr/bin/env python3
"""
Inference functionality for the Grateful Dead show dating model.
"""

import datetime
from typing import Dict, Union, Optional, List, Any

import numpy as np
import librosa
import torch
import torch.nn.functional as F

from models.dead_model import DeadShowDatingModel
from inference.utils.tta import test_time_augmentation
from config import Config
from utils.helpers import ensure_device_consistency, prepare_batch


def extract_audio_features(
    audio_np: np.ndarray, 
    sr: int = 24000
) -> Dict[str, torch.Tensor]:
    """
    Extract all advanced audio features from numpy array.
    
    Args:
        audio_np: Audio data as numpy array
        sr: Sample rate
        
    Returns:
        Dictionary of extracted features as tensors
    """
    # Harmonic-percussive source separation
    harmonic, percussive = librosa.effects.hpss(audio_np)
    
    # Compute spectral contrast for both components
    harmonic_contrast = librosa.feature.spectral_contrast(
        y=harmonic, 
        sr=sr, 
        n_fft=2048,
        hop_length=512,
        n_bands=6
    )
    
    percussive_contrast = librosa.feature.spectral_contrast(
        y=percussive, 
        sr=sr, 
        n_fft=2048,
        hop_length=512,
        n_bands=6
    )
    
    # Compute chroma from harmonic component
    chroma = librosa.feature.chroma_stft(
        y=harmonic, 
        sr=sr, 
        n_fft=2048,
        hop_length=512,
        n_chroma=12
    )
    
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=audio_np, 
        sr=sr,
        hop_length=512
    )
    
    # Convert all features to tensors
    features = {
        "harmonic": torch.from_numpy(harmonic),
        "percussive": torch.from_numpy(percussive),
        "spectral_contrast_harmonic": torch.from_numpy(harmonic_contrast),
        "spectral_contrast_percussive": torch.from_numpy(percussive_contrast),
        "chroma": torch.from_numpy(chroma),
        "onset_env": torch.from_numpy(onset_env)
    }
    
    return features


def predict_date(
    model: DeadShowDatingModel,
    audio_path: str = None,
    base_date: datetime.date = datetime.date(1968, 1, 1),
    target_sr: int = 24000,
    device: Union[str, torch.device] = "cuda",
    use_tta: bool = False,
    tta_transforms: int = 5,
    tta_intensity: str = "medium",
    audio_tensor: torch.Tensor = None,
    enable_uncertainty: bool = False,
    uncertainty_samples: int = 30,
    config: Optional[Config] = None,
) -> Dict:
    """
    Run inference on a single audio file to predict its date.

    Args:
        model: Trained model
        audio_path: Path to audio file
        base_date: Base date for conversion
        target_sr: Target sample rate
        device: Device to run inference on
        use_tta: Whether to use test-time augmentation
        tta_transforms: Number of TTA transforms to use
        tta_intensity: Intensity of TTA transforms ("light", "medium", "heavy")
        audio_tensor: Optional audio tensor to use instead of loading from path
        enable_uncertainty: Whether to estimate uncertainty using Monte Carlo Dropout
        uncertainty_samples: Number of samples for Monte Carlo Dropout
        config: Optional configuration object for device consistency

    Returns:
        Dictionary with prediction results and uncertainty if enabled
    """
    # Create default config if none provided
    if config is None:
        config = Config()
        config.device = str(device)
    
    model.eval()
    # Use device consistency utility
    model = ensure_device_consistency(model, config)
    
    # Load and process audio
    if audio_tensor is not None:
        audio = audio_tensor
    else:
        audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Standardize length to 15 seconds
    desired_length = target_sr * 15
    if isinstance(audio, torch.Tensor):
        # Handle tensor input
        if audio.dim() == 1:
            if audio.size(0) < desired_length:
                # Pad shorter audio
                padding = desired_length - audio.size(0)
                audio = F.pad(audio, (0, padding))
            else:
                # Trim longer audio
                audio = audio[:desired_length]
    else:
        # Handle numpy array
        if len(audio) < desired_length:
            # Pad shorter audio
            padding = desired_length - len(audio)
            audio = np.pad(audio, (0, padding))
        else:
            # Trim longer audio
            audio = audio[:desired_length]
    
    if use_tta:
        # Extract features using TTA
        prediction = test_time_augmentation(
            model=model,
            audio=audio,
            sr=target_sr,
            num_transforms=tta_transforms,
            intensity=tta_intensity,
            device=device,
            extract_features_fn=extract_audio_features
        )
    elif enable_uncertainty:
        # Import the uncertainty estimation function here to avoid circular imports
        from inference.utils.uncertainty import estimate_date_uncertainty
        
        # Extract features
        features = extract_audio_features(audio, target_sr)
        
        # Add batch dimension for all tensors
        features = {k: v.unsqueeze(0) if v.dim() == 1 else v.unsqueeze(0) 
                   for k, v in features.items()}
        
        # Use device consistency utility for features
        features = prepare_batch(features, config)
        
        # Use Monte Carlo Dropout for uncertainty estimation
        prediction = estimate_date_uncertainty(
            model=model,
            inputs=features,
            base_date=base_date,
            num_samples=uncertainty_samples,
            device=device
        )
    else:
        # Normal inference without TTA or uncertainty
        # Extract features
        features = extract_audio_features(audio, target_sr)
        
        # Add batch dimension for all tensors
        features = {k: v.unsqueeze(0) if v.dim() == 1 else v.unsqueeze(0) 
                   for k, v in features.items()}
        
        # Use device consistency utility for features
        features = prepare_batch(features, config)
        
        # Run inference
        with torch.no_grad():
            prediction = model(features)
    
    # Process prediction
    if isinstance(prediction, dict):
        # If using uncertainty estimation from Monte Carlo, the function already returns a processed result
        if enable_uncertainty:
            return prediction
        
        result = {}
        
        # Handle date offset prediction
        if "days" in prediction:
            days = prediction["days"].item()
            predicted_date = base_date + datetime.timedelta(days=days)
            result["predicted_date"] = predicted_date
            result["days_offset"] = days
            
            # Handle direct uncertainty prediction if available
            if "log_variance" in prediction:
                log_var = prediction["log_variance"].item()
                # Convert log_variance to standard deviation (in days)
                uncertainty = torch.exp(0.5 * torch.tensor(log_var)).item()
                
                # Calculate 95% confidence interval (approximately ±2 standard deviations)
                lower_bound = predicted_date - datetime.timedelta(days=int(uncertainty * 2))
                upper_bound = predicted_date + datetime.timedelta(days=int(uncertainty * 2))
                
                # Calculate a confidence score (0-100%)
                # Lower uncertainty = higher confidence
                confidence_score = min(100, int(100 * (1 / (1 + uncertainty/50))))
                
                # Add uncertainty info to result
                result["date_uncertainty"] = {
                    "std_days": uncertainty,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "confidence_score": confidence_score
                }
        
        # Handle era classification
        if "era_logits" in prediction:
            era_probs = F.softmax(prediction["era_logits"], dim=-1)
            era_idx = era_probs.argmax().item()
            era_conf = era_probs[0, era_idx].item()
            
            # Map index to era
            era_map = {0: "early", 1: "seventies", 2: "eighties", 3: "nineties", 4: "later"}
            result["predicted_era"] = era_map[era_idx]
            result["era_confidence"] = era_conf
        
        # Include raw predictions for debugging
        for key, val in prediction.items():
            if isinstance(val, torch.Tensor):
                result[f"raw_{key}"] = val.cpu().numpy()
    else:
        # Simpler model just predicts days offset
        days = prediction.item()
        predicted_date = base_date + datetime.timedelta(days=days)
        result = {
            "predicted_date": predicted_date,
            "days_offset": days
        }
    
    return result


def batch_predict(
    model: DeadShowDatingModel,
    audio_paths: List[str],
    base_date: datetime.date = datetime.date(1968, 1, 1),
    target_sr: int = 24000,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 16,
    use_tta: bool = False,
    tta_transforms: int = 5,
    tta_intensity: str = "medium",
    enable_uncertainty: bool = False,
    uncertainty_samples: int = 30,
    config: Optional[Config] = None,
) -> List[Dict[str, Any]]:
    """
    Run inference on multiple audio files in batches.

    Args:
        model: Trained model
        audio_paths: List of paths to audio files
        base_date: Base date for conversion
        target_sr: Target sample rate
        device: Device to run inference on
        batch_size: Batch size for processing
        use_tta: Whether to use test-time augmentation
        tta_transforms: Number of TTA transforms to use
        tta_intensity: Intensity of TTA transforms
        enable_uncertainty: Whether to estimate uncertainty using Monte Carlo Dropout
        uncertainty_samples: Number of samples for Monte Carlo Dropout
        config: Optional configuration object for device consistency

    Returns:
        List of dictionaries with prediction results
    """
    # Create default config if none provided
    if config is None:
        config = Config()
        config.device = str(device)
    
    model.eval()
    # Use device consistency utility
    model = ensure_device_consistency(model, config)
    
    results = []
    
    # Process in batches
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        batch_results = []
        
        for audio_path in batch_paths:
            # Use single prediction function to handle each file
            result = predict_date(
                model=model,
                audio_path=audio_path,
                base_date=base_date,
                target_sr=target_sr,
                device=device,
                use_tta=use_tta,
                tta_transforms=tta_transforms,
                tta_intensity=tta_intensity,
                enable_uncertainty=enable_uncertainty,
                uncertainty_samples=uncertainty_samples,
                config=config  # Pass the config object
            )
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results
