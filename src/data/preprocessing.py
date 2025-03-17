#!/usr/bin/env python3
"""
Data preprocessing utilities for the Grateful Dead show dating model.
"""

import glob
import os
import platform
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import librosa
import logging
import datetime

from .dataset import DeadShowDataset, identity_collate


def compute_spectral_contrast(audio_np: np.ndarray, sr: int = 24000, n_bands: int = 6) -> np.ndarray:
    """
    Compute spectral contrast for an audio signal.
    
    Args:
        audio_np: Numpy array containing audio data
        sr: Sample rate
        n_bands: Number of contrast bands
        
    Returns:
        Spectral contrast features
    """
    # Compute spectral contrast with 6 bands (default) and 2048 FFT size
    contrast = librosa.feature.spectral_contrast(
        y=audio_np, 
        sr=sr, 
        n_fft=2048,
        hop_length=512,
        n_bands=n_bands
    )
    return contrast


def compute_chroma(audio_np: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Compute chromagram for an audio signal.
    
    Args:
        audio_np: Numpy array containing audio data
        sr: Sample rate
        
    Returns:
        Chroma features
    """
    chroma = librosa.feature.chroma_stft(
        y=audio_np, 
        sr=sr, 
        n_fft=2048,
        hop_length=512,
        n_chroma=12
    )
    return chroma


def extract_advanced_features(
    audio: torch.Tensor, 
    sr: int = 24000
) -> Dict[str, torch.Tensor]:
    """
    Extract multiple advanced audio features from a given audio tensor.
    
    Args:
        audio: Tensor containing audio data
        sr: Sample rate
        
    Returns:
        Dictionary of audio features
    """
    # Convert to numpy for librosa processing
    audio_np = audio.numpy()
    
    # Harmonic-percussive source separation
    harmonic, percussive = librosa.effects.hpss(audio_np)
    
    # Compute spectral contrast for both harmonic and percussive components
    harmonic_contrast = compute_spectral_contrast(harmonic, sr=sr)
    percussive_contrast = compute_spectral_contrast(percussive, sr=sr)
    
    # Compute chroma from harmonic component (better for tonality)
    chroma = compute_chroma(harmonic, sr=sr)
    
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=audio_np, 
        sr=sr,
        hop_length=512
    )
    
    # Convert all features back to tensors
    features = {
        "harmonic": torch.from_numpy(harmonic),
        "percussive": torch.from_numpy(percussive),
        "harmonic_contrast": torch.from_numpy(harmonic_contrast),
        "percussive_contrast": torch.from_numpy(percussive_contrast),
        "chroma": torch.from_numpy(chroma),
        "onset_env": torch.from_numpy(onset_env)
    }
    
    return features


def preprocess_dataset(
    config: "Config", force_preprocess: bool = False, store_audio: bool = False, limit: int = 0
) -> str:
    """
    Preprocess the entire dataset once to avoid repeated CPU work.

    Args:
        config: Configuration object
        force_preprocess: Whether to force preprocessing even if files exist
        store_audio: Whether to store audio files alongside preprocessed data
        limit: Limit preprocessing to this many files (0 for no limit)

    Returns:
        Path to the preprocessed data directory
    """
    import multiprocessing
    from functools import partial
    
    # Check if the data directory already contains preprocessed data
    if os.path.exists(config.data_dir) and os.path.isdir(config.data_dir):
        # Check if this appears to be a preprocessed directory by looking for npz/pt files
        pt_files = glob.glob(os.path.join(config.data_dir, "**/*.pt"), recursive=True)
        npz_files = glob.glob(os.path.join(config.data_dir, "**/*.npz"), recursive=True)
        
        # Direct folder check for year-based organization
        year_dirs = []
        for item in os.listdir(config.data_dir):
            item_path = os.path.join(config.data_dir, item)
            if os.path.isdir(item_path):
                try:
                    # Check if folder name is a year (1965-1995)
                    year = int(item)
                    if 1965 <= year <= 1995:
                        year_files = glob.glob(os.path.join(item_path, "*.pt"))
                        if year_files:
                            year_dirs.append(item)
                except ValueError:
                    pass  # Not a year folder
        
        if pt_files or npz_files or year_dirs:
            if not force_preprocess:
                print(f"Found preprocessed files in {config.data_dir}, skipping preprocessing.")
                if year_dirs:
                    print(f"Year directories with .pt files: {', '.join(year_dirs)}")
                return config.data_dir
            else:
                print("Force preprocessing enabled, will overwrite existing files.")
    
    # Create a new directory for preprocessed data
    # Use a timestamp to avoid overwriting existing files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocessed_dir = os.path.join(config.data_dir, f"preprocessed_{timestamp}")
    
    # Safety check - don't overwrite if there are existing files
    if os.path.exists(preprocessed_dir):
        files_in_dir = os.listdir(preprocessed_dir)
        if files_in_dir and not force_preprocess:
            print(f"Directory {preprocessed_dir} already exists with {len(files_in_dir)} files.")
            print(f"Using existing directory. Use force_preprocess=True to overwrite.")
            return preprocessed_dir
    
    os.makedirs(preprocessed_dir, exist_ok=True)
    print(f"Preprocessing dataset to {preprocessed_dir}...")

    # Create original dataset for preprocessing
    try:
        orig_dataset = DeadShowDataset(
            root_dir=config.data_dir,
            target_sr=config.target_sr,
            augment=False,  # No augmentation during preprocessing
            verbose=True,
        )
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        print("Returning existing data directory as the preprocessed directory.")
        return config.data_dir

    # Apply limit if specified
    if limit > 0 and limit < len(orig_dataset):
        print(f"Limiting preprocessing to {limit} files (out of {len(orig_dataset)})")
        indices = list(range(min(limit, len(orig_dataset))))
        dataset_size = len(indices)
        from torch.utils.data import Subset
        dataset = Subset(orig_dataset, indices)
    else:
        indices = None
        dataset_size = len(orig_dataset)
        dataset = orig_dataset

    # Determine if MPS is available for GPU acceleration
    use_mps = torch.backends.mps.is_available() and config.device == "mps"
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Create mel spectrogram transform with advanced options
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.target_sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=20,
        f_max=8000,
        center=True,
        norm='slaney',
        mel_scale='htk'
    )
    
    # Move transform to device if using MPS
    if use_mps:
        mel_spec = mel_spec.to(device)
    
    # Set up audio storage (if needed)
    audio_dir = os.path.join(preprocessed_dir, "audio_files") if store_audio else None
    if store_audio:
        os.makedirs(audio_dir, exist_ok=True)
    
    def process_single_item(idx, item, preprocessed_dir, audio_dir, store_audio, use_mps, device, mel_spec, target_sr):
        try:
            # Convert audio to tensor if it's not already
            if isinstance(item, dict) and "audio" in item:
                audio = torch.as_tensor(item["audio"], dtype=torch.float)
            else:
                audio = torch.as_tensor(item["audio"], dtype=torch.float)
            
            # Pad or trim to exact length
            if audio.size(0) < target_sr * 15:
                audio = F.pad(audio, (0, target_sr * 15 - audio.size(0)))
            else:
                audio = audio[:target_sr * 15]
            
            # Extract all advanced features
            audio_np = audio.numpy()
            
            # Harmonic-percussive source separation
            harmonic, percussive = librosa.effects.hpss(audio_np)
            
            # Convert to tensors
            harmonic_tensor = torch.from_numpy(harmonic)
            percussive_tensor = torch.from_numpy(percussive)
            
            # Pre-compute mel spectrograms
            harmonic_mel = mel_spec(harmonic_tensor)
            harmonic_mel = torch.log(harmonic_mel + 1e-4)
            harmonic_mel = harmonic_mel.unsqueeze(0)
            
            percussive_mel = mel_spec(percussive_tensor)
            percussive_mel = torch.log(percussive_mel + 1e-4)
            percussive_mel = percussive_mel.unsqueeze(0)
            
            # Compute additional features
            harmonic_contrast = librosa.feature.spectral_contrast(
                y=harmonic, sr=target_sr, n_fft=2048, hop_length=512
            )
            
            chroma = librosa.feature.chroma_stft(
                y=harmonic, sr=target_sr, n_fft=2048, hop_length=512
            )
            
            onset_env = librosa.onset.onset_strength(
                y=percussive, sr=target_sr, hop_length=512
            )
            
            # Convert features to tensors
            harmonic_contrast_tensor = torch.from_numpy(harmonic_contrast)
            chroma_tensor = torch.from_numpy(chroma)
            onset_env_tensor = torch.from_numpy(onset_env)
            
            # Store data with all features
            output_data = {
                "mel_spec": harmonic_mel.half(),
                "mel_spec_percussive": percussive_mel.half(),
                "spectral_contrast_harmonic": harmonic_contrast_tensor.half(),
                "chroma": chroma_tensor.half(),
                "onset_env": onset_env_tensor.half(),
                "label": item["path"],  # Using path as label for now
                "era": torch.tensor(0),  # Placeholder
                "file": item["path"],
            }
            
            # Only store audio if explicitly requested
            if store_audio:
                audio_path = f"{audio_dir}/{idx:06d}.pt"
                torch.save(audio, audio_path)
                output_data["audio_file"] = audio_path
            
            # Save preprocessed data
            output_path = os.path.join(preprocessed_dir, f"{idx:06d}.pt")
            torch.save(output_data, output_path)
            return True
            
        except Exception as e:
            print(f"\nError processing item {idx}: {str(e)}")
            return False
    
    # Process items in parallel
    num_cores = min(16, multiprocessing.cpu_count())
    print(f"Using {num_cores} parallel workers for preprocessing")
    
    successful = 0
    with multiprocessing.Pool(num_cores) as pool:
        process_fn = partial(
            process_single_item,
            preprocessed_dir=preprocessed_dir,
            audio_dir=audio_dir,
            store_audio=store_audio,
            use_mps=use_mps,
            device=device,
            mel_spec=mel_spec,
            target_sr=config.target_sr
        )
        
        results = []
        with tqdm(total=dataset_size, desc="Preprocessing") as pbar:
            for idx in range(dataset_size):
                item = dataset[idx]
                result = pool.apply_async(
                    process_fn,
                    args=(idx, item),
                    callback=lambda _: pbar.update(1)
                )
                results.append(result)
            
            # Wait for all tasks to complete
            for result in results:
                if result.get():
                    successful += 1
    
    print(f"Preprocessing completed. {successful} of {dataset_size} items saved successfully.")
    return preprocessed_dir
