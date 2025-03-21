#!/usr/bin/env python3
"""
Test the Transformer model with real audio data from the audsnippets-all collection.
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.models.dead_model import DeadShowDatingModel
from src.data.dataset import DeadShowDataset

def _find_data_dir():
    """Find the data directory with audio snippets."""
    # Try common locations
    candidates = [
        os.path.join(parent_dir, "data", "audsnippets-all"),  # Most likely location
        os.path.join(parent_dir, "data", "snippets"),
        os.path.join(parent_dir, "data"),
        os.path.join(parent_dir, "..", "data", "audsnippets-all"),
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            # Check if it contains MP3 files
            mp3_files = list(Path(candidate).glob("**/*.mp3"))
            if mp3_files:
                return candidate
    
    raise FileNotFoundError(
        "Could not find data directory with audio snippets. "
        "Please make sure the audsnippets-all directory exists in the data/ directory."
    )

def run_transformer_real_data_test():
    """
    Test the transformer model with real audio data.
    
    Returns:
        True if the test passed, False otherwise
    """
    print("\n" + "="*50)
    print("TRANSFORMER MODEL WITH REAL DATA TEST")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if CUDA or MPS is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Find data directory
    try:
        data_dir = _find_data_dir()
        print(f"Using data from: {data_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    # Create model instance
    print("\nInitializing model...")
    model = DeadShowDatingModel(sample_rate=24000)
    model.to(device)
    model.eval()
    
    # Load a single audio file for testing
    print("\nSearching for audio files...")
    mp3_files = list(Path(data_dir).glob("**/*.mp3"))
    
    if not mp3_files:
        print("Error: No MP3 files found in the data directory")
        return False
    
    # Select a random audio file
    audio_path = random.choice(mp3_files)
    print(f"Selected audio file: {audio_path}")
    
    try:
        # Load and process the audio
        import torchaudio
        
        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        
        print(f"Audio shape: {audio.shape}")
        
        # Ensure we have enough audio (at least 5 seconds @ 24kHz)
        min_length = 24000 * 5
        if audio.shape[1] < min_length:
            # Pad if too short
            padding = min_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            # Trim if too long
            audio = audio[:, :min_length]
        
        print(f"Processed audio shape: {audio.shape}")
        
        # Move to device (after processing)
        audio = audio.to(device)
        
        # Create input dictionary for the model
        inputs = {
            "audio": audio,
            "date": torch.tensor([100.0], device=device)  # Dummy date (days since 1968-01-01)
        }
        
        # Run the model
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(inputs)
        
        # Check outputs
        print("\nChecking outputs...")
        
        # Check if we have the expected output keys
        expected_keys = ['days', 'era_logits']
        missing_keys = [key for key in expected_keys if key not in outputs]
        
        if missing_keys:
            print(f"❌ Missing expected keys in output: {missing_keys}")
            return False
        
        # Check output shapes and values
        print(f"days: {outputs['days']}")
        print(f"era_logits shape: {outputs['era_logits'].shape}")
        
        # Check for NaN values
        has_nans = torch.isnan(outputs['days']).any() or torch.isnan(outputs['era_logits']).any()
        
        if has_nans:
            print("❌ Output contains NaN values")
            return False
        
        # Test passed
        print("\n" + "="*50)
        print("✅ TRANSFORMER MODEL WITH REAL DATA TEST PASSED")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if run_transformer_real_data_test() else 1)