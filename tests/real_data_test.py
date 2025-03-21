#!/usr/bin/env python3
"""
Test to verify model works with real data from the audsnippets-all collection.
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
from src.data.dataset import DeadShowDataset, optimized_collate_fn
from torch.utils.data import DataLoader

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

def run_real_data_test(sample_size=5, batch_size=2):
    """
    Test the model with real data from the audsnippets-all collection.
    
    Args:
        sample_size: Number of audio files to sample for testing
        batch_size: Batch size for the dataloader
        
    Returns:
        True if the test passed, False otherwise
    """
    print("\n" + "="*50)
    print("REAL DATA TEST")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # Create dataset
    print("\nCreating dataset...")
    try:
        dataset = DeadShowDataset(root_dir=data_dir, target_sr=24000, verbose=True)
        print(f"Dataset contains {len(dataset)} audio files")
        
        if len(dataset) < sample_size:
            print(f"Warning: Dataset contains fewer files ({len(dataset)}) than requested sample size ({sample_size})")
            sample_size = len(dataset)
        
        # Create indices for a random subset
        indices = random.sample(range(len(dataset)), sample_size)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda x: optimized_collate_fn(x, device=device),
            sampler=indices,  # Use only the sampled indices
        )
        
        print(f"\nProcessing {sample_size} audio files with batch size {batch_size}...")
        
        # Process batches
        all_outputs = []
        success = True
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                try:
                    # Forward pass
                    harmonic = batch.get("harmonic")
                    percussive = batch.get("percussive")
                    chroma = batch.get("chroma")
                    spectral_contrast = batch.get("spectral_contrast")
                    
                    # Manually create input batch if needed
                    if harmonic is None:
                        # Convert raw audio to input features
                        audio = batch.get("audio")
                        if audio is None:
                            print(f"Batch {i+1}: Missing audio data")
                            continue
                        
                        # Process audio
                        outputs = model(audio)
                    else:
                        # Use preprocessed features
                        inputs = {
                            'harmonic': harmonic,
                            'percussive': percussive,
                            'chroma': chroma,
                            'spectral_contrast': spectral_contrast
                        }
                        
                        # Add dummy date for classification
                        if 'date' not in inputs:
                            inputs['date'] = torch.tensor([100] * harmonic.size(0), device=device)
                        
                        # Forward pass
                        outputs = model(inputs)
                    
                    # Check outputs
                    if not isinstance(outputs, dict):
                        print(f"Batch {i+1}: Expected dict output, got {type(outputs)}")
                        success = False
                        continue
                    
                    required_keys = ['days', 'era_logits']
                    missing_keys = [key for key in required_keys if key not in outputs]
                    
                    if missing_keys:
                        print(f"Batch {i+1}: Missing required keys: {missing_keys}")
                        success = False
                    else:
                        # Store outputs
                        all_outputs.append(outputs)
                        
                        # Print output values
                        print(f"Batch {i+1}:")
                        print(f"  days: {outputs['days']}")
                        print(f"  era_logits.shape: {outputs['era_logits'].shape}")
                        
                        # Check for NaN values
                        if torch.isnan(outputs['days']).any() or torch.isnan(outputs['era_logits']).any():
                            print(f"Batch {i+1}: Output contains NaN values")
                            success = False
                        
                except Exception as e:
                    print(f"Batch {i+1}: Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    success = False
                    
        # Print summary
        print("\n" + "="*50)
        if success:
            print(f"✅ Successfully processed {len(all_outputs)} batches")
            print(f"✅ REAL DATA TEST PASSED")
        else:
            print(f"❌ REAL DATA TEST FAILED")
        print("="*50)
        
        return success
    
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if run_real_data_test() else 1)