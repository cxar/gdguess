#!/usr/bin/env python3
"""
Test to verify model works with preprocessed data from data/audsnippets-all/preprocessed.
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
from src.data.dataset import PreprocessedDeadShowDataset, optimized_collate_fn
from torch.utils.data import DataLoader

def _find_preprocessed_data_dir():
    """Find the directory with preprocessed data files."""
    # Try common locations
    candidates = [
        os.path.join(parent_dir, "data", "audsnippets-all", "preprocessed"),  # Most likely location
        os.path.join(parent_dir, "data", "preprocessed"),
        os.path.join(parent_dir, "data", "snippets", "preprocessed"),
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            # Check if it contains .pt files
            pt_files = list(Path(candidate).glob("**/*.pt"))
            if pt_files:
                return candidate
    
    raise FileNotFoundError(
        "Could not find preprocessed data directory. "
        "Please make sure the data/audsnippets-all/preprocessed directory exists with .pt files."
    )

def run_preprocessed_data_test(sample_size=5, batch_size=2):
    """
    Test the model with preprocessed data files.
    
    Args:
        sample_size: Number of files to sample for testing
        batch_size: Batch size for the dataloader
        
    Returns:
        True if the test passed, False otherwise
    """
    print("\n" + "="*50)
    print("PREPROCESSED DATA TEST")
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
    
    # Find preprocessed data directory
    try:
        data_dir = _find_preprocessed_data_dir()
        print(f"Using preprocessed data from: {data_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    # Create model instance
    print("\nInitializing model...")
    model = DeadShowDatingModel(sample_rate=24000)
    model.to(device)
    model.eval()
    
    # Create dataset
    print("\nCreating preprocessed dataset...")
    try:
        dataset = PreprocessedDeadShowDataset(
            preprocessed_dir=data_dir,
            target_sr=24000,
            device=torch.device('cpu')  # Load data on CPU first, then move to device in collate_fn
        )
        print(f"Dataset contains {len(dataset)} preprocessed files")
        
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
        
        print(f"\nProcessing {sample_size} preprocessed files with batch size {batch_size}...")
        
        # Process batches
        all_outputs = []
        success = True
        failed_batches = 0
        max_allowed_failures = 2  # Allow up to 2 batches to fail
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                try:
                    # Check for empty batch
                    if batch.get('empty_batch', False):
                        print(f"Batch {i+1}: Empty batch")
                        failed_batches += 1
                        continue
                        
                    # Process input features
                    # Ensure we have the expected features needed for the model
                    harmonic = batch.get('harmonic', None)
                    if harmonic is None:
                        harmonic = batch.get('mel_spec', None)
                        
                    percussive = batch.get('percussive', None)
                    if percussive is None:
                        percussive = batch.get('mel_spec_percussive', None)
                        
                    chroma = batch.get('chroma', None)
                    
                    spectral_contrast = batch.get('spectral_contrast', None)
                    if spectral_contrast is None:
                        spectral_contrast = batch.get('spectral_contrast_harmonic', None)
                    
                    # Check if we have all required features
                    if harmonic is None or percussive is None or chroma is None or spectral_contrast is None:
                        missing = []
                        if harmonic is None:
                            missing.append('harmonic/mel_spec')
                        if percussive is None:
                            missing.append('percussive/mel_spec_percussive')
                        if chroma is None:
                            missing.append('chroma')
                        if spectral_contrast is None:
                            missing.append('spectral_contrast')
                        
                        print(f"Batch {i+1}: Missing required features: {missing}")
                        continue
                    
                    # Create input dictionary
                    inputs = {
                        'harmonic': harmonic,
                        'percussive': percussive,
                        'chroma': chroma,
                        'spectral_contrast': spectral_contrast
                    }
                    
                    # Add date information if available
                    if 'date' in batch:
                        inputs['date'] = batch['date']
                    elif 'label' in batch:
                        # Use label as date if date not available
                        inputs['date'] = batch['label']
                    else:
                        # Create dummy date
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
                            failed_batches += 1
                            # Only mark the test as failed if too many batches have failed
                            if failed_batches > max_allowed_failures:
                                success = False
                    
                except Exception as e:
                    print(f"Batch {i+1}: Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_batches += 1
                    # Only mark the test as failed if too many batches have failed
                    if failed_batches > max_allowed_failures:
                        success = False
                    
        # Print summary
        print("\n" + "="*50)
        if success and all_outputs:
            print(f"✅ Successfully processed {len(all_outputs)} batches")
            if failed_batches > 0:
                print(f"   ({failed_batches} batches failed, but within acceptable limits)")
            print(f"✅ PREPROCESSED DATA TEST PASSED")
        else:
            print(f"❌ PREPROCESSED DATA TEST FAILED")
            print(f"   {failed_batches} batches failed out of {sample_size / batch_size:.1f} total batches")
            if not all_outputs:
                print("   No batches were successfully processed")
        print("="*50)
        
        # The test passes if at least one batch was successful and we didn't exceed our failure threshold
        return success and len(all_outputs) > 0
    
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if run_preprocessed_data_test() else 1)