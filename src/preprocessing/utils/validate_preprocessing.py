#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
import argparse
from collections import defaultdict

def validate_preprocessed_data(preprocessed_dir: str, verbose: bool = False):
    """
    Validate preprocessed data files to ensure they contain all expected features.
    
    Args:
        preprocessed_dir: Directory containing preprocessed .pt files
        verbose: Whether to print detailed information about each file
    """
    # Expected features
    expected_features = {
        'mel_spec',
        'mel_spec_percussive',
        'spectral_contrast_harmonic',
        'chroma',
        'onset_env',
        'label',
        'era',
        'file'
    }
    
    # Statistics
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'missing_features': defaultdict(int),
        'shape_stats': defaultdict(list)
    }
    
    # Get all .pt files
    pt_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')])
    
    if not pt_files:
        print(f"No .pt files found in {preprocessed_dir}")
        return
    
    print(f"\nValidating {len(pt_files)} files in {preprocessed_dir}")
    
    for pt_file in tqdm(pt_files, desc="Validating files"):
        stats['total_files'] += 1
        file_path = os.path.join(preprocessed_dir, pt_file)
        
        try:
            data = torch.load(file_path)
            
            # Check for missing features
            missing = expected_features - set(data.keys())
            if missing:
                for feature in missing:
                    stats['missing_features'][feature] += 1
                if verbose:
                    print(f"\nFile {pt_file} missing features: {missing}")
            
            # Collect shape statistics for tensor features
            for feature, value in data.items():
                if isinstance(value, torch.Tensor):
                    stats['shape_stats'][feature].append(value.shape)
            
            if not missing:
                stats['valid_files'] += 1
                
        except Exception as e:
            print(f"\nError loading {pt_file}: {str(e)}")
            continue
    
    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Valid files (all features present): {stats['valid_files']}")
    
    if stats['missing_features']:
        print("\nMissing features count:")
        for feature, count in stats['missing_features'].items():
            print(f"  {feature}: missing in {count} files")
    
    print("\nFeature shapes (min/max dimensions):")
    for feature, shapes in stats['shape_stats'].items():
        if shapes:
            print(f"\n{feature}:")
            # Convert shapes to strings for easier comparison
            shape_strs = [str(s) for s in shapes]
            if len(set(shape_strs)) == 1:
                print(f"  Consistent shape: {shapes[0]}")
            else:
                print(f"  Variable shapes found: {set(shape_strs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate preprocessed data files")
    parser.add_argument("--dir", type=str, default="data/preprocessed",
                      help="Directory containing preprocessed .pt files")
    parser.add_argument("--verbose", action="store_true",
                      help="Print detailed information about each file")
    
    args = parser.parse_args()
    validate_preprocessed_data(args.dir, args.verbose) 