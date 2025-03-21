#!/usr/bin/env python3
"""
Create a smaller training dataset for faster iterations.
"""

import os
import shutil
import argparse
import random
import glob
from pathlib import Path


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create a smaller training dataset subset")
    parser.add_argument("--source-dir", type=str, required=True, help="Source directory with preprocessed data")
    parser.add_argument("--target-dir", type=str, required=True, help="Target directory for the subset")
    parser.add_argument("--size", type=int, default=100, help="Number of files in the subset")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symbolic links")
    args = parser.parse_args()
    
    # Ensure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)
    
    # Find all .pt files in the source directory
    print(f"Searching for .pt files in {args.source_dir}...")
    pt_files = glob.glob(os.path.join(args.source_dir, "**/*.pt"), recursive=True)
    
    if not pt_files:
        print(f"No .pt files found in {args.source_dir}!")
        return
    
    print(f"Found {len(pt_files)} preprocessed files")
    
    # Sample files randomly
    subset_size = min(args.size, len(pt_files))
    sampled_files = random.sample(pt_files, subset_size)
    
    print(f"Creating a subset of {subset_size} files in {args.target_dir}")
    
    # Create subdirectories in target to match source structure
    for i, src_path in enumerate(sampled_files):
        # Get relative path from source root
        rel_path = os.path.relpath(src_path, args.source_dir)
        target_path = os.path.join(args.target_dir, rel_path)
        
        # Create parent directories
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Copy or link the file
        if args.copy:
            shutil.copy2(src_path, target_path)
        else:
            # For macOS compatibility, use relative paths for symlinks
            src_abs = os.path.abspath(src_path)
            target_abs = os.path.abspath(target_path)
            os.symlink(os.path.relpath(src_abs, os.path.dirname(target_abs)), target_path)
        
        # Progress update
        if (i + 1) % 10 == 0 or (i + 1) == subset_size:
            print(f"Processed {i + 1}/{subset_size} files")
    
    print("Done!")
    print(f"Created a subset with {subset_size} files in {args.target_dir}")
    print("\nTo use this subset for training, use:")
    print(f"./gdguess.py train --data-dir {args.target_dir} --batch-size 4 --steps 1000")


if __name__ == "__main__":
    main()