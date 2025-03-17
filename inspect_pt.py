#!/usr/bin/env python
"""
Inspect the structure of pt files in the dataset to understand the error.
"""

import os
import sys
import glob
import torch

def inspect_pt_file(file_path):
    """Inspect a single pt file and print its structure."""
    print(f"Examining {file_path}")
    try:
        data = torch.load(file_path, map_location='cpu')
        print(f"Keys in file: {list(data.keys())}")
        
        for k, v in data.items():
            print(f"{k}: {type(v)}", end="")
            if isinstance(v, torch.Tensor):
                print(f" shape={v.shape} dtype={v.dtype}", end="")
            print("")
            
        return True
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

def main():
    """Main function to inspect pt files."""
    # Find pt files
    data_dir = "./data/audsnippets-all/preprocessed"
    if not os.path.exists(data_dir):
        print(f"Directory does not exist: {data_dir}")
        print("Current directory:", os.getcwd())
        print("Available directories:", os.listdir("./data") if os.path.exists("./data") else "No ./data directory")
        return
    
    pt_files = glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True)
    
    if not pt_files:
        print(f"No .pt files found in {data_dir}")
        return
    
    print(f"Found {len(pt_files)} .pt files")
    
    # Examine a sample of files
    for file_path in pt_files[:5]:
        inspect_pt_file(file_path)
        print("-" * 50)

if __name__ == "__main__":
    main() 