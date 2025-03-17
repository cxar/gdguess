#!/usr/bin/env python3
"""
Utility script to inspect preprocessed files.
"""

import sys
import os
import torch
from pprint import pprint

def inspect_preprocessed_file(file_path):
    """Inspect a preprocessed PyTorch file and print its structure."""
    print(f"Inspecting file: {file_path}")
    
    try:
        # Load the preprocessed file
        data = torch.load(file_path)
        
        print("\nKeys in the file:")
        print(list(data.keys()))
        
        print("\nData types:")
        types = {k: type(v).__name__ for k, v in data.items()}
        pprint(types)
        
        print("\nShapes:")
        shapes = {}
        for k, v in data.items():
            if hasattr(v, 'shape'):
                shapes[k] = v.shape
            elif isinstance(v, list):
                shapes[k] = f"List of length {len(v)}"
            else:
                shapes[k] = "No shape attribute"
        pprint(shapes)
        
        print("\nFirst few values (where applicable):")
        for k, v in data.items():
            if hasattr(v, 'shape'):
                if v.numel() > 0:
                    print(f"{k}: {v.flatten()[:5]} ...")
            elif isinstance(v, list) and len(v) > 0:
                print(f"{k}: {v[:5]} ...")
            else:
                print(f"{k}: {v}")
        
        return True
    except Exception as e:
        print(f"Error inspecting file: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_preprocessed.py <preprocessed_file.pt>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
        
    inspect_preprocessed_file(file_path) 