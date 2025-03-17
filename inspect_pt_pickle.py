#!/usr/bin/env python
"""
Inspect the structure of pt files using pickle module (no torch dependency).
"""

import os
import sys
import glob
import pickle
import zipfile
import struct

def inspect_pt_file(file_path):
    """Inspect a single pt file using pickle."""
    print(f"Examining {file_path}")
    
    try:
        # PyTorch saves files as zip archives
        if zipfile.is_zipfile(file_path):
            print("File is a zip archive (PyTorch storage format)")
            with zipfile.ZipFile(file_path, 'r') as z:
                print(f"Archive contents: {z.namelist()}")
                
                # Look for data.pkl which contains the actual tensor data
                if 'data.pkl' in z.namelist():
                    with z.open('data.pkl') as f:
                        try:
                            data = pickle.load(f)
                            print(f"Keys in data.pkl: {list(data.keys())}")
                            
                            for k, v in data.items():
                                print(f"{k}: {type(v)}", end="")
                                # Check if it has a shape attribute (tensor-like)
                                if hasattr(v, 'shape'):
                                    print(f" shape={v.shape}", end="")
                                print("")
                            
                        except Exception as e:
                            print(f"Error unpickling data.pkl: {e}")
        else:
            # Try to open as a standard pickle file
            with open(file_path, 'rb') as f:
                try:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        print(f"Keys in file: {list(data.keys())}")
                        
                        for k, v in data.items():
                            print(f"{k}: {type(v)}", end="")
                            # Check if it has a shape attribute (tensor-like)
                            if hasattr(v, 'shape'):
                                print(f" shape={v.shape}", end="")
                            print("")
                    else:
                        print(f"File contains {type(data)}, not a dictionary")
                except Exception as e:
                    print(f"Error unpickling file: {e}")
                        
        return True
    except Exception as e:
        print(f"Error inspecting file: {e}")
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