#!/usr/bin/env python3
"""
Utility script to examine a torch file as a zip archive and extract key information.
"""

import sys
import os
import zipfile
import pickle

def examine_torch_file(filepath):
    """Examine a PyTorch file as a zip archive"""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        return False
    
    print(f"Examining Torch file: {filepath}")
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            print("\nContents of the Torch archive:")
            for info in z.infolist():
                print(f"  - {info.filename} ({info.file_size} bytes)")
            
            # Try to extract data.pkl which contains the keys
            if any('data.pkl' in f for f in z.namelist()):
                data_pkl_file = next(f for f in z.namelist() if 'data.pkl' in f)
                print(f"\nExtracting and examining: {data_pkl_file}")
                
                try:
                    with z.open(data_pkl_file) as f:
                        data_pkl_content = f.read()
                        print(f"Raw content (first 100 bytes): {data_pkl_content[:100]}")
                        
                        # Try to unpickle this file
                        try:
                            data_dict = pickle.loads(data_pkl_content)
                            if isinstance(data_dict, dict):
                                print("\nKeys in the PyTorch file:")
                                for key in data_dict.keys():
                                    print(f"  - {key}")
                        except Exception as e:
                            print(f"Could not unpickle data.pkl: {e}")
                except Exception as e:
                    print(f"Error reading data.pkl: {e}")
            
            # Print byteorder
            byteorder_files = [f for f in z.namelist() if 'byteorder' in f]
            if byteorder_files:
                try:
                    with z.open(byteorder_files[0]) as f:
                        byteorder = f.read().decode('utf-8').strip()
                        print(f"\nByte order: {byteorder}")
                except Exception as e:
                    print(f"Error reading byteorder: {e}")
            
            # Print version
            version_files = [f for f in z.namelist() if 'version' in f]
            if version_files:
                try:
                    with z.open(version_files[0]) as f:
                        version = f.read().decode('utf-8').strip()
                        print(f"\nTorch version used: {version}")
                except Exception as e:
                    print(f"Error reading version: {e}")
                    
            return True
    except Exception as e:
        print(f"Error examining torch file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examine_torch_archive.py <torch_file.pt>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not examine_torch_file(file_path):
        print("Failed to examine the torch file")
        sys.exit(1) 