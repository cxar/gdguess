#!/usr/bin/env python3
"""
Inspect a PyTorch file using the pickle module.
"""

import sys
import os
import pickle
import struct

def safe_unpickle(filename):
    """Try to examine a PyTorch pickle file without loading it completely."""
    with open(filename, 'rb') as f:
        # Read the first few bytes to check the file format
        magic_number = f.read(2)
        f.seek(0)
        
        # Check if it's likely a PyTorch file
        print(f"First bytes (hex): {magic_number.hex()}")
        
        try:
            # Try to get the protocol version
            protocol = struct.unpack('B', f.read(1))[0]
            print(f"Pickle protocol version: {protocol}")
            f.seek(0)
            
            # Try to extract some information without full deserialization
            obj = None
            try:
                unpickler = pickle.Unpickler(f)
                obj = unpickler.load()
                print("\nSuccessfully unpickled the file!")
            except Exception as e:
                print(f"Could not fully unpickle: {e}")
                return False
            
            if obj is not None:
                # Check if it's a dictionary
                if isinstance(obj, dict):
                    print("\nKeys in the file:")
                    for key in obj.keys():
                        print(f"  - {key}")
                    
                    print("\nData types:")
                    for key, value in obj.items():
                        if hasattr(value, 'shape'):
                            print(f"  - {key}: {type(value).__name__} with shape {value.shape}")
                        else:
                            print(f"  - {key}: {type(value).__name__}")
                else:
                    print(f"\nRoot object type: {type(obj).__name__}")
            
            return True
            
        except Exception as e:
            print(f"Error during inspection: {e}")
            return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pickle.py <pytorch_file.pt>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
        
    print(f"Inspecting file: {file_path}")
    if not safe_unpickle(file_path):
        print("Failed to inspect the file.")
        sys.exit(1) 