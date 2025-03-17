#!/usr/bin/env python3
"""
Extract keys from a PyTorch file using string parsing.
"""

import sys
import os
import zipfile
import re

def extract_keys_from_torch_file(filepath):
    """Extract keys from a PyTorch file using string pattern matching."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        return False
    
    print(f"Examining PyTorch file: {filepath}")
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            # Find the data.pkl file
            data_pkl_files = [f for f in z.namelist() if 'data.pkl' in f]
            if not data_pkl_files:
                print("Error: No data.pkl file found in the archive")
                return False
            
            data_pkl_file = data_pkl_files[0]
            
            # Read the data.pkl file as bytes
            with z.open(data_pkl_file) as f:
                data_content = f.read()
                
                # Convert to string with safety
                try:
                    data_str = data_content.decode('latin-1')
                except:
                    data_str = str(data_content)
                
                # Look for key patterns
                print("\nPossible keys found (using string pattern matching):")
                
                # Method 1: Look for standard pickle dictionary patterns
                # Format: X[number]_[string]q
                key_pattern1 = r'X[0-9]+\\x00\\x00\\x00([a-zA-Z_]+)q'
                keys1 = re.findall(key_pattern1, str(data_content))
                
                # Method 2: Look for key patterns directly
                # This handles the case where keys are directly encoded
                data_ascii = "".join(chr(c) if 32 <= c <= 126 else " " for c in data_content)
                key_pattern2 = r'[a-zA-Z_]{3,20}(?:\s+[a-zA-Z0-9._]+){1,3}'
                potential_keys = re.findall(key_pattern2, data_ascii)
                
                # Filter potential keys
                keys2 = []
                known_patterns = ['mel_spec', 'chroma', 'onset_env', 'spectral_contrast', 'label', 'era', 'file']
                for item in potential_keys:
                    for pattern in known_patterns:
                        if pattern in item and pattern not in keys2:
                            keys2.append(pattern)
                
                # Combine and deduplicate keys
                all_keys = list(set(keys1 + keys2))
                
                if all_keys:
                    for key in all_keys:
                        print(f"  - {key}")
                else:
                    print("  No keys could be extracted using pattern matching")
                
                # Byte inspection approach
                print("\nRaw content inspection (looking for key strings):")
                found_keys = []
                for known_key in ['mel_spec', 'mel_spec_percussive', 'spectral_contrast', 'chroma', 'onset_env', 'label', 'era', 'file']:
                    key_bytes = known_key.encode('ascii')
                    if key_bytes in data_content:
                        found_keys.append(known_key)
                        print(f"  - Found '{known_key}'")
                    
                if not found_keys:
                    print("  No known keys found in raw content")
                    
                return True
    except Exception as e:
        print(f"Error examining PyTorch file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_keys.py <pytorch_file.pt>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not extract_keys_from_torch_file(file_path):
        print("Failed to extract keys from the PyTorch file")
        sys.exit(1) 