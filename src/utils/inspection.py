#!/usr/bin/env python3
"""
Utilities for inspecting PyTorch tensors, models, and checkpoint files.
"""

import os
import sys
import glob
import pickle
import zipfile
import struct
from typing import Optional, Dict, Any, List, Tuple
import torch


def inspect_pt_file(file_path: str, use_torch: bool = True) -> Dict[str, Any]:
    """
    Inspect a PyTorch .pt file structure and contents.
    
    Args:
        file_path: Path to the .pt file
        use_torch: Whether to use torch.load or raw pickle module
        
    Returns:
        Dictionary with file information
    """
    result = {
        "file_path": file_path,
        "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
        "content_type": None,
        "keys": None,
        "tensors": [],
        "error": None
    }
    
    try:
        if use_torch:
            # Use PyTorch to load the file (preferred method)
            data = torch.load(file_path, map_location='cpu')
            result["content_type"] = type(data).__name__
            
            # Analyze content structure
            if isinstance(data, dict):
                result["keys"] = list(data.keys())
                
                # Collect tensor information
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        tensor_info = {
                            "name": k,
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "device": str(v.device),
                            "min": float(v.min()) if v.numel() > 0 else None,
                            "max": float(v.max()) if v.numel() > 0 else None,
                            "has_nan": bool(torch.isnan(v).any()) if v.numel() > 0 else None,
                            "has_inf": bool(torch.isinf(v).any()) if v.numel() > 0 else None
                        }
                        result["tensors"].append(tensor_info)
            
            # Check if it's a model state dict
            if isinstance(data, dict) and any(k.endswith('weight') or k.endswith('bias') for k in data.keys()):
                result["is_model_state"] = True
                result["num_parameters"] = len(data)
                total_params = sum(p.numel() for p in data.values() if isinstance(p, torch.Tensor))
                result["total_parameters"] = total_params
                
        else:
            # Use pickle and zipfile modules for raw inspection (no PyTorch required)
            if zipfile.is_zipfile(file_path):
                result["content_type"] = "PyTorch zip archive"
                
                with zipfile.ZipFile(file_path, 'r') as z:
                    result["archive_files"] = z.namelist()
                    
                    # Look for data.pkl for older PyTorch versions
                    if 'data.pkl' in z.namelist():
                        with z.open('data.pkl', 'r') as f:
                            data = pickle.load(f)
                            if isinstance(data, dict):
                                result["keys"] = list(data.keys())
            else:
                # Try to open as a plain pickle file
                with open(file_path, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        result["content_type"] = "Pickle file"
                        if isinstance(data, dict):
                            result["keys"] = list(data.keys())
                    except:
                        result["content_type"] = "Unknown binary format"
                        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def print_pt_file_info(file_path: str, use_torch: bool = True) -> None:
    """
    Print information about a PyTorch .pt file in a readable format.
    
    Args:
        file_path: Path to the .pt file
        use_torch: Whether to use torch.load or raw pickle module
    """
    info = inspect_pt_file(file_path, use_torch)
    
    print(f"==== File: {info['file_path']} ====")
    print(f"Size: {info['file_size_mb']:.2f} MB")
    
    if info['error']:
        print(f"Error: {info['error']}")
        return
        
    print(f"Content type: {info['content_type']}")
    
    if info.get('is_model_state'):
        print(f"Model state dict with {info['num_parameters']} parameters")
        print(f"Total parameters: {info['total_parameters']:,}")
    
    if info['keys']:
        print("\nKeys:")
        for key in info['keys']:
            print(f"  - {key}")
    
    if info.get('tensors'):
        print("\nTensors:")
        for t in info['tensors']:
            print(f"  - {t['name']}: shape={t['shape']}, dtype={t['dtype']}")
            if t['has_nan'] or t['has_inf']:
                issues = []
                if t['has_nan']: issues.append("has NaN")
                if t['has_inf']: issues.append("has Inf")
                print(f"    WARNING: {', '.join(issues)}")
    
    if info.get('archive_files'):
        print("\nArchive contents:")
        for f in info['archive_files']:
            print(f"  - {f}")


def inspect_directory(directory: str, pattern: str = "*.pt", 
                      recursive: bool = True, use_torch: bool = True) -> None:
    """
    Inspect all PyTorch files in a directory.
    
    Args:
        directory: Directory to search for files
        pattern: File pattern to match
        recursive: Whether to search recursively
        use_torch: Whether to use torch.load or raw pickle module
    """
    if recursive:
        search_pattern = os.path.join(directory, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files matching '{pattern}' found in {directory}")
        return
        
    print(f"Found {len(files)} {pattern} files")
    
    for file_path in files:
        print_pt_file_info(file_path, use_torch)
        print("\n" + "-"*60 + "\n")


def main():
    """Command-line interface for inspection utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect PyTorch files and checkpoints")
    parser.add_argument("path", help="File or directory to inspect")
    parser.add_argument("--pattern", default="*.pt", help="File pattern to match (if directory)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search recursively")
    parser.add_argument("--no-torch", action="store_true", help="Don't use torch.load (pure Python inspection)")
    
    args = parser.parse_args()
    
    # Determine if path is a file or directory
    if os.path.isfile(args.path):
        print_pt_file_info(args.path, not args.no_torch)
    elif os.path.isdir(args.path):
        inspect_directory(args.path, args.pattern, not args.no_recursive, not args.no_torch)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()