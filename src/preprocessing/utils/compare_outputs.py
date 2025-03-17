#!/usr/bin/env python3
"""
Compare outputs from the original preprocessing script and the turbo-optimized version
to ensure compatibility with the model.
"""

import os
import glob
import torch
import numpy as np
from pathlib import Path
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare outputs from different preprocessing scripts")
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Directory containing original preprocessed files",
    )
    parser.add_argument(
        "--turbo",
        type=str,
        required=True,
        help="Directory containing turbo preprocessed files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of files to compare",
    )
    return parser.parse_args()


def load_preprocessed_files(directory, limit):
    """Load preprocessed files from the specified directory."""
    files = glob.glob(os.path.join(directory, "*.pt"))
    if limit > 0:
        files = files[:limit]
    
    loaded_data = []
    for file in files:
        try:
            data = torch.load(file)
            loaded_data.append((Path(file).name, data))
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return loaded_data


def compare_tensor_features(feature_name, tensor1, tensor2):
    """Compare two tensors and return detailed information about differences."""
    if tensor1.shape != tensor2.shape:
        return {
            "compatible": False,
            "reason": f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}",
            "diff_percent": 100.0  # Complete mismatch
        }
    
    # Convert to float32 for comparison to avoid precision differences
    if tensor1.dtype != torch.float32:
        tensor1 = tensor1.float()
    if tensor2.dtype != torch.float32:
        tensor2 = tensor2.float()
    
    # Calculate differences
    if tensor1.dim() > 0:
        abs_diff = (tensor1 - tensor2).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Calculate percentage difference relative to the range of values
        max_val = max(tensor1.max().item(), tensor2.max().item())
        min_val = min(tensor1.min().item(), tensor2.min().item())
        value_range = max(1e-6, max_val - min_val)  # Avoid division by zero
        
        diff_percent = (mean_diff / value_range) * 100
        
        # Determine if differences are acceptable
        compatible = diff_percent < 5.0  # 5% difference threshold
        
        return {
            "compatible": compatible,
            "shape": tensor1.shape,
            "dtype": str(tensor1.dtype),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "diff_percent": diff_percent,
            "range": [min_val, max_val]
        }
    else:
        # Scalar tensor
        diff = abs(tensor1.item() - tensor2.item())
        return {
            "compatible": diff < 1e-3,  # Small threshold for scalar values
            "shape": "scalar",
            "dtype": str(tensor1.dtype),
            "diff": diff
        }


def compare_datasets(original_data, turbo_data):
    """Compare datasets from different preprocessing methods."""
    print(f"\nComparing {len(original_data)} files from original and turbo preprocessing...")
    
    # Check if the same number of files were generated
    if len(original_data) != len(turbo_data):
        print(f"WARNING: Number of files differs: Original: {len(original_data)}, Turbo: {len(turbo_data)}")
    
    # We'll analyze the features across all files
    feature_stats = {}
    
    for (orig_name, orig_features), (turbo_name, turbo_features) in zip(original_data, turbo_data):
        print(f"\nComparing file: {orig_name}")
        
        # Check if both files have the same set of features
        orig_keys = set(orig_features.keys())
        turbo_keys = set(turbo_features.keys())
        
        if orig_keys != turbo_keys:
            missing_in_turbo = orig_keys - turbo_keys
            extra_in_turbo = turbo_keys - orig_keys
            
            if missing_in_turbo:
                print(f"  WARNING: Features missing in turbo version: {missing_in_turbo}")
            if extra_in_turbo:
                print(f"  WARNING: Extra features in turbo version: {extra_in_turbo}")
        
        # Compare common features
        common_keys = orig_keys.intersection(turbo_keys)
        
        for key in common_keys:
            if key not in feature_stats:
                feature_stats[key] = {
                    "num_compared": 0,
                    "num_compatible": 0,
                    "max_diff_percent": 0,
                    "avg_diff_percent": 0,
                    "total_diff_percent": 0
                }
            
            # Skip non-tensor features
            if not isinstance(orig_features[key], torch.Tensor) or not isinstance(turbo_features[key], torch.Tensor):
                print(f"  Skipping non-tensor feature: {key}")
                continue
            
            # Compare tensors
            comparison = compare_tensor_features(key, orig_features[key], turbo_features[key])
            feature_stats[key]["num_compared"] += 1
            
            if comparison.get("compatible", False):
                feature_stats[key]["num_compatible"] += 1
            
            if "diff_percent" in comparison:
                feature_stats[key]["total_diff_percent"] += comparison["diff_percent"]
                feature_stats[key]["max_diff_percent"] = max(
                    feature_stats[key]["max_diff_percent"], 
                    comparison["diff_percent"]
                )
            
            # Print detailed comparison for this feature
            if "diff_percent" in comparison:
                print(f"  {key}: {'✓' if comparison['compatible'] else '✗'} " 
                      f"Shape: {comparison.get('shape', 'N/A')}, "
                      f"Diff: {comparison['diff_percent']:.2f}%")
            else:
                print(f"  {key}: {'✓' if comparison['compatible'] else '✗'} "
                      f"Shape: {comparison.get('shape', 'N/A')}, "
                      f"Diff: {comparison.get('diff', 'N/A')}")
    
    # Calculate averages and print summary
    print("\n===== COMPATIBILITY SUMMARY =====")
    for key, stats in feature_stats.items():
        if stats["num_compared"] > 0:
            stats["avg_diff_percent"] = stats["total_diff_percent"] / stats["num_compared"]
            compatibility_rate = (stats["num_compatible"] / stats["num_compared"]) * 100
            
            print(f"{key}:")
            print(f"  Compatibility: {compatibility_rate:.1f}% ({stats['num_compatible']}/{stats['num_compared']})")
            print(f"  Average difference: {stats['avg_diff_percent']:.2f}%")
            print(f"  Maximum difference: {stats['max_diff_percent']:.2f}%")
            print(f"  Verdict: {'COMPATIBLE' if compatibility_rate >= 95 else 'NOT COMPATIBLE'}")
            print()


def main():
    args = parse_arguments()
    
    # Load both datasets
    print(f"Loading original preprocessed files from: {args.original}")
    original_data = load_preprocessed_files(args.original, args.limit)
    print(f"Found {len(original_data)} files")
    
    print(f"\nLoading turbo preprocessed files from: {args.turbo}")
    turbo_data = load_preprocessed_files(args.turbo, args.limit)
    print(f"Found {len(turbo_data)} files")
    
    # Compare datasets
    if original_data and turbo_data:
        compare_datasets(original_data, turbo_data)
    else:
        print("Error: No files to compare")


if __name__ == "__main__":
    main() 