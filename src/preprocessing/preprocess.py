#!/usr/bin/env python3
"""
Script to run preprocessing with the enhanced audio features.
This will force reprocessing of the entire dataset.
"""

import argparse
import sys
import os

from ..config import get_training_config
from ..data.preprocessing import preprocess_dataset


def parse_arguments():
    """Parse command-line arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocess Grateful Dead audio snippets")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/Users/charlie/projects/gdguess/data/audsnippets-all",
        help="Directory containing audio snippets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for preprocessed files (defaults to [input-dir]/preprocessed)",
    )
    parser.add_argument(
        "--store-audio",
        action="store_true",
        help="Store audio files alongside preprocessed features",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit preprocessing to this many files (0 for no limit)",
    )
    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_arguments()
    
    # Process the dataset with the extended configuration
    config = get_training_config()
    config["input_dir"] = args.input_dir
    
    # Add output directory if specified
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Print configuration info
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir if args.output_dir else os.path.join(args.input_dir, 'preprocessed')}")
    print(f"Store audio: {args.store_audio}")
    if args.limit > 0:
        print(f"Limiting to {args.limit} files")
    
    # Preprocess the dataset
    preprocessed_dir = preprocess_dataset(
        config, 
        force_preprocess=True, 
        store_audio=args.store_audio, 
        limit=args.limit
    )
    print(f"Preprocessing completed! Data saved to: {preprocessed_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0) 