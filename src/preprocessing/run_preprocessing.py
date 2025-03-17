#!/usr/bin/env python3
"""
Unified preprocessing script that provides access to all preprocessing methods.
Serves as the main entry point for preprocessing in the Grateful Dead show dating model.
"""

import argparse
import os
import sys
import time

def parse_arguments():
    """Parse command-line arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(description="Unified preprocessing for Grateful Dead audio snippets")
    
    # Basic arguments
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
        "--method",
        type=str,
        choices=["turbo", "fast", "basic"],
        default="turbo",
        help="Preprocessing method to use (turbo=high performance, fast=balanced, basic=simple)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit preprocessing to this many files (0 for no limit)",
    )
    
    # GPU acceleration options
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    parser.add_argument(
        "--mps-optimize",
        action="store_true",
        help="Enable specific optimizations for Apple Silicon MPS",
    )
    
    # Performance tuning
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for file processing (affects memory usage)",
    )
    parser.add_argument(
        "--high-memory",
        action="store_true",
        help="Enable optimizations for high-memory systems (>64GB RAM)",
    )
    parser.add_argument(
        "--ultrafast",
        action="store_true",
        help="Use simplified processing for ultra-fast speed",
    )
    parser.add_argument(
        "--optimize-level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Optimization level (0=low, 1=medium, 2=aggressive)",
    )
    
    # Other options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of files even if they already exist",
    )
    parser.add_argument(
        "--store-audio",
        action="store_true",
        help="Store audio files alongside preprocessed features (basic method only)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for audio processing",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=15,
        help="Length of audio clips in seconds",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging except errors and progress bar",
    )
    
    return parser.parse_args()


def run_turbo_preprocessing(args):
    """Run preprocessing using the turbo (high-performance) method."""
    from .turboprocess import main as turbo_main
    
    # Map our arguments to turboprocess arguments
    sys.argv = [
        "turboprocess.py",
        f"--input-dir={args.input_dir}",
    ]
    
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    
    if args.limit > 0:
        sys.argv.append(f"--limit={args.limit}")
    
    if args.use_gpu:
        sys.argv.append("--use-gpu")
    
    if args.mps_optimize:
        sys.argv.append("--mps-optimize")
    
    if args.workers > 0:
        sys.argv.append(f"--workers={args.workers}")
    
    if args.batch_size != 64:
        sys.argv.append(f"--batch-size={args.batch_size}")
    
    if args.high_memory:
        sys.argv.append("--high-memory")
    
    if args.ultrafast:
        sys.argv.append("--ultrafast")
    
    if args.optimize_level != 1:
        sys.argv.append(f"--optimize-level={args.optimize_level}")
    
    if args.force:
        sys.argv.append("--force")
    
    if args.sample_rate != 16000:
        sys.argv.append(f"--sample-rate={args.sample_rate}")
    
    if args.clip_length != 15:
        sys.argv.append(f"--clip-length={args.clip_length}")
    
    if args.log_level != "info":
        sys.argv.append(f"--log-level={args.log_level}")
    
    if args.quiet:
        sys.argv.append("--quiet")
    
    # Run the turbo preprocessing
    print(f"Running high-performance (turbo) preprocessing with the following arguments:")
    print(f"  {' '.join(sys.argv[1:])}")
    turbo_main()


def run_fast_preprocessing(args):
    """Run preprocessing using the fast (balanced) method."""
    from .fastprocess import main as fast_main
    
    # Map our arguments to fastprocess arguments
    sys.argv = [
        "fastprocess.py",
        f"--input-dir={args.input_dir}",
    ]
    
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    
    if args.limit > 0:
        sys.argv.append(f"--limit={args.limit}")
    
    if args.workers > 0:
        sys.argv.append(f"--workers={args.workers}")
    
    if args.force:
        sys.argv.append("--force")
    
    if args.sample_rate != 16000:
        sys.argv.append(f"--sample-rate={args.sample_rate}")
    
    # Run the fast preprocessing
    print(f"Running balanced (fast) preprocessing with the following arguments:")
    print(f"  {' '.join(sys.argv[1:])}")
    fast_main()


def run_basic_preprocessing(args):
    """Run preprocessing using the basic (simple) method."""
    from .preprocess import main as basic_main
    
    # Map our arguments to preprocess arguments
    sys.argv = [
        "preprocess.py",
        f"--input-dir={args.input_dir}",
    ]
    
    if args.output_dir:
        sys.argv.append(f"--output-dir={args.output_dir}")
    
    if args.limit > 0:
        sys.argv.append(f"--limit={args.limit}")
    
    if args.store_audio:
        sys.argv.append("--store-audio")
    
    # Run the basic preprocessing
    print(f"Running basic preprocessing with the following arguments:")
    print(f"  {' '.join(sys.argv[1:])}")
    basic_main()


def main():
    """Main entry point for the unified preprocessing script."""
    args = parse_arguments()
    
    # Print settings
    print(f"Preprocessing method: {args.method}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir if args.output_dir else os.path.join(args.input_dir, 'preprocessed')}")
    
    start_time = time.time()
    
    # Run the selected preprocessing method
    if args.method == "turbo":
        run_turbo_preprocessing(args)
    elif args.method == "fast":
        run_fast_preprocessing(args)
    elif args.method == "basic":
        run_basic_preprocessing(args)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Preprocessing completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0) 