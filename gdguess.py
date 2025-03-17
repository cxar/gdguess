#!/usr/bin/env python3
"""
Command-line tools for the Grateful Dead show dating project.
"""

import os
import sys
import argparse
import torch
import subprocess
from pathlib import Path

# Add the src directory to the path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, src_dir)

from config import Config
from utils.device_utils import print_system_info
from utils.inspection import inspect_pt_file, print_pt_file_info, inspect_directory


def command_train(args):
    """Run model training."""
    from training.run_training import main as run_training_main
    
    # Convert command-line args to sys.argv format
    sys.argv = [sys.argv[0]]
    
    if args.config:
        sys.argv.extend(["--config", args.config])
    
    if args.batch_size:
        sys.argv.extend(["--batch-size", str(args.batch_size)])
    
    if args.lr:
        sys.argv.extend(["--lr", str(args.lr)])
    
    if args.steps:
        sys.argv.extend(["--steps", str(args.steps)])
    
    if args.checkpoint:
        sys.argv.extend(["--checkpoint", args.checkpoint])
    
    if args.data_dir:
        sys.argv.extend(["--data-dir", args.data_dir])
    
    if args.device:
        sys.argv.extend(["--device", args.device])
    
    # Run training
    run_training_main()


def command_infer(args):
    """Run inference on audio files."""
    from src.infer import main as infer_main
    
    # Convert command-line args to sys.argv format
    sys.argv = [sys.argv[0]]
    
    if args.model:
        sys.argv.extend(["--model", args.model])
    
    if args.input:
        sys.argv.extend(["--input", args.input])
    
    if args.output:
        sys.argv.extend(["--output", args.output])
    
    if args.device:
        sys.argv.extend(["--device", args.device])
    
    if args.verbose:
        sys.argv.append("--verbose")
    
    # Run inference
    infer_main()


def command_inspect(args):
    """Inspect PyTorch files and checkpoints."""
    if os.path.isfile(args.path):
        print_pt_file_info(args.path, use_torch=not args.no_torch)
    elif os.path.isdir(args.path):
        inspect_directory(args.path, args.pattern, not args.no_recursive, not args.no_torch)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)


def command_system_info(args):
    """Print system information."""
    device = print_system_info()
    
    if args.test_device:
        from utils.device_utils import test_basic_operations
        test_basic_operations(device)
    
    if args.benchmark:
        from utils.device_utils import run_benchmark
        run_benchmark(max_size=args.benchmark_size, step=500, device=device)


def command_test(args):
    """Run tests."""
    # Build command arguments
    test_args = []
    
    if args.transformer:
        test_args.append("--transformer")
    
    if args.model:
        test_args.append("--model")
    
    if args.unittest:
        test_args.append("--unittest")
    
    if args.all:
        test_args.append("--all")
    
    # Run the test script in a subprocess to ensure proper environment
    test_script = os.path.join(os.path.dirname(__file__), "tests", "run_tests.py")
    cmd = [sys.executable, test_script] + test_args
    
    return subprocess.call(cmd)


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description='Grateful Dead Show Dating Project',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--batch-size', type=int, help='Batch size for training')
    train_parser.add_argument('--lr', type=float, help='Initial learning rate')
    train_parser.add_argument('--steps', type=int, help='Total training steps')
    train_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    train_parser.add_argument('--data-dir', type=str, help='Path to data directory')
    train_parser.add_argument('--device', type=str, help='Device to use (cuda, mps, cpu, auto)')
    train_parser.set_defaults(func=command_train)
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on audio files')
    infer_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    infer_parser.add_argument('--input', type=str, required=True, help='Path to input audio file or directory')
    infer_parser.add_argument('--output', type=str, help='Path to output directory')
    infer_parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda, mps, cpu, auto)')
    infer_parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    infer_parser.set_defaults(func=command_infer)
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect PyTorch files and checkpoints')
    inspect_parser.add_argument('path', help='File or directory to inspect')
    inspect_parser.add_argument('--pattern', default='*.pt', help='File pattern to match (if directory)')
    inspect_parser.add_argument('--no-recursive', action='store_true', help="Don't search recursively")
    inspect_parser.add_argument('--no-torch', action='store_true', help="Don't use torch.load (pure Python inspection)")
    inspect_parser.set_defaults(func=command_inspect)
    
    # System info command
    sysinfo_parser = subparsers.add_parser('sysinfo', help='Print system information')
    sysinfo_parser.add_argument('--test-device', action='store_true', help='Run basic device tests')
    sysinfo_parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    sysinfo_parser.add_argument('--benchmark-size', type=int, default=2000, help='Maximum matrix size for benchmark')
    sysinfo_parser.set_defaults(func=command_system_info)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--transformer', action='store_true', help='Run transformer tests')
    test_parser.add_argument('--model', action='store_true', help='Run model tests')
    test_parser.add_argument('--unittest', action='store_true', help='Run unittest-based tests')
    test_parser.add_argument('--all', action='store_true', help='Run all tests')
    test_parser.set_defaults(func=command_test)
    
    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()