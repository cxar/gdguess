#!/usr/bin/env python3
"""
Simplified command-line interface for the Grateful Dead show dating project.
Optimized for clarity and maximum compatibility using CPU-only operations.
"""

import os
import sys
import argparse
import subprocess

def command_train(args):
    """Run model training."""
    # Create command for the training script
    train_script = os.path.join(os.path.dirname(__file__), "simplified", "train.py")
    
    cmd = [sys.executable, train_script]
    
    # Add all arguments
    cmd.extend(["--data-dir", args.data_dir])
    cmd.extend(["--output-dir", args.output_dir])
    cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(["--lr", str(args.lr)])
    cmd.extend(["--epochs", str(args.epochs)])
    
    # Pass device parameter to allow MPS
    cmd.extend(["--device", args.device])
    print(f"Using device: {args.device}")
        
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
        
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
        
    if args.save_interval:
        cmd.extend(["--save-interval", str(args.save_interval)])
        
    if args.validation_interval:
        cmd.extend(["--validation-interval", str(args.validation_interval)])
        
    if args.early_stopping:
        cmd.append("--early-stopping")
        
    if args.patience:
        cmd.extend(["--patience", str(args.patience)])
        
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
        
    if args.tensorboard:
        cmd.append("--tensorboard")
        
    if args.tensorboard_dir:
        cmd.extend(["--tensorboard-dir", args.tensorboard_dir])
        
    if args.load_to_device:
        cmd.append("--load-to-device")
    
    # Run the training script
    print(f"Running training command: {' '.join(cmd)}")
    return subprocess.call(cmd)

def command_infer(args):
    """Run inference."""
    # Create command for the inference script
    infer_script = os.path.join(os.path.dirname(__file__), "simplified", "infer.py")
    
    cmd = [sys.executable, infer_script]
    
    # Add required arguments
    cmd.extend(["--model", args.model])
    cmd.extend(["--input", args.input])
    
    # Add optional arguments
    if args.output:
        cmd.extend(["--output", args.output])
        
    # Pass device parameter to allow MPS
    cmd.extend(["--device", args.device])
    print(f"Using device: {args.device}")
        
    if args.verbose:
        cmd.append("--verbose")
    
    # Run the inference script
    print(f"Running inference command: {' '.join(cmd)}")
    return subprocess.call(cmd)

def command_system_info(args):
    """Print system information."""
    from simplified.utils.device_utils import print_system_info, benchmark_device, get_device
    
    # Print system info
    device = print_system_info()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_device(device, [500, 1000, 2000] if args.benchmark_size is None else 
                                 list(range(500, args.benchmark_size + 1, 500)))
    
    return 0

def main():
    """Parse command-line arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(
        description='Simplified Grateful Dead Show Dating Project',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    train_parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    train_parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--device', type=str, default='auto', help='Device (cuda, mps, cpu, auto)')
    train_parser.add_argument('--max-samples', type=int, help='Maximum number of samples to use')
    train_parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers')
    train_parser.add_argument('--save-interval', type=int, default=500, help='Save checkpoint every N steps')
    train_parser.add_argument('--validation-interval', type=int, default=1, help='Validate every N epochs')
    train_parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping')
    train_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    train_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    train_parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    train_parser.add_argument('--tensorboard-dir', type=str, default='./runs', help='TensorBoard log directory')
# Removed load-to-device parameter as data is always loaded to device for MPS compatibility
    train_parser.set_defaults(func=command_train)
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    infer_parser.add_argument('--input', type=str, required=True, help='Path to input file or directory')
    infer_parser.add_argument('--output', type=str, help='Path to output file or directory')
    infer_parser.add_argument('--device', type=str, default='auto', help='Device (cuda, mps, cpu, auto)')
    infer_parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    infer_parser.set_defaults(func=command_infer)
    
    # System info command
    sysinfo_parser = subparsers.add_parser('sysinfo', help='Print system information')
    sysinfo_parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    sysinfo_parser.add_argument('--benchmark-size', type=int, help='Maximum matrix size for benchmark')
    sysinfo_parser.set_defaults(func=command_system_info)
    
    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()