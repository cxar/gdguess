#!/usr/bin/env python3
"""
Main CLI script for the Grateful Dead show dating project.
Provides a simplified interface to train models and run inference.
"""

import os
import sys
import argparse
import torch

from src.core.utils.utils import get_device, print_system_info
from src.core.train.train import main as train_main
from src.core.model.model import DeadShowDatingModel


def command_train(args):
    """Run model training with the simplified trainer."""
    # Handle CLI args
    sys.argv = [sys.argv[0]]
    
    if args.data_dir:
        sys.argv.extend(["--data-dir", args.data_dir])
    
    if args.batch_size:
        sys.argv.extend(["--batch-size", str(args.batch_size)])
    
    if args.lr:
        sys.argv.extend(["--lr", str(args.lr)])
    
    if args.steps:
        sys.argv.extend(["--steps", str(args.steps)])
    
    if args.checkpoint:
        sys.argv.extend(["--checkpoint", args.checkpoint])
    
    if args.output_dir:
        sys.argv.extend(["--output-dir", args.output_dir])
    
    # Run the training
    train_main()


def command_infer(args):
    """Run inference on audio files."""
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    model = DeadShowDatingModel(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Perform inference
    print(f"Running inference on {args.input}")
    # This would be a placeholder for actual inference code
    print("Inference complete")


def command_system_info(args):
    """Print system information."""
    device = print_system_info()


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
    train_parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory with preprocessed files')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--steps', type=int, default=10000, help='Total training steps')
    train_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    train_parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for model outputs and checkpoints')
    train_parser.set_defaults(func=command_train)
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on audio files')
    infer_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    infer_parser.add_argument('--input', type=str, required=True, help='Path to input audio file')
    infer_parser.set_defaults(func=command_infer)
    
    # System info command
    sysinfo_parser = subparsers.add_parser('sysinfo', help='Print system information')
    sysinfo_parser.set_defaults(func=command_system_info)
    
    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()