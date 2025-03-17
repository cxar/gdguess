#!/usr/bin/env python3
"""
Main entry point for training the Grateful Dead show dating model.
"""

import argparse
import sys
import torch
import platform

from config import get_training_config
from training.train import train_model


def parse_arguments():
    """Parse command line arguments."""
    # Detect available devices for default and choices
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    # Set default device based on availability
    if cuda_available:
        default_device = "cuda"
    elif mps_available:
        default_device = "mps"
    else:
        default_device = "cpu"
        
    # Create device choices list
    device_choices = ["auto", "cpu"]
    if cuda_available:
        device_choices.append("cuda")
    if mps_available:
        device_choices.append("mps")
    
    parser = argparse.ArgumentParser(
        description="Train a model to date Grateful Dead shows from audio"
    )

    parser.add_argument(
        "--input-dir", type=str, help="Directory containing audio snippets"
    )

    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )

    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")

    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--steps", type=int, default=180000, help="Total training steps"
    )

    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Run learning rate finder before training",
    )

    parser.add_argument(
        "--no-early-stopping", action="store_true", help="Disable early stopping"
    )

    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )

    parser.add_argument(
        "--use-jit",
        action="store_true",
        help="Use JIT compilation for model (CUDA only)",
    )

    parser.add_argument(
        "--no-augmentation", action="store_true", help="Disable audio augmentation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=device_choices,
        help=f"Device to use for training: {', '.join(device_choices)}"
    )

    parser.add_argument(
        "--disable-device-consistency", 
        action="store_true",
        help="Disable device consistency checks (may help with some torch.compile issues)"
    )

    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Enable preprocessing (disabled by default)"
    )
    
    # MPS specific options (relevant only on Apple Silicon Macs)
    if mps_available:
        parser.add_argument(
            "--mps-fallback",
            action="store_true",
            help="Enable CPU fallback for unsupported MPS operations"
        )

        parser.add_argument(
            "--mps-optimize",
            action="store_true",
            help="Enable MPS-specific optimizations"
        )
        
        parser.add_argument(
            "--force-cpu",
            action="store_true",
            help="Force CPU usage even on Apple Silicon Mac"
        )

    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_arguments()

    # Create config with command line overrides
    config_overrides = {}
    
    # Preprocessing is disabled by default
    config_overrides["disable_preprocessing"] = not args.preprocess

    if args.input_dir:
        config_overrides["data_dir"] = args.input_dir

    if args.batch_size:
        config_overrides["training.batch_size"] = args.batch_size

    if args.lr:
        config_overrides["training.learning_rate"] = args.lr

    if args.resume:
        config_overrides["resume_checkpoint"] = args.resume

    if args.steps:
        config_overrides["total_training_steps"] = args.steps

    if args.find_lr:
        config_overrides["run_lr_finder"] = True

    if args.no_early_stopping:
        config_overrides["use_early_stopping"] = False

    if args.patience:
        config_overrides["patience"] = args.patience

    if args.use_jit:
        config_overrides["use_jit"] = True

    if args.no_augmentation:
        config_overrides["use_augmentation"] = False

    if args.device:
        config_overrides["device"] = args.device
        
        # Handle forced CPU usage on Mac
        if hasattr(args, 'force_cpu') and args.force_cpu:
            config_overrides["device"] = "cpu"

    if args.disable_device_consistency:
        config_overrides["disable_device_consistency"] = True
        
    # MPS specific options
    if hasattr(args, 'mps_fallback') and args.mps_fallback:
        config_overrides["mps_fallback_to_cpu"] = True
        
    if hasattr(args, 'mps_optimize') and args.mps_optimize:
        config_overrides["mps_enable_all_optimizations"] = True
    
    # Print platform information
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if config_overrides.get("device", "auto") == "auto":
        print("Auto-detecting best available device...")
    else:
        print(f"Using device: {config_overrides.get('device')}")

    # Get configuration
    config = get_training_config(**config_overrides)

    # Train model
    train_model(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0)
