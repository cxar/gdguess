#!/usr/bin/env python3
"""
Main entry point for training the Grateful Dead show dating model.
"""

import argparse
import sys

from config import get_training_config
from training.train import train_model


def parse_arguments():
    """Parse command line arguments."""
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

    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_arguments()

    # Create config with command line overrides
    config_overrides = {}

    if args.input_dir:
        config_overrides["input_dir"] = args.input_dir

    if args.batch_size:
        config_overrides["batch_size"] = args.batch_size

    if args.lr:
        config_overrides["initial_lr"] = args.lr

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
