#!/usr/bin/env python3
"""
Minimal test script to verify the model training pipeline works.
Use this for quick testing before running a full training session.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add the src directory to the path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, src_dir)

# Import the unified training module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_training import create_dataloaders, OptimizedTrainer
from src.config import Config
from src.utils.device_utils import print_system_info


def run_test(data_dir, batch_size=8, steps=10, device="auto"):
    """Run a minimal test of the training pipeline."""
    print("Running minimal training test...")
    
    # Print system info and get device
    print_system_info()
    
    # Set up device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
        
    print(f"Using device: {device}")
    
    # Set up a minimal config
    config = Config()
    config.data_dir = data_dir
    config.batch_size = batch_size
    config.total_training_steps = steps
    config.output_dir = os.path.join(os.path.dirname(__file__), "output", "test")
    # Add required attributes that might be missing in the Config class
    config.aggressive_memory = device.type == 'mps'  # Enable for MPS by default
    config.use_mixed_precision = False
    config.use_augmentation = False
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    
    # Create dataloaders with a limited number of samples
    train_loader, val_loader = create_dataloaders(
        config, device, max_samples=batch_size * 3
    )
    
    # Create trainer
    trainer = OptimizedTrainer(
        config, train_loader, val_loader, device, 
        tensorboard_dir=os.path.join(config.output_dir, "runs")
    )
    
    # Run training for a few steps
    trainer.train(max_steps=steps)
    
    print("Test completed!")
    print(f"Check {config.output_dir} for outputs.")


def main():
    """Parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Run a minimal training test")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to train")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda, mps, cpu, auto)")
    
    args = parser.parse_args()
    run_test(args.data_dir, args.batch_size, args.steps, args.device)


if __name__ == "__main__":
    main()