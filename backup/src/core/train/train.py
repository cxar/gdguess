#!/usr/bin/env python3
"""
Simplified training script for the Grateful Dead show dating model.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.core.data.dataset import PreprocessedDeadShowDataset, SimpleCollate
from src.core.train.trainer import SimpleTrainer


class SimpleConfig:
    """Simple configuration class for training."""
    def __init__(self):
        self.lr = 0.001
        self.steps = 10000
        self.batch_size = 16
        self.validation_interval = 100
        self.checkpoint_interval = 500
        self.log_interval = 10
        self.patience = 10
        self.early_stopping = True
        self.output_dir = "output"


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train the Grateful Dead show dating model")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create config
    config = SimpleConfig()
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.steps = args.steps
    config.patience = args.patience
    
    # Create datasets and dataloaders
    print("Creating datasets...")
    train_dataset = PreprocessedDeadShowDataset(
        preprocessed_dir=args.data_dir,
        device=None
    )
    
    # Split dataset 90/10 for training/validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Use random_split to create train/val datasets
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=SimpleCollate(device=None),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=SimpleCollate(device=None),
        pin_memory=True
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = SimpleTrainer(config, train_loader, val_loader, device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()