#!/usr/bin/env python3
"""
Script to run training for the Grateful Dead show dating model.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import PreprocessedDeadShowDataset, optimized_collate_fn
from data.preprocessing import preprocess_dataset
from training.trainer import Trainer


def get_device(config: Config) -> torch.device:
    """
    Get the appropriate device for training based on availability and configuration.
    
    Args:
        config: Configuration object with device preferences
        
    Returns:
        PyTorch device object
    """
    # Check for available devices
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    if config.device == "auto":
        # Automatically select the best available device
        if cuda_available:
            return torch.device("cuda")
        elif mps_available:
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        # Use the specified device if available
        requested_device = config.device
        if requested_device == "cuda" and not cuda_available:
            print("CUDA requested but not available.")
            if mps_available:
                print("Falling back to MPS (Apple Silicon)")
                return torch.device("mps")
            else:
                print("Falling back to CPU")
                return torch.device("cpu")
        elif requested_device == "mps" and not mps_available:
            print("MPS requested but not available.")
            if cuda_available:
                print("Falling back to CUDA")
                return torch.device("cuda")
            else:
                print("Falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device(requested_device)


def create_dataloaders(config: Config, device: torch.device):
    """
    Create and return train and validation DataLoader objects.
    
    Args:
        config: Configuration object
        device: Device to use for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Preprocess data if needed
    if not config.disable_preprocessing:
        if not os.path.exists(os.path.join(config.data_dir, "preprocessed")):
            print("Preprocessing dataset...")
            preprocess_dataset(config)
        else:
            print("Using existing preprocessed dataset")
    
    # Set up dataset
    data_dir = os.path.join(config.data_dir, "preprocessed")
    if not os.path.exists(data_dir) and not config.disable_preprocessing:
        print(f"Error: Preprocessed data directory not found at {data_dir}")
        print("Please run preprocessing or check data directory path.")
        sys.exit(1)
    elif not os.path.exists(data_dir):
        # Try to find another data source
        if os.path.exists(config.data_dir):
            data_dir = config.data_dir
            print(f"Using non-preprocessed data directory at {data_dir}")
        else:
            print(f"Error: Data directory not found at {config.data_dir}")
            print("Please check data directory path.")
            sys.exit(1)
    
    # Load dataset
    full_dataset = PreprocessedDeadShowDataset(
        data_dir,
        augment=config.use_augmentation,
        target_sr=config.target_sr
    )
    
    # Split dataset
    train_size = int(len(full_dataset) * config.train_split)
    val_size = int(len(full_dataset) * config.val_split)
    test_size = len(full_dataset) - train_size - val_size
    
    # Use fixed random seeds for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Training with {len(train_dataset)} samples")
    print(f"Validation with {len(val_dataset)} samples")
    print(f"Test with {len(test_dataset)} samples")
    
    # Configure dataloader settings based on device
    if device.type == 'mps':
        # MPS-specific optimizations for dataloaders
        num_workers = 0  # Use 0 workers for MPS to avoid sharing tensor issues
        persistent_workers = False  # Not needed with 0 workers
        prefetch_factor = 2
        pin_memory = False  # Pin memory can cause issues with MPS
        batch_size = min(8, config.batch_size)  # Use a smaller batch size for MPS
        print(f"Using batch size of {batch_size} for MPS device")
    elif device.type == 'cpu':
        # CPU-specific settings
        num_workers = min(config.num_workers, os.cpu_count() or 2)  # Fewer workers for CPU
        persistent_workers = False  # No need for persistent workers on CPU
        prefetch_factor = 2
        pin_memory = False  # No need for pinned memory on CPU
        batch_size = config.batch_size
    else:
        # For CUDA devices
        num_workers = min(config.num_workers, 4)  # Limit workers to reduce memory usage
        persistent_workers = True
        prefetch_factor = 3
        pin_memory = True  # Use pinned memory for faster CPU->GPU transfer
        batch_size = config.batch_size
    
    # Create a reusable dictionary of dataloader kwargs
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": optimized_collate_fn,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "drop_last": False,
    }
    
    # Create DataLoaders
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs["shuffle"] = False
    
    # For MPS devices, use an even smaller batch size for validation to prevent OOM
    if device.type == 'mps':
        val_dataloader_kwargs["batch_size"] = min(4, batch_size)
        print(f"Using validation batch size: {val_dataloader_kwargs['batch_size']}")
    
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    
    return train_loader, val_loader


def main():
    """Main function to run training."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the Grateful Dead show dating model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu, auto)")
    args = parser.parse_args()
    
    # Load default config
    config = Config()
    
    # Override config with command line arguments
    if args.config:
        config.load_from_file(args.config)
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.lr:
        config.initial_lr = args.lr
    
    if args.steps:
        config.total_training_steps = args.steps
    
    if args.checkpoint:
        config.resume_checkpoint = args.checkpoint
    
    if args.data_dir:
        config.data_dir = args.data_dir
    
    if args.device:
        config.device = args.device
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    
    # Set up device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, device)
    
    # Create trainer
    trainer = Trainer(config, train_loader, val_loader, device)
    
    # Load checkpoint if specified
    if config.resume_checkpoint:
        trainer.load_checkpoint(config.resume_checkpoint)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()