#!/usr/bin/env python3
"""
Simplified single-process training script that avoids multiprocessing issues.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.core.data.dataset import PreprocessedDeadShowDataset
from src.core.train.trainer import SimpleTrainer
from src.core.utils.utils import get_device

# Define a direct collate function that doesn't use nested functions (for better pickling)
def direct_collate_fn(batch):
    """
    Simple collate function that handles audio data batching - no nested functions.
    """
    # Filter out error items
    valid_batch = [item for item in batch if not item.get('error', False)]
    
    if not valid_batch:
        return {'empty_batch': True}
    
    # Initialize output dictionary
    output = {}
    
    # Process each key in the batch
    for key in valid_batch[0].keys():
        if key == 'file' or key == 'path':
            # Keep strings as a list
            output[key] = [item[key] for item in valid_batch]
        else:
            # Handle tensor data
            tensors = []
            for item in valid_batch:
                if key in item and isinstance(item[key], torch.Tensor):
                    tensors.append(item[key].float())
            
            if tensors:
                # Stack tensors if possible
                try:
                    output[key] = torch.stack(tensors)
                except:
                    # Skip keys that can't be stacked
                    pass
    
    return output


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


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train the Grateful Dead show dating model (single process)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
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
    
    # Create datasets
    print("Creating datasets...")
    try:
        full_dataset = PreprocessedDeadShowDataset(
            preprocessed_dir=args.data_dir,
            device=None
        )
        
        # Split dataset 90/10 for training/validation
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Create dataset indices
        indices = list(range(len(full_dataset)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create dataloaders - single process to avoid pickling issues
    print("Creating dataloaders (single process)...")
    try:
        # Test the collate function (quietly)
        sample_batch = []
        for i in range(min(2, len(train_dataset))):
            try:
                sample_item = train_dataset[i]
                sample_batch.append(sample_item)
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue
                
        if sample_batch:
            try:
                collated = direct_collate_fn(sample_batch)
                print("Collate function test successful")
            except Exception as e:
                print(f"Collate function test failed: {e}")
                # Fall back to very simple collate
                print("Using fallback collate function")
        
        # Create actual dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Single process
            collate_fn=direct_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,  # Single process
            collate_fn=direct_collate_fn
        )
        
        # Test first batch to make sure dataloader works
        print("Testing dataloader with a batch...")
        try:
            first_batch = next(iter(train_loader))
            print(f"Dataloader test successful")
        except Exception as e:
            print(f"Dataloader test failed: {e}")
            return
            
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return
    
    # Create trainer
    print("Creating trainer...")
    trainer = SimpleTrainer(config, train_loader, val_loader, device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()