#!/usr/bin/env python3
"""
Super simplified stand-alone training script for the Grateful Dead show dating model.
This script has everything in one file for the simplest possible training experience.
"""

import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# --- MODEL ARCHITECTURE ---

class ResidualBlock(nn.Module):
    """Simple residual block with 2D convolutions."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv for residual connection if dimensions change
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.gelu(x)


class DeadShowModel(nn.Module):
    """Simplified model for dead show dating."""
    
    def __init__(self):
        super().__init__()
        
        # Harmonic branch
        self.harmonic_branch = nn.Sequential(
            ResidualBlock(1, 32),
            ResidualBlock(32, 64)
        )
        
        # Percussive branch
        self.percussive_branch = nn.Sequential(
            ResidualBlock(1, 32),
            ResidualBlock(32, 64)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.GELU()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, batch):
        """Forward pass."""
        # Get features from batch
        harmonic = batch['harmonic']
        percussive = batch['percussive']
        
        # Process features
        h_out = self.harmonic_branch(harmonic)
        p_out = self.percussive_branch(percussive)
        
        # Concatenate and fuse
        concat = torch.cat([h_out, p_out], dim=1)
        fused = self.fusion(concat)
        
        # Global pooling
        pooled = self.global_pool(fused).view(fused.size(0), -1)
        
        # Final prediction
        days = self.final_layers(pooled)
        log_var = self.uncertainty_head(pooled)
        
        return {
            'days': days,
            'log_variance': log_var
        }


# --- LOSS FUNCTION ---

class UncertaintyLoss(nn.Module):
    """Loss function with uncertainty estimation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, log_var):
        """Compute loss with uncertainty weighting."""
        # Compute precision (inverse variance)
        precision = torch.exp(-log_var)
        
        # Compute squared error weighted by precision
        sq_error = (pred - target) ** 2
        uncertainty_loss = 0.5 * (precision * sq_error + log_var)
        
        return uncertainty_loss.mean()


# --- DATASET ---

class SimpleDeadShowDataset(Dataset):
    """Simplified dataset for Grateful Dead shows."""
    
    def __init__(self, data_dir):
        """Initialize dataset."""
        self.data_dir = data_dir
        
        # In a real dataset, we would load files here
        # For this simplified example, we'll just create random data
        self.size = 100
    
    def __len__(self):
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx):
        """Get a dataset item."""
        # Create random data for harmonic and percussive spectrograms
        harmonic = torch.randn(1, 128, 50)
        percussive = torch.randn(1, 128, 50)
        
        # Create a random label (days since a reference date)
        label = torch.rand(1) * 10000
        
        return {
            'harmonic': harmonic,
            'percussive': percussive,
            'label': label
        }


def collate_fn(batch):
    """Collate batch of samples."""
    if not batch:
        return {'empty_batch': True}
    
    # Create output dictionary
    output = {}
    
    # Get keys from first item
    keys = batch[0].keys()
    
    # Stack tensors
    for key in keys:
        tensors = [item[key] for item in batch]
        output[key] = torch.stack(tensors)
    
    return output


# --- TRAINER ---

class SimpleTrainer:
    """Simplified trainer for Dead Show dating model."""
    
    def __init__(self, train_loader, val_loader, device, learning_rate=0.001, steps=1000):
        """Initialize trainer."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.steps = steps
        
        # Create model
        self.model = DeadShowModel().to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create loss function
        self.loss_fn = UncertaintyLoss()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Run training loop."""
        # Create progress bar
        pbar = tqdm(total=self.steps, initial=self.global_step)
        
        # Create data iterator
        train_iter = iter(self.train_loader)
        
        # Training loop
        while self.global_step < self.steps:
            # Reset model to training mode
            self.model.train()
            
            # Get batch (restart iterator if needed)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Skip empty batches
            if batch.get('empty_batch', True):
                continue
            
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute loss
            pred_days = outputs['days']
            true_days = batch['label']
            log_var = outputs['log_variance']
            loss = self.loss_fn(pred_days, true_days, log_var)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
            
            # Update step counter
            self.global_step += 1
            
            # Run validation every 100 steps
            if self.global_step % 100 == 0:
                val_loss = self.validate()
                
                # Update progress bar with validation loss
                pbar.set_description(f"Train: {loss.item():.4f}, Val: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "best_model.pt")
        
        # Close progress bar
        pbar.close()
    
    def validate(self):
        """Run validation loop."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Skip empty batches
                if batch.get('empty_batch', True):
                    continue
                
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                pred_days = outputs['days']
                true_days = batch['label']
                log_var = outputs['log_variance']
                loss = self.loss_fn(pred_days, true_days, log_var)
                
                # Store loss
                val_losses.append(loss.item())
        
        # Return average validation loss
        return sum(val_losses) / len(val_losses) if val_losses else float('inf')


# --- MAIN FUNCTION ---

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simple Dead Show Dating Model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = SimpleDeadShowDataset(args.data_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create trainer
    trainer = SimpleTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        steps=args.steps
    )
    
    # Run training
    print("Starting training...")
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()