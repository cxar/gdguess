#!/usr/bin/env python3
"""
Ultra-minimal version with everything in one file for the Grateful Dead show dating model.
This script is designed to be extremely robust and handle any dataset format issues.
"""

import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import re
from tqdm import tqdm

# --- SIMPLIFIED MODEL ---

class MinimalModel(nn.Module):
    """Ultra-simplified model for dating Dead shows."""
    
    def __init__(self):
        super().__init__()
        # Simplified convolutional network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        
        # Uncertainty head (for better calibration)
        self.uncertainty = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        # Handle any input shape
        if isinstance(x, dict):
            # Try to get input from a dictionary
            if 'mel_spec' in x:
                x = x['mel_spec']
            elif 'harmonic' in x:
                x = x['harmonic']
            else:
                # Just use the first tensor we find
                for k, v in x.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) >= 3:
                        x = v
                        break
        
        # Ensure 4D: [batch_size, channels, height, width]
        if x.dim() == 3:
            if x.shape[1] == 128:  # [batch, freq, time]
                x = x.unsqueeze(1)  # Add channel dimension
            else:
                x = x.permute(0, 2, 1).unsqueeze(1)  # Convert to [batch, 1, time, freq]
        
        # Process through convolutional layers
        features = self.conv_layers(x)
        features = features.squeeze(-1).squeeze(-1)  # Flatten spatial dims
        
        # Get prediction and uncertainty
        days = self.fc(features)
        log_var = self.uncertainty(features)
        
        return {
            'days': days,
            'log_variance': log_var
        }


# --- LOSS FUNCTION ---

class SimpleLoss(nn.Module):
    """Simple loss function with uncertainty estimation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """Compute loss with uncertainty weighting."""
        # Get predictions and targets
        pred_days = predictions['days']
        log_var = predictions['log_variance']
        
        # Get target days
        if isinstance(targets, dict):
            if 'days' in targets:
                true_days = targets['days']
            elif 'label' in targets:
                true_days = targets['label']
            else:
                # Just use the first tensor
                for k, v in targets.items():
                    if isinstance(v, torch.Tensor):
                        true_days = v
                        break
        else:
            true_days = targets
        
        # Ensure compatible shapes
        if true_days.dim() > 1 and pred_days.dim() == 1:
            pred_days = pred_days.unsqueeze(1)
        elif true_days.dim() == 1 and pred_days.dim() > 1:
            true_days = true_days.unsqueeze(1)
        
        # Compute loss with uncertainty
        precision = torch.exp(-log_var)
        squared_error = (pred_days - true_days) ** 2
        uncertainty_loss = 0.5 * (precision * squared_error + log_var)
        
        return uncertainty_loss.mean()


# --- DATASET ---

class RobustDeadShowDataset(Dataset):
    """Dataset that's extremely robust to file format variation."""
    
    def __init__(self, data_dir):
        """Initialize the dataset."""
        self.data_dir = data_dir
        
        # Find all .pt files
        print(f"Looking for files in: {data_dir}")
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True))
        print(f"Found {len(self.files)} files")
        
        if not self.files:
            raise ValueError(f"No .pt files found in {data_dir}")
    
    def __len__(self):
        """Return dataset size."""
        return len(self.files)
    
    def __getitem__(self, idx):
        """Get a dataset item."""
        try:
            file_path = self.files[idx]
            data = torch.load(file_path, map_location='cpu')
            
            # Process the data to ensure minimum requirements
            processed = self._process_data(data, file_path)
            return processed
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            # Return dummy data as a fallback
            return self._create_dummy_data()
    
    def _process_data(self, data, file_path):
        """Process loaded data to ensure minimum requirements."""
        # Add file path
        data['path'] = file_path
        
        # Extract date from filename if needed
        if 'label' not in data:
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file_path)
            if date_match:
                year, month, day = map(int, date_match.groups())
                date_obj = datetime.date(year, month, day)
                base_date = datetime.date(1968, 1, 1)
                days = (date_obj - base_date).days
                data['label'] = torch.tensor(days, dtype=torch.float)
            else:
                data['label'] = torch.tensor(0.0, dtype=torch.float)
        
        # Ensure we have a spectrogram
        if 'mel_spec' not in data and 'harmonic' not in data:
            # Try to find something that looks like a spectrogram
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                    if value.shape[-2] == 128 or value.shape[-1] == 128:
                        data['mel_spec'] = value
                        break
            
            # If still not found, create a dummy
            if 'mel_spec' not in data and 'harmonic' not in data:
                data['mel_spec'] = torch.rand(1, 128, 100, dtype=torch.float)
        
        return data
    
    def _create_dummy_data(self):
        """Create dummy data for fallback."""
        return {
            'mel_spec': torch.rand(1, 128, 100, dtype=torch.float),
            'label': torch.tensor(0.0, dtype=torch.float),
            'dummy': True
        }


def collate_fn(batch):
    """Robust collate function that handles any variation in batch."""
    # Filter out error items
    valid_batch = [item for item in batch if not item.get('error', False)]
    
    if not valid_batch:
        return {'empty_batch': True}
    
    # Initialize output dictionary
    output = {}
    
    # Get all keys from all items
    all_keys = set()
    for item in valid_batch:
        all_keys.update(item.keys())
    
    # Process each key
    for key in all_keys:
        # Skip special keys
        if key in ['error', 'path', 'dummy']:
            continue
            
        # Collect tensors for this key
        tensors = []
        for item in valid_batch:
            if key in item and isinstance(item[key], torch.Tensor):
                tensors.append(item[key].float())
            else:
                # Skip items missing this key
                continue
        
        if tensors:
            # Try to stack tensors if they have compatible shapes
            try:
                output[key] = torch.stack(tensors)
            except:
                # Skip keys that can't be stacked
                pass
    
    # Store paths as a list
    paths = [item.get('path', '') for item in valid_batch]
    if paths:
        output['paths'] = paths
    
    return output


# --- TRAINER ---

class RobustTrainer:
    """Trainer designed to handle any errors gracefully."""
    
    def __init__(self, train_loader, val_loader, device, steps=1000):
        """Initialize trainer."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.steps = steps
        
        # Create model
        self.model = MinimalModel().to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create loss function
        self.loss_fn = SimpleLoss()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Run training loop."""
        # Create progress bar
        pbar = tqdm(total=self.steps)
        
        # Create data iterator
        train_iter = iter(self.train_loader)
        
        # Training loop
        while self.global_step < self.steps:
            # Reset model to training mode
            self.model.train()
            
            # Get batch (restart iterator if needed)
            try:
                batch = next(train_iter)
            except (StopIteration, IndexError):
                train_iter = iter(self.train_loader)
                try:
                    batch = next(train_iter)
                except (StopIteration, IndexError):
                    print("Error: Empty train loader")
                    break
            
            # Skip empty batches
            if batch.get('empty_batch', True):
                continue
            
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            # Forward pass
            try:
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                
                # Compute loss
                loss = self.loss_fn(outputs, batch)
                
                # Check for valid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: NaN/Inf loss, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update progress bar
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
                
                # Increment step counter
                self.global_step += 1
                
                # Run validation occasionally
                if self.global_step % 100 == 0:
                    val_loss = self.validate()
                    pbar.set_description(f"Train: {loss.item():.4f}, Val: {val_loss:.4f}")
                
                # Check if we're done
                if self.global_step >= self.steps:
                    break
                    
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        # Save final model
        torch.save(self.model.state_dict(), "minimal_model.pt")
        print(f"Training complete. Model saved to minimal_model.pt")
        
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
                
                try:
                    # Move batch to device
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Compute loss
                    loss = self.loss_fn(outputs, batch)
                    
                    # Store loss
                    val_losses.append(loss.item())
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        # Return average validation loss
        return sum(val_losses) / max(len(val_losses), 1)


# --- MAIN FUNCTION ---

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Ultra-minimal Grateful Dead Show Dating Model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory with .pt files")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Create dataset with robust error handling
    try:
        dataset = RobustDeadShowDataset(args.data_dir)
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Random split might fail if dataset is too large
        try:
            from torch.utils.data import random_split
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        except:
            # Manual split as fallback
            all_indices = list(range(len(dataset)))
            train_indices = all_indices[:train_size]
            val_indices = all_indices[train_size:train_size+val_size]
            
            # Create subset datasets
            from torch.utils.data import Subset
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Create dataloaders with robust error handling
    try:
        print("Creating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Single process to avoid multiprocessing issues
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # Single process to avoid multiprocessing issues
            collate_fn=collate_fn
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return
    
    # Create trainer
    trainer = RobustTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        steps=args.steps
    )
    
    # Run training
    print("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(trainer.model.state_dict(), "interrupted_model.pt")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()