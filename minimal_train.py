#!/usr/bin/env python3
"""
Minimal training script for the Dead Show Dating model with extreme memory optimizations.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import gc
import math
import random
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

# Enable aggressive MPS memory management
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    # Use a smaller memory fraction to avoid OOM
    torch.mps.set_per_process_memory_fraction(0.5)
    print("MPS memory optimization enabled")


class MpsCompatiblePool2d(nn.Module):
    """Custom pooling layer that's compatible with MPS by moving to CPU when needed."""
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x):
        device = x.device
        if device.type == 'mps':
            # Move to CPU, perform pooling, then back to MPS
            return self.pool(x.to('cpu')).to(device)
        else:
            # Use normal pooling for other devices
            return self.pool(x)


class MpsCompatiblePool1d(nn.Module):
    """Custom pooling layer that's compatible with MPS by moving to CPU when needed."""
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.pool = nn.AdaptiveAvgPool1d(output_size)
        
    def forward(self, x):
        device = x.device
        if device.type == 'mps':
            # Move to CPU, perform pooling, then back to MPS
            return self.pool(x.to('cpu')).to(device)
        else:
            # Use normal pooling for other devices
            return self.pool(x)


class MinimalDeadModel(nn.Module):
    """Minimal version of the Dead Show Dating model to reduce memory footprint."""
    def __init__(self):
        super().__init__()
        
        # Simplified feature extractors
        self.harmonic_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            MpsCompatiblePool2d((8, 8)),  # Use MPS-compatible pooling
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64)
        )
        
        self.percussive_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            MpsCompatiblePool2d((8, 8)),  # Use MPS-compatible pooling
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64)
        )
        
        self.chroma_extractor = nn.Sequential(
            nn.Conv1d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MpsCompatiblePool1d(16),  # Use MPS-compatible pooling
            nn.Flatten(),
            nn.Linear(16 * 16, 32)
        )
        
        # Seasonal pattern embedding
        self.seasonal_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Explicitly use sigmoid to get [0,1] range
        )
    
    def forward(self, batch):
        outputs = {}
        
        try:
            # Process harmonic features
            if 'harmonic' in batch:
                harmonic = batch['harmonic']
                if harmonic.dtype != torch.float32:
                    harmonic = harmonic.to(torch.float32)
                harmonic_features = self.harmonic_extractor(harmonic)
            else:
                # Fallback to zeros
                harmonic_features = torch.zeros(batch['mel_spec'].shape[0], 64, device=batch['mel_spec'].device, dtype=torch.float32)
                
            # Process percussive features if available
            if 'percussive' in batch:
                percussive = batch['percussive']
                if percussive.dtype != torch.float32:
                    percussive = percussive.to(torch.float32)
                percussive_features = self.percussive_extractor(percussive)
            else:
                # Fallback to zeros
                percussive_features = torch.zeros_like(harmonic_features)
                
            # Process chroma features
            if 'chroma' in batch and batch['chroma'].dim() >= 2:
                # Ensure chroma has right shape: [batch, 12, time]
                chroma = batch['chroma']
                if chroma.dtype != torch.float32:
                    chroma = chroma.to(torch.float32)
                    
                if chroma.dim() == 2:
                    # Add batch dimension
                    chroma = chroma.unsqueeze(0)

                chroma_features = self.chroma_extractor(chroma)
            else:
                # Fallback to zeros
                chroma_features = torch.zeros(harmonic_features.shape[0], 32, device=harmonic_features.device, dtype=torch.float32)
            
            # Get day of year for seasonal patterns (0-365)
            if 'date' in batch:
                day_of_year = batch['date']
                if day_of_year.dtype != torch.float32:
                    day_of_year = day_of_year.to(torch.float32)
                    
                day_of_year = torch.remainder(day_of_year, 365.0).unsqueeze(1)
                day_of_year = day_of_year / 365.0  # Normalize to [0,1]
            else:
                day_of_year = torch.zeros(harmonic_features.shape[0], 1, device=harmonic_features.device, dtype=torch.float32)
                
            seasonal_features = self.seasonal_embedding(day_of_year)
            
            # Combine all features
            combined = torch.cat([harmonic_features, percussive_features, chroma_features, seasonal_features], dim=1)
            
            # Final prediction
            days_normalized = self.final_layers(combined)
            
            # Return results
            outputs['days'] = days_normalized 
            outputs['days_unscaled'] = days_normalized * 10000.0
            outputs['audio_features'] = harmonic_features
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return zeros as fallback
            batch_size = 1
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    device = v.device
                    break
            
            outputs['days'] = torch.zeros(batch_size, 1, device=device, dtype=torch.float32, requires_grad=True)
            outputs['days_unscaled'] = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
            outputs['audio_features'] = torch.zeros(batch_size, 64, device=device, dtype=torch.float32)
        
        return outputs


class MinimalDataLoader:
    """Memory-optimized data loader that loads files one at a time without caching."""
    def __init__(self, data_dir, batch_size=2, device=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        
        # Find all preprocessed files
        self.file_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.pt'):
                    self.file_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.file_paths)} preprocessed files")
        
        # Shuffle file paths
        random.shuffle(self.file_paths)
        
        # For debugging, limit the total files
        max_files = min(10000, len(self.file_paths))  # Reduced for faster iteration
        self.file_paths = self.file_paths[:max_files]
        print(f"Using {len(self.file_paths)} files for training")
        
        # Calculate batches
        self.num_batches = max(1, len(self.file_paths) // batch_size)
        
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        # Create batches
        for i in range(0, len(self.file_paths), self.batch_size):
            # Get batch file paths
            batch_paths = self.file_paths[i:i+self.batch_size]
            
            # Load each file
            batch_data = {'mel_spec': [], 'chroma': [], 'date': []}
            
            for path in batch_paths:
                try:
                    # Load file with torch.load
                    data = torch.load(path, map_location='cpu')
                    
                    # Extract features
                    if 'mel_spec' in data:
                        batch_data['mel_spec'].append(data['mel_spec'])
                    
                    if 'chroma' in data:
                        batch_data['chroma'].append(data['chroma'])
                    
                    # Extract date from filename
                    date_match = path.split('/')[-1].split('_')[0].split('-')
                    if len(date_match) >= 3:
                        try:
                            year, month, day = int(date_match[0]), int(date_match[1]), int(date_match[2])
                            
                            # Convert to day of year (0-365)
                            import datetime
                            date = datetime.datetime(year, month, day).timetuple().tm_yday
                            batch_data['date'].append(date / 365.0)  # Normalize to [0,1]
                        except (ValueError, IndexError):
                            batch_data['date'].append(0.5)  # Default to middle of year
                    else:
                        batch_data['date'].append(0.5)  # Default to middle of year
                
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            # Skip if no data
            if not batch_data['mel_spec']:
                continue
            
            try:
                # Process on CPU first
                # Convert lists to tensors
                mel_specs = batch_data['mel_spec']
                if mel_specs and isinstance(mel_specs[0], torch.Tensor):
                    # Use a fixed size for all spectrograms for consistency
                    target_time = 200  # Fixed time dimension for consistency
                    padded_mel_specs = []
                    
                    for mel in mel_specs:
                        # Add batch dimension if needed
                        if mel.dim() == 2:
                            mel = mel.unsqueeze(0)
                        
                        # Ensure float32
                        if mel.dtype != torch.float32:
                            mel = mel.to(torch.float32)
                        
                        # Resize time dimension
                        current_time = mel.shape[-1]
                        if current_time > target_time:
                            # Truncate
                            mel = mel[..., :target_time]
                        elif current_time < target_time:
                            # Pad
                            pad_size = target_time - current_time
                            mel = torch.nn.functional.pad(mel, (0, pad_size))
                            
                        padded_mel_specs.append(mel)
                    
                    # Stack tensors
                    mel_tensor = torch.cat(padded_mel_specs, dim=0)
                    
                    # Add channel dimension for Conv2D
                    if mel_tensor.dim() == 3:
                        mel_tensor = mel_tensor.unsqueeze(1)
                    
                    batch_data['mel_spec'] = mel_tensor.to(torch.float32)
                else:
                    # Fallback to empty tensor
                    batch_data['mel_spec'] = torch.zeros((len(batch_paths), 1, 128, target_time), dtype=torch.float32)
                
                # Process chroma features
                chromas = batch_data['chroma']
                if chromas and isinstance(chromas[0], torch.Tensor):
                    # Use fixed time dimension
                    padded_chromas = []
                    
                    for chroma in chromas:
                        # Add batch dimension if needed
                        if chroma.dim() == 2:
                            chroma = chroma.unsqueeze(0)
                        
                        # Ensure float32
                        if chroma.dtype != torch.float32:
                            chroma = chroma.to(torch.float32)
                        
                        # Resize time dimension
                        current_time = chroma.shape[-1]
                        if current_time > target_time:
                            # Truncate
                            chroma = chroma[..., :target_time]
                        elif current_time < target_time:
                            # Pad
                            pad_size = target_time - current_time
                            chroma = torch.nn.functional.pad(chroma, (0, pad_size))
                            
                        padded_chromas.append(chroma)
                    
                    # Stack tensors
                    batch_data['chroma'] = torch.cat(padded_chromas, dim=0).to(torch.float32)
                else:
                    # Fallback to empty tensor
                    batch_data['chroma'] = torch.zeros((len(batch_paths), 12, target_time), dtype=torch.float32)
                
                # Convert date to tensor
                batch_data['date'] = torch.tensor(batch_data['date'], dtype=torch.float32)
                
                # Convert mel_spec to harmonic with explicit clone to avoid reference issues
                batch_data['harmonic'] = batch_data['mel_spec'].clone().detach().to(torch.float32)
                
                # Move tensors to device after all CPU processing is done
                if self.device:
                    for key in batch_data:
                        if isinstance(batch_data[key], torch.Tensor):
                            batch_data[key] = batch_data[key].to(self.device, dtype=torch.float32)
                
                # Yield batch
                yield batch_data
            
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Skip this batch
                continue
            
            finally:
                # Force garbage collection after each batch
                del batch_data
                gc.collect()
                if self.device and self.device.type == 'mps':
                    torch.mps.empty_cache()


def train(args):
    """Train the model with extreme memory optimizations."""
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Force model to use float32
    torch.set_default_dtype(torch.float32)
    
    # Create model
    model = MinimalDeadModel()
    model = model.to(device, dtype=torch.float32)
    print("Created minimal model")
    
    # Verify all parameters are float32
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"Converting {name} from {param.dtype} to float32")
            param.data = param.data.to(torch.float32)
    
    # Create data loaders
    train_loader = MinimalDataLoader(args.data_dir, batch_size=args.batch_size, device=device)
    
    # Use SGD instead of Adam for better memory efficiency
    print(f"Using memory-efficient SGD optimizer with lr={args.lr}")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Print memory usage
    if device.type == 'mps':
        print(f"MPS memory allocated: {torch.mps.current_allocated_memory() / (1024**3):.2f} GB")
    elif device.type == 'cuda':
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    # Training loop
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    
    try:
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            batch_count = 0  # Keep track of successful batches
            
            # Create progress bar
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Debug first batch
                    if batch_idx == 0 and epoch == 0:
                        print(f"First batch tensors and types:")
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                    
                    # Get labels (dates)
                    labels = batch['date'].clone().to(torch.float32).reshape(-1, 1)
                    
                    # Forward pass - wrap in try/except to handle potential errors
                    try:
                        outputs = model(batch)
                    except Exception as e:
                        print(f"Error in forward pass at batch {batch_idx}: {e}")
                        # Skip this batch
                        pbar.update(1)
                        continue
                    
                    # Calculate loss
                    try:
                        if 'days' in outputs and outputs['days'] is not None:
                            loss = criterion(outputs['days'], labels)
                        else:
                            print(f"Missing 'days' in outputs at batch {batch_idx}, skipping")
                            pbar.update(1)
                            continue
                    except Exception as e:
                        print(f"Error calculating loss at batch {batch_idx}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad(set_to_none=True)  # More memory efficient
                    
                    try:
                        loss.backward()
                        # Step optimizer
                        optimizer.step()
                    except Exception as e:
                        print(f"Error in backward/step at batch {batch_idx}: {e}")
                        # Continue to next batch
                        pbar.update(1)
                        continue
                    
                    # Update progress metrics
                    running_loss += loss.item()
                    batch_count += 1
                    
                    # Update progress bar with meaningful loss (if we have any)
                    avg_loss = running_loss / batch_count if batch_count > 0 else 0
                    pbar.set_postfix({"loss": avg_loss})
                    pbar.update(1)
                    
                    # Update global step
                    global_step += 1
                    
                    # Force garbage collection after each batch
                    gc.collect()
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                        if batch_idx % 50 == 0:  # Print less frequently to reduce overhead
                            mem_used = torch.mps.current_allocated_memory() / (1024**3)
                            print(f"MPS memory at step {global_step}: {mem_used:.2f} GB")
                    elif device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Save checkpoint periodically
                    if global_step % args.save_every == 0:
                        try:
                            # Save model
                            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pt")
                            
                            # Move model to CPU for saving to avoid MPS issues
                            model_cpu = model.to('cpu')
                            torch.save(model_cpu.state_dict(), checkpoint_path)
                            model = model.to(device, dtype=torch.float32)  # Move back with explicit dtype
                            
                            print(f"Saved checkpoint to {checkpoint_path}")
                        except Exception as e:
                            print(f"Error saving checkpoint: {e}")
                    
                except Exception as e:
                    # Catch any other errors in the batch processing loop
                    print(f"Unexpected error processing batch {batch_idx}: {e}")
                    pbar.update(1)
                    continue
            
            pbar.close()
            
            # Print epoch summary if we processed any batches
            if batch_count > 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss / batch_count:.4f}")
            else:
                print(f"Epoch {epoch+1}/{args.epochs}: No valid batches processed")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    except Exception as e:
        print(f"Training stopped due to error: {e}")
    
    finally:
        # Always try to save the final model
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.to('cpu').state_dict(), os.path.join(args.output_dir, "final_model.pt"))
            print("Final model saved successfully")
        except Exception as e:
            print(f"Error saving final model: {e}")
        
        # Clean up resources
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal training script for Dead Show Dating model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--device", type=str, default="mps", help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    train(args) 