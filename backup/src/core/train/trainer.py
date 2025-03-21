"""
Simplified trainer class for the Grateful Dead show dating model.
Focus on core functionality with reduced complexity.
"""

import os
import time
import datetime
import signal
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from src.core.model.loss import SimplifiedDeadLoss
from src.core.model.model import DeadShowDatingModel


class SimpleTrainer:
    """
    Simplified trainer class for the Grateful Dead show dating model.
    
    Args:
        config: Configuration with training parameters
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
    """
    
    def __init__(self, config, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize model
        self.model = DeadShowDatingModel(sample_rate=24000)
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=0.01
        )
        
        # Initialize scheduler with simple step decay
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize loss function
        self.criterion = SimplifiedDeadLoss()
        
        # Set up logging with TensorBoard
        self.log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_start_time = None
        
        # Create checkpoints directory
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        
        # Interruption flag
        self.interrupt_training = False
        
        # Register signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Signal handler for graceful interruption."""
        print("\nInterrupt received. Will save checkpoint and exit after current batch...")
        self.interrupt_training = True
    
    def save_checkpoint(self, path, is_best=False):
        """Save a checkpoint of the training state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter
        }
        
        # Save to a temporary file first
        temp_path = f"{path}.tmp"
        torch.save(checkpoint, temp_path)
        
        # Atomic rename to ensure checkpoint is not corrupted if interrupted
        if os.path.exists(path):
            os.replace(temp_path, path)
        else:
            os.rename(temp_path, path)
        
        print(f"Checkpoint saved to {path}")
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(os.path.dirname(path), "best_model.pt")
            if os.path.exists(path):
                os.replace(path, best_path)
            else:
                os.rename(temp_path, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path):
        """Load a checkpoint to resume training."""
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found, starting training from scratch.")
            return False
            
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]
        
        print(f"Resumed training from step {self.global_step}")
        return True
    
    def train(self):
        """Main training loop."""
        # Record training start time
        self.training_start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=self.config.steps, initial=self.global_step)
        
        # Train until we reach the total number of steps or early stopping
        early_stopping_triggered = False
        epoch = 0
        
        while True:
            if (self.global_step >= self.config.steps or 
                early_stopping_triggered or 
                self.interrupt_training):
                break
                
            self.model.train()
            train_losses = []
            
            # Training loop for this epoch
            for batch_idx, batch in enumerate(self.train_loader):
                # Skip empty batches
                if batch.get('empty_batch', False):
                    continue
                
                # Move data to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Prepare labels
                label = batch.get('label')
                
                # Forward pass
                try:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch)
                    
                    # Calculate loss
                    loss_dict = self.criterion(
                        outputs,
                        {'days': label},
                        self.global_step,
                        self.config.steps
                    )
                    loss = loss_dict['loss']
                    
                    # Check for NaN/Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("Warning: NaN/Inf detected in loss - skipping batch")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    
                    # Update weights
                    self.optimizer.step()
                except Exception as e:
                    print(f"Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Store loss for reporting
                train_losses.append(loss.item())
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar(
                        "train/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step
                    )
                
                # Update progress bar
                avg_loss = np.mean(train_losses[-50:]) if train_losses else 0
                pbar.set_description(
                    f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                pbar.update(1)
                
                # Increment global step
                self.global_step += 1
                
                # Check if we need to do validation
                if (self.global_step % self.config.validation_interval == 0 or 
                    self.global_step >= self.config.steps):
                    val_loss = self.validate()
                    
                    # Update LR scheduler
                    self.scheduler.step(val_loss)
                    
                    # Check for improvement
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(
                            os.path.join(self.config.output_dir, "checkpoints", "best_checkpoint.pt"),
                            is_best=True
                        )
                    else:
                        self.patience_counter += 1
                        
                        # Check for early stopping
                        if (self.config.early_stopping and 
                            self.patience_counter >= self.config.patience):
                            print(f"Early stopping triggered after {self.patience_counter} validations without improvement")
                            early_stopping_triggered = True
                            break
                
                # Check if we need to save a checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(
                        os.path.join(
                            self.config.output_dir, 
                            "checkpoints", 
                            f"checkpoint_step_{self.global_step}.pt"
                        )
                    )
                
                # Check if we've reached the total training steps
                if self.global_step >= self.config.steps:
                    break
                
                # Check for interrupt signal
                if self.interrupt_training:
                    print("Training interrupted by signal")
                    break
            
            # End of epoch
            epoch += 1
        
        # Training complete
        pbar.close()
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.config.output_dir, "checkpoints", "final_checkpoint.pt")
        )
        
        # Print final training information
        training_time = time.time() - self.training_start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time / 3600:.2f} hours)")
        print(f"Reached step {self.global_step} of {self.config.steps}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Clean up
        self.writer.close()
    
    def validate(self):
        """Run validation loop and return validation loss."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Skip empty batches
                if batch.get('empty_batch', False):
                    continue
                
                # Move data to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Prepare labels
                label = batch.get('label')
                
                # Forward pass
                try:
                    outputs = self.model(batch)
                    
                    # Calculate loss
                    loss_dict = self.criterion(
                        outputs,
                        {'days': label},
                        self.global_step,
                        self.config.steps
                    )
                    val_loss = loss_dict['loss']
                    
                    val_losses.append(val_loss.item())
                except Exception as e:
                    print(f"Error during validation step: {e}")
                    continue
        
        # Calculate average validation loss
        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        
        # Log validation loss
        self.writer.add_scalar("val/loss", avg_val_loss, self.global_step)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Return to training mode
        self.model.train()
        
        return avg_val_loss