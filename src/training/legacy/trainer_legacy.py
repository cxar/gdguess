"""
Trainer class for the Grateful Dead show dating model.
"""

import os
import time
import datetime
import signal
import math
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np
from tqdm import tqdm
import gc

from models import CombinedDeadLoss
from models.dead_model import DeadShowDatingModel
from config import Config
from utils.helpers import cleanup_file_handles


class Trainer:
    """
    Trainer class for training and evaluating the Dead Show Dating model.
    
    Args:
        config: Configuration object with training parameters
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
    """
    
    def __init__(self, config: Config, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Track if we're using an Apple Silicon device
        self.is_mps = device.type == 'mps'
        
        # Initialize model
        self.model = DeadShowDatingModel(sample_rate=config.target_sr)
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        # Initialize SWA model
        self.swa_model = AveragedModel(self.model)
        self.swa_model = self.swa_model.to(device)
        
        # Set up learning rate scheduler with warmup
        self.warmup_steps = int(config.total_training_steps * 0.1)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda step: self._lr_lambda(step)
        )
        
        # Set up SWA scheduler
        self.swa_start = int(config.total_training_steps * 0.75)
        self.swa_scheduler = SWALR(
            self.optimizer, 
            anneal_strategy="cos", 
            anneal_epochs=int(config.total_training_steps * 0.05),
            swa_lr=config.initial_lr * 0.1
        )
        
        # Initialize loss function
        self.criterion = CombinedDeadLoss()
        
        # Set up logging
        self.log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_start_time = None
        
        # Interruption flag
        self.interrupt_training = False
        
        # Register signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
    def _lr_lambda(self, step):
        """Custom learning rate schedule with warmup and cosine decay."""
        if step < self.warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(step - self.warmup_steps) / float(
                max(1, self.config.total_training_steps - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    def _handle_interrupt(self, signum, frame):
        """Signal handler for graceful interruption."""
        print("\nInterrupt received. Will save checkpoint and exit after current epoch...")
        self.interrupt_training = True
    
    def save_checkpoint(self, path, is_best=False):
        """Save a checkpoint of the training state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "swa_scheduler_state_dict": self.swa_scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "swa_model_state_dict": self.swa_model.state_dict(),
            "rng_states": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state()
            }
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
        
        if "swa_model_state_dict" in checkpoint:
            self.swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
        
        if "swa_scheduler_state_dict" in checkpoint:
            self.swa_scheduler.load_state_dict(checkpoint["swa_scheduler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]
        
        print(f"Resumed training from step {self.global_step}")
        return True
    
    def train(self):
        """Main training loop."""
        # Record training start time
        self.training_start_time = time.time()
        
        # Log system information
        self._log_system_info()
        
        # Create progress bar
        pbar = tqdm(total=self.config.total_training_steps, initial=self.global_step)
        
        # Train until we reach the total number of steps or early stopping
        early_stopping_triggered = False
        epoch = 0
        
        while True:
            if (self.global_step >= self.config.total_training_steps or 
                early_stopping_triggered or 
                self.interrupt_training):
                break
                
            self.model.train()
            train_losses = []
            
            # Training loop for this epoch
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    print(f"Processing batch {batch_idx}, keys: {list(batch.keys())}")
                    
                    # Skip empty batches
                    if batch.get('empty_batch', False):
                        print("Skipping empty batch")
                        continue
                    
                    # Move data to device
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                    
                    # Prepare labels
                    label = batch.get('label')
                    era = batch.get('era')
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch)
                    
                    # Calculate loss
                    loss_dict = self.criterion(
                        outputs,
                        {'days': label, 'era': era},
                        self.global_step,
                        self.config.total_training_steps
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
                        self.model.parameters(), self.config.grad_clip_value
                    )
                    
                    # Update weights
                    self.optimizer.step()
                    
                    # Update SWA if we've reached the SWA start point
                    if self.global_step >= self.swa_start:
                        self.swa_model.update_parameters(self.model)
                        self.swa_scheduler.step()
                    else:
                        self.scheduler.step()
                    
                    # Clear memory for MPS devices
                    if self.is_mps:
                        if 'outputs' in locals(): del outputs
                        if 'loss' in locals(): del loss
                        if 'loss_dict' in locals(): del loss_dict
                        if 'batch' in locals() and 'label' in batch: del batch['label']
                        if 'batch' in locals() and 'era' in batch: del batch['era']
                        if 'batch' in locals(): del batch
                        gc.collect()
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    
                    # Store loss for reporting if available
                    if 'loss' in locals():
                        train_losses.append(loss.item())
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0 and 'loss' in locals():
                        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                        self.writer.add_scalar(
                            "train/learning_rate",
                            self.optimizer.param_groups[0]["lr"],
                            self.global_step
                        )
                        
                        # Also log individual loss components if available
                        if 'loss_dict' in locals():
                            for k, v in loss_dict.items():
                                if k != 'loss':
                                    self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)
                    
                    # Update progress bar
                    avg_loss = np.mean(train_losses[-100:]) if train_losses else 0
                    pbar.set_description(
                        f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                    pbar.update(1)
                    
                    # Increment global step
                    self.global_step += 1
                    
                    # Check if we need to do validation
                    if (self.global_step % self.config.validation_interval == 0 or 
                        self.global_step >= self.config.total_training_steps):
                        val_loss = self.validate()
                        
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
                            if (self.config.use_early_stopping and 
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
                    if self.global_step >= self.config.total_training_steps:
                        break
                    
                    # Check for interrupt signal
                    if self.interrupt_training:
                        print("Training interrupted by signal")
                        break
                
                except Exception as e:
                    print(f"Error during training step: {e}")
                    import traceback
                    traceback.print_exc()
                    # Increment global step and update progress bar
                    self.global_step += 1
                    pbar.update(1)
                    continue
            
            # End of epoch
            epoch += 1
        
        # Training complete
        pbar.close()
        
        # Final validation with SWA model if we did enough steps for SWA
        if self.global_step >= self.swa_start:
            self._evaluate_swa_model()
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.config.output_dir, "checkpoints", "final_checkpoint.pt")
        )
        
        # Print final training information
        training_time = time.time() - self.training_start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time / 3600:.2f} hours)")
        print(f"Reached step {self.global_step} of {self.config.total_training_steps}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Clean up
        self.writer.close()
        cleanup_file_handles()
    
    def validate(self):
        """Run validation loop and return validation loss."""
        self.model.eval()
        val_losses = []
        
        # Limit validation set for MPS devices to avoid OOM
        validation_loader = self.val_loader
        
        with torch.no_grad():
            for batch in validation_loader:
                try:
                    # Skip empty batches
                    if batch.get('empty_batch', False):
                        continue
                    
                    # Move data to device
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                    
                    # Prepare labels
                    label = batch.get('label')
                    era = batch.get('era')
                    
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Calculate loss
                    loss_dict = self.criterion(
                        outputs,
                        {'days': label, 'era': era},
                        self.global_step,
                        self.config.total_training_steps
                    )
                    val_loss = loss_dict['loss']
                    
                    val_losses.append(val_loss.item())
                    
                    # Clear memory for MPS devices
                    if self.is_mps:
                        if 'outputs' in locals(): del outputs
                        if 'loss_dict' in locals(): del loss_dict
                        if 'val_loss' in locals(): del val_loss
                        if 'label' in batch: del batch['label']
                        if 'era' in batch: del batch['era']
                        del batch
                        gc.collect()
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
        
        # Calculate average validation loss
        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        
        # Log validation loss
        self.writer.add_scalar("val/loss", avg_val_loss, self.global_step)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Return to training mode
        self.model.train()
        
        return avg_val_loss
    
    def _evaluate_swa_model(self):
        """Evaluate the SWA model on the validation set."""
        print("Evaluating SWA model...")
        
        # Update batch norm statistics for SWA model
        with torch.no_grad():
            self.swa_model.eval()
            update_bn(self.train_loader, self.swa_model, device=self.device)
        
        # Validate SWA model
        swa_val_losses = []
        self.swa_model.eval()
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # Skip empty batches
                    if batch.get('empty_batch', False):
                        continue
                    
                    # Move data to device
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                    
                    # Prepare labels
                    label = batch.get('label')
                    era = batch.get('era')
                    
                    # Forward pass
                    outputs = self.swa_model(batch)
                    
                    # Calculate loss
                    loss_dict = self.criterion(
                        outputs,
                        {'days': label, 'era': era},
                        self.global_step,
                        self.config.total_training_steps
                    )
                    val_loss = loss_dict['loss']
                    
                    swa_val_losses.append(val_loss.item())
                
                except Exception as e:
                    print(f"Error during SWA validation: {e}")
                    continue
        
        # Calculate average SWA validation loss
        avg_swa_val_loss = np.mean(swa_val_losses) if swa_val_losses else float("inf")
        
        print(f"SWA model validation loss: {avg_swa_val_loss:.4f}")
        self.writer.add_scalar("val/swa_loss", avg_swa_val_loss, self.global_step)
        
        # Save SWA model if it's better than the best model so far
        if avg_swa_val_loss < self.best_val_loss:
            print(f"SWA model is better than best model ({avg_swa_val_loss:.4f} < {self.best_val_loss:.4f})")
            
            # Save SWA checkpoint
            checkpoint_path = os.path.join(
                self.config.output_dir, "checkpoints", "best_swa_checkpoint.pt"
            )
            torch.save(
                {
                    "model_state_dict": self.swa_model.state_dict(),
                    "global_step": self.global_step,
                    "val_loss": avg_swa_val_loss
                },
                checkpoint_path
            )
            
            # Update best validation loss
            self.best_val_loss = avg_swa_val_loss
    
    def _log_system_info(self):
        """Log system information to TensorBoard."""
        # Log system information
        self.writer.add_text("system/device", str(self.device))
        self.writer.add_text("system/platform", platform.platform())
        self.writer.add_text("system/python", platform.python_version())
        self.writer.add_text("system/pytorch", torch.__version__)
        self.writer.add_text("training/config", str(self.config))
        
        if self.device.type == 'cuda':
            self.writer.add_text("system/cuda", torch.version.cuda)
            self.writer.add_text("system/cudnn", str(torch.backends.cudnn.version()))
            
            # Log GPU info
            self.writer.add_text(
                "system/gpu", 
                f"GPU: {torch.cuda.get_device_name(self.device)}"
            )
            self.writer.add_text(
                "system/gpu_memory",
                f"Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.2f} GB"
            )
        
        elif self.device.type == 'mps':
            self.writer.add_text("system/mps", "Apple Silicon (MPS) acceleration enabled")