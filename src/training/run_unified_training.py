#!/usr/bin/env python3
"""
Unified efficient training script for the Grateful Dead show dating model.
Optimized for CUDA/MPS acceleration with preprocessed tensors.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import sys
import time
import numpy as np
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import PreprocessedDeadShowDataset, HighPerformanceMPSDataset, high_performance_collate_fn
from models.dead_model import DeadShowDatingModel
from models import CombinedDeadLoss
from utils.device_utils import print_system_info
from utils.helpers import ensure_device_consistency


def setup_logger():
    """Set up and configure the logger."""
    logger = logging.getLogger("DeadShowTraining")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_optimal_device():
    """Get the optimal available device for training."""
    # Check for available devices
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    if cuda_available:
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn benchmark mode for faster training
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif mps_available:
        # Configure MPS-specific settings
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_accelerated_dataloaders(config, device, logger):
    """
    Create optimized DataLoader objects for the preprocessed dataset.
    
    Args:
        config: Configuration object
        device: Device to use for training
        logger: Logger instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # First check if the provided path already points to the preprocessed directory
    preprocessed_dir = config.data_dir
    
    # If not found, try extending the path
    if not os.path.exists(preprocessed_dir):
        alternative_path = os.path.join(config.data_dir, "audsnippets-all", "preprocessed")
        if os.path.exists(alternative_path):
            preprocessed_dir = alternative_path
            logger.info(f"Using preprocessed directory at {preprocessed_dir}")
    
    # Check if the directory exists
    if not os.path.exists(preprocessed_dir):
        raise ValueError(f"Preprocessed data directory not found at {preprocessed_dir}. "
                         "Please ensure data has been preprocessed.")
    
    # Set optimal DataLoader parameters based on device
    if device.type == 'mps':
        # For MPS (Apple Silicon)
        Dataset = HighPerformanceMPSDataset
        default_workers = min(8, os.cpu_count() or 4)  # Limit workers to avoid MPS issues
        prefetch_factor = 4
        pin_memory = True
        
        logger.info(f"Using MPS-optimized dataset")
    elif device.type == 'cuda':
        # For CUDA
        Dataset = PreprocessedDeadShowDataset
        default_workers = min(16, os.cpu_count() or 8)  # More workers for CUDA
        prefetch_factor = 4
        pin_memory = True
        
        logger.info(f"Using CUDA-optimized dataset")
    else:
        # For CPU
        Dataset = PreprocessedDeadShowDataset
        default_workers = os.cpu_count() or 4
        prefetch_factor = 2
        pin_memory = False
        
        logger.info(f"Using CPU-optimized dataset")
        
    # Use user-specified worker count if provided
    num_workers = config.num_workers if config.num_workers is not None else default_workers
    persistent_workers = num_workers > 0
    
    logger.info(f"Using {num_workers} dataloader workers")
    
    # Create dataset
    full_dataset = Dataset(
        preprocessed_dir=preprocessed_dir,
        augment=config.use_augmentation,
        target_sr=config.target_sr
    )
    
    # Limit dataset size if specified
    if config.max_samples and config.max_samples < len(full_dataset):
        logger.info(f"Limiting dataset to {config.max_samples} samples (from {len(full_dataset)})")
        indices = list(range(len(full_dataset)))
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(len(full_dataset))[:config.max_samples].tolist()
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split dataset
    train_size = int(len(full_dataset) * config.train_split)
    val_size = int(len(full_dataset) * config.val_split)
    test_size = len(full_dataset) - train_size - val_size
    
    # Use fixed random seeds for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    logger.info(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples")
    
    # Define a top-level collate function for multiprocessing compatibility
    def collate_fn(x):
        return high_performance_collate_fn(x, device=None)
    
    # Create DataLoaders with optimal settings
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "collate_fn": collate_fn,
        "drop_last": True  # Drop incomplete batches for better performance
    }
    
    # Only add prefetch_factor if we have workers
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader


def get_optimized_optimizer(model, config, device):
    """
    Create optimized optimizer based on device.
    
    Args:
        model: The model to optimize
        config: Configuration object
        device: Device to use
        
    Returns:
        Optimizer instance
    """
    # Choose best optimizer based on device
    if device.type == 'cuda':
        # Use fused AdamW for CUDA if available
        try:
            from torch.optim.adamw import AdamW as FusedAdamW
            logger.info("Using Fused AdamW implementation for CUDA")
            optimizer = FusedAdamW(
                model.parameters(),
                lr=config.initial_lr,
                weight_decay=config.weight_decay,
                eps=1e-8,
                foreach=True,  # Use faster foreach implementation
                fused=True     # Enable CUDA fusion for faster updates
            )
        except (ImportError, AttributeError):
            # Fall back to standard AdamW with optimizations
            logger.info("Using standard AdamW with CUDA optimizations")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.initial_lr,
                weight_decay=config.weight_decay,
                eps=1e-8,
                foreach=True  # Use faster foreach implementation if available
            )
    else:
        # Standard optimizer for MPS/CPU
        logger.info(f"Using standard AdamW for {device.type}")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
    
    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
    """
    Create a learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: The optimizer
        config: Configuration object
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        LR scheduler
    """
    total_steps = steps_per_epoch * config.training.num_epochs
    warmup_steps = int(total_steps * 0.1)
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            decay_steps = total_steps - warmup_steps
            cosine_decay = 0.5 * (1 + torch.cos(
                torch.tensor(3.14159 * (step - warmup_steps) / decay_steps)
            ).item())
            return max(0.05, cosine_decay)  # Don't decay below 5% of initial LR
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate(model, val_loader, criterion, device, use_amp, step, total_steps, logger):
    """
    Run validation and return the average loss.
    
    Args:
        model: Model to validate
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        step: Current training step
        total_steps: Total training steps
        logger: Logger instance
        
    Returns:
        Average validation loss
    """
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        for batch in val_pbar:
            # Skip empty batches
            if batch.get('empty_batch', False):
                continue
            
            # Move data to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            
            # Forward pass with AMP if enabled
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
                    loss_dict = criterion(
                        outputs,
                        {'days': batch.get('label'), 'era': batch.get('era')},
                        step,
                        total_steps
                    )
            else:
                outputs = model(batch)
                loss_dict = criterion(
                    outputs,
                    {'days': batch.get('label'), 'era': batch.get('era')},
                    step,
                    total_steps
                )
            
            val_losses.append(loss_dict['loss'].item())
            
            # Clean up memory for MPS
            if device.type == 'mps':
                # Release any large tensors
                del outputs
                del loss_dict
                
                # Extra memory cleanup occasionally
                if len(val_losses) % 10 == 0:
                    gc.collect()
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
    
    # Calculate average loss
    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
    logger.info(f"Validation loss: {avg_val_loss:.4f}")
    
    return avg_val_loss


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, config, device, logger):
    """
    Train the model with optimized settings for the device.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        config: Configuration object
        device: Device to use
        logger: Logger instance
        
    Returns:
        Dictionary with training results
    """
    # Set up TensorBoard writer
    log_dir = os.path.join(config.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logging enabled at {log_dir}")
    
    # Set up for mixed precision training on CUDA
    use_amp = device.type == 'cuda' and config.use_mixed_precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Set up gradient accumulation for better performance
    grad_accum_steps = 1
    if device.type == 'mps':
        grad_accum_steps = 2  # May need accumulation on MPS
    
    if grad_accum_steps > 1:
        logger.info(f"Using gradient accumulation with {grad_accum_steps} steps")
        logger.info(f"Effective batch size: {config.batch_size * grad_accum_steps}")
    
    # Initialize training state
    steps_per_epoch = len(train_loader)
    total_epochs = config.training.num_epochs
    total_steps = steps_per_epoch * total_epochs
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    train_losses = []
    
    # Clean up memory before starting
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    # Calculate validation frequency
    validation_interval = config.validation_interval * steps_per_epoch
    
    # Training loop
    logger.info(f"Starting training for {total_epochs} epochs ({total_steps} steps)")
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_losses = []
        
        # Create progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Skip empty batches
            if batch.get('empty_batch', False):
                continue
            
            # Only zero gradients at the start of accumulation cycle
            if batch_idx % grad_accum_steps == 0:
                optimizer.zero_grad()
            
            # Move data to device with non-blocking transfer
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            
            # Mixed precision training for CUDA
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = model(batch, global_step=global_step)
                    
                    # Calculate loss
                    loss_dict = criterion(
                        outputs,
                        {'days': batch.get('label'), 'era': batch.get('era')},
                        global_step,
                        total_steps
                    )
                    loss = loss_dict['loss']
                
                # Normalize loss for gradient accumulation
                loss = loss / grad_accum_steps
                
                # Backward with scaler
                scaler.scale(loss).backward()
                
                # Update weights at the end of accumulation
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Clip gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_value)
                    
                    # Step optimizer and scaler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                # Standard precision training
                outputs = model(batch, global_step=global_step)
                
                # Calculate loss
                loss_dict = criterion(
                    outputs,
                    {'days': batch.get('label'), 'era': batch.get('era')},
                    global_step,
                    total_steps
                )
                loss = loss_dict['loss']
                
                # Normalize loss for gradient accumulation
                loss = loss / grad_accum_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights at the end of accumulation
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_value)
                    
                    # Step optimizer and scheduler
                    optimizer.step()
                    scheduler.step()
            
            # Record loss (multiplied back to get actual loss)
            actual_loss = loss.item() * grad_accum_steps
            train_losses.append(actual_loss)
            epoch_losses.append(actual_loss)
            
            # Log to TensorBoard
            if global_step % 10 == 0:
                writer.add_scalar('training/loss', actual_loss, global_step)
                writer.add_scalar('training/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            
            # Update progress bar
            avg_loss = sum(epoch_losses[-10:]) / min(len(epoch_losses), 10)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Memory cleanup for MPS
            if device.type == 'mps' and batch_idx % 5 == 0:
                # Clean up tensors
                del outputs
                del loss
                del loss_dict
                
                # More aggressive cleanup occasionally
                if batch_idx % 20 == 0:
                    gc.collect()
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
            
            # Increment global step
            global_step += 1
            
            # Run validation at specified intervals
            if (global_step % validation_interval == 0) or (
                    batch_idx == len(train_loader) - 1 and epoch % config.validation_interval == 0):
                
                val_loss = validate(
                    model, val_loader, criterion, device, use_amp, 
                    global_step, total_steps, logger
                )
                
                # Log validation loss
                writer.add_scalar('validation/loss', val_loss, global_step)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if device.type == 'mps':
                        # Save on CPU for MPS
                        torch.save({
                            'model_state_dict': model.to('cpu').state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                            'val_loss': val_loss
                        }, os.path.join(config.output_dir, "checkpoints", "best_model.pt"))
                        model = model.to(device)  # Move back to device
                    else:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                            'val_loss': val_loss
                        }, os.path.join(config.output_dir, "checkpoints", "best_model.pt"))
                        
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"Validation did not improve. Patience: {patience_counter}/{config.patience}")
                
                # Early stopping
                if config.use_early_stopping and patience_counter >= config.patience:
                    logger.info(f"Early stopping triggered after {patience_counter} validations without improvement")
                    break
                
                # Back to training mode
                model.train()
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        logger.info(f"Epoch {epoch+1}/{total_epochs} completed in {epoch_time:.2f}s with avg loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % config.save_frequency == 0:
            checkpoint_path = os.path.join(
                config.output_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pt"
            )
            
            # Save based on device
            if device.type == 'mps':
                torch.save({
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'val_loss': best_val_loss
                }, checkpoint_path)
                model = model.to(device)  # Move back to device
            else:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'val_loss': best_val_loss
                }, checkpoint_path)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Check for early stopping
        if config.use_early_stopping and patience_counter >= config.patience:
            logger.info("Early stopping triggered. Ending training.")
            break
    
    # Training complete
    writer.close()
    
    # Return training results
    return {
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'epochs_completed': epoch + 1
    }


def evaluate(model, test_loader, criterion, device, use_amp, logger):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        criterion: Loss function
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        logger: Logger instance
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Skip empty batches
            if batch.get('empty_batch', False):
                continue
            
            # Move data to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            
            # Forward pass with AMP if enabled
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
                    loss_dict = criterion(
                        outputs,
                        {'days': batch.get('label'), 'era': batch.get('era')},
                        0, 1  # Step values not used in evaluation
                    )
            else:
                outputs = model(batch)
                loss_dict = criterion(
                    outputs,
                    {'days': batch.get('label'), 'era': batch.get('era')},
                    0, 1  # Step values not used in evaluation
                )
            
            test_losses.append(loss_dict['loss'].item())
            
            # Store predictions and targets
            all_predictions.append(outputs['days'].cpu())
            all_targets.append(batch['label'].cpu())
            
            # Memory cleanup for MPS
            if device.type == 'mps' and len(test_losses) % 10 == 0:
                gc.collect()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
    
    # Calculate metrics
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    test_loss = sum(test_losses) / len(test_losses) if test_losses else float('inf')
    
    # Mean absolute error
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # Calculate additional metrics
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    # Log results
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse:.4f}")
    
    return {
        'test_loss': test_loss,
        'mae': mae,
        'rmse': rmse
    }


def main():
    """Main function to run training."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Grateful Dead show dating model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], 
                        help="Device to use (auto for automatic selection)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--save-frequency", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--validation-interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--tensorboard-dir", type=str, default="./runs", help="TensorBoard log directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    parser.add_argument("--max-samples", type=int, help="Limit the dataset to this many samples for faster training")
    parser.add_argument("--num-workers", type=int, help="Number of dataloader workers (defaults to auto)")
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info("Starting Grateful Dead show dating model training")
    
    # Set up configuration
    config = Config()
    config.batch_size = args.batch_size
    config.initial_lr = args.lr
    config.training.num_epochs = args.epochs
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.resume_checkpoint = args.checkpoint
    config.use_mixed_precision = not args.no_amp
    config.save_frequency = args.save_frequency
    config.validation_interval = args.validation_interval
    config.patience = args.patience
    config.use_early_stopping = not args.no_early_stopping
    config.log_dir = args.tensorboard_dir
    config.debug = args.debug
    config.max_samples = args.max_samples
    config.num_workers = args.num_workers
    
    if args.device != "auto":
        config.device = args.device
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Get device
    device = get_optimal_device() if args.device == "auto" else torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Print system info
    print_system_info()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_accelerated_dataloaders(config, device, logger)
    
    # Initialize model
    model = DeadShowDatingModel(sample_rate=config.target_sr)
    model = model.to(device)
    logger.info(f"Initialized {model.__class__.__name__}")
    
    # Create optimizer
    optimizer = get_optimized_optimizer(model, config, device)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Create loss function
    criterion = CombinedDeadLoss()
    
    # Load checkpoint if specified
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        logger.info(f"Loading checkpoint from {config.resume_checkpoint}")
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Resumed from epoch {checkpoint.get('epoch', 0)}, step {checkpoint.get('global_step', 0)}")
    else:
        if config.resume_checkpoint:
            logger.warning(f"Checkpoint not found at {config.resume_checkpoint}, starting from scratch")
    
    # Train model
    logger.info("Starting training...")
    train_results = train(model, train_loader, val_loader, optimizer, scheduler, criterion, config, device, logger)
    
    # Load best model for evaluation
    best_model_path = os.path.join(config.output_dir, "checkpoints", "best_model.pt")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    logger.info("Evaluating model...")
    use_amp = device.type == 'cuda' and config.use_mixed_precision
    metrics = evaluate(model, test_loader, criterion, device, use_amp, logger)
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "checkpoints", "final_model.pt")
    if device.type == 'mps':
        torch.save({
            'model_state_dict': model.to('cpu').state_dict(),
            'config': config,
            'metrics': metrics,
            'train_results': train_results
        }, final_model_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics,
            'train_results': train_results
        }, final_model_path)
    
    logger.info(f"Final model saved to {final_model_path}")
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {train_results['best_val_loss']:.4f}")
    logger.info(f"Test metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    # Set up logger
    logger = setup_logger()
    main()