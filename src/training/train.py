#!/usr/bin/env python3
"""
Training loop implementation for the Grateful Dead show dating model.
"""

# Set memory limits for MPS before importing PyTorch
import os
# Enable memory stats for debugging
os.environ['PYTORCH_MPS_DEBUG_MEM'] = '1'

import datetime
import platform
import time
import signal
import sys
import random
import glob
from typing import Dict, List, Tuple, Optional
import math
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import DeadShowDatingModel
from models.losses import CombinedDeadLoss
from config import Config
from utils.helpers import reset_parameters, ensure_device_consistency, prepare_batch, ensure_compile_device_consistency, cleanup_file_handles
from data.dataset import (
    DeadShowDataset, 
    PreprocessedDeadShowDataset, 
    optimized_collate_fn, 
    h200_optimized_collate_fn, 
    identity_collate,
    ensure_tensor_consistency,
    get_device,
    MPS_EFFICIENT_DATASET
)
from data.preprocessing import preprocess_dataset
from training.loss import combined_loss
from training.lr_finder import find_learning_rate
from utils.h200_optimizations import optimize_h200_memory, create_cuda_streams, get_optimal_batch_size
from utils.visualization import (
    log_era_confusion_matrix,
    log_error_by_era,
    log_prediction_samples,
)
from models.dtype_helpers import force_dtype, verify_model_dtype, disable_half_precision

# Global flag for graceful interruption
_interrupt_training = False

# Define a specific collate function that works well with MPS
def mps_collate_fn(batch):
    # Filter out error items
    valid_batch = [item for item in batch if not item.get('error', False)]
    
    # If all items were filtered out, return an empty batch signal
    if not valid_batch:
        return {'empty_batch': True}
    
    # For memory efficiency, only keep essential fields in MPS mode
    essential_fields = ['mel_spec', 'mel_spec_percussive', 'label', 'era', 'year', 'date']
    filtered_batch = []
    
    for item in valid_batch:
        # Only keep essential fields to reduce memory usage
        filtered_item = {k: item[k] for k in essential_fields if k in item}
        
        # Convert large tensors to float16 to save memory during collation
        if 'mel_spec' in filtered_item and isinstance(filtered_item['mel_spec'], torch.Tensor):
            filtered_item['mel_spec'] = filtered_item['mel_spec'].to(dtype=torch.float16)
            
        if 'mel_spec_percussive' in filtered_item and isinstance(filtered_item['mel_spec_percussive'], torch.Tensor):
            filtered_item['mel_spec_percussive'] = filtered_item['mel_spec_percussive'].to(dtype=torch.float16)
            
        filtered_batch.append(filtered_item)
    
    # Use the optimized collate function on our filtered batch, keeping tensors on CPU
    result = optimized_collate_fn(filtered_batch, device=None)
    
    # Force immediate garbage collection to free memory
    import gc
    gc.collect()
    
    return result

def handle_interrupt(signum, frame):
    """Signal handler for graceful interruption."""
    global _interrupt_training
    print("\nInterrupt received. Will save checkpoint and exit after current epoch...")
    _interrupt_training = True

def save_checkpoint(
    save_path, model, optimizer, scheduler, swa_scheduler, epoch, global_step, 
    best_val_mae, patience_counter, swa_model=None, scaler=None, using_swa=False,
    curriculum_stage=None, rng_states=None
):
    """
    Save a checkpoint of the training state.
    
    Args:
        save_path: Path to save the checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: LR scheduler to save
        swa_scheduler: SWA scheduler to save
        epoch: Current epoch
        global_step: Current global step
        best_val_mae: Best validation MAE
        patience_counter: Early stopping counter
        swa_model: Stochastic Weight Averaging model
        scaler: Gradient scaler for mixed precision
        using_swa: Whether SWA is in use
        curriculum_stage: Current curriculum stage
        rng_states: Random number generator states
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "swa_scheduler_state_dict": swa_scheduler.state_dict() if swa_scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_mae": best_val_mae,
        "patience_counter": patience_counter,
        "using_swa": using_swa,
        "curriculum_stage": curriculum_stage,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": rng_states.get("numpy") if rng_states else None,
            "python": rng_states.get("python") if rng_states else None
        }
    }
    
    # Save SWA model if available
    if swa_model is not None:
        checkpoint["swa_model_state_dict"] = swa_model.state_dict()
    
    # Save AMP scaler if available
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    # Save to a temporary file first
    temp_path = save_path + ".tmp"
    torch.save(checkpoint, temp_path)
    
    # Atomic rename to ensure checkpoint is not corrupted if interrupted
    if os.path.exists(save_path):
        os.replace(temp_path, save_path)
    else:
        os.rename(temp_path, save_path)
    
    print(f"Checkpoint saved to {save_path}")

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """
    Find the latest checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
        
    # Look for all checkpoint files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not checkpoints:
        return None
        
    # Get the most recently modified checkpoint
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def train_model(config: Config) -> None:
    """
    Main training loop for the Dead Show Dating model.
    
    Args:
        config: Configuration object with training parameters
    """
    # Register signal handlers for graceful interruption
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    
    # Record training start time
    training_start_time = time.time()
    
    # Print system information
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for available devices
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    if mps_available:
        print("Apple Silicon (MPS) is available for acceleration")
    
    # Set up device based on availability and configuration
    if config.device == "auto":
        device = get_device()  # Use the helper to automatically select best device
    else:
        requested_device = config.device
        if requested_device == "cuda" and not cuda_available:
            print("CUDA requested but not available. Checking for MPS...")
            if mps_available:
                print("Falling back to MPS (Apple Silicon)")
                device = torch.device("mps")
            else:
                print("Falling back to CPU")
                device = torch.device("cpu")
        elif requested_device == "mps" and not mps_available:
            print("MPS requested but not available. Checking for CUDA...")
            if cuda_available:
                print("Falling back to CUDA")
                device = torch.device("cuda")
            else:
                print("Falling back to CPU")
                device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        print(f"Primary GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]}")
        if hasattr(torch.cuda, 'get_device_properties'):
            print(f"SM Count: {torch.cuda.get_device_properties(device).multi_processor_count}")
        
        # Force all PyTorch operations to use float32 precision only
        torch.set_default_dtype(torch.float32)
        
        # Explicitly disable half-precision operations
        # Disable automatic mixed precision
        config.use_mixed_precision = False
        
        # Ensure cuDNN doesn't use TF32 or any mixed precision formats
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
    elif device.type == 'mps':
        # MPS-specific optimizations
        print("Configuring for Apple Silicon (MPS) device")
        
        # Force float32 for consistency on MPS
        torch.set_default_dtype(torch.float32)
        
        # Disable mixed precision on MPS for stability
        config.use_mixed_precision = False
        
        # Adjust workers for MPS for better performance
        config.num_workers = min(config.num_workers, os.cpu_count() or 4)
        print(f"Using {config.num_workers} workers for data loading on MPS")
        
        # Set memory usage maximum for MPS (~60% of total system RAM)
        if hasattr(torch, 'mps'):
            try:
                # Add some runtime memory management
                print("MPS memory management setup...")
                
                # Manually clear the MPS cache
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    print("Cleared MPS cache")
            except Exception as e:
                print(f"Failed to configure MPS memory management: {e}")

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log system information
    writer.add_text("system/device", str(device))
    writer.add_text("system/platform", platform.platform())
    writer.add_text("system/python", platform.python_version())
    writer.add_text("system/pytorch", torch.__version__)
    writer.add_text("training/config", str(config))
    
    # Preprocess data if needed (skipped if disable_preprocessing is True)
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
    
    # Load dataset with device-specific optimizations
    if device.type == 'mps':
        print("Using memory-efficient MPS dataset")
        full_dataset = MPS_EFFICIENT_DATASET(
            data_dir,
            augment=config.use_augmentation,
            target_sr=config.target_sr,
            device=None,  # Keep on CPU until needed
        )
    else:
        # Standard dataset for other devices
        full_dataset = PreprocessedDeadShowDataset(
            data_dir,
            augment=config.use_augmentation,
            target_sr=config.target_sr,
            device=None,  # Always keep tensors on CPU in the dataset
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
        
        # Use a fixed batch size for MPS devices
        batch_size = 8  # Use a fixed size of 64 as requested
        print(f"Using batch size of {batch_size} for MPS device")
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
        
        # Safer way to clear MPS cache if possible
        try:
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                print("Successfully cleared MPS cache")
        except Exception as e:
            print(f"Warning: Could not clear MPS cache: {e}")
        
        collate_function = mps_collate_fn
    elif device.type == 'cpu':
        # CPU-specific settings
        num_workers = min(config.num_workers, os.cpu_count() or 2)  # Fewer workers for CPU
        persistent_workers = False  # No need for persistent workers on CPU
        prefetch_factor = 2
        pin_memory = False  # No need for pinned memory on CPU
        collate_function = optimized_collate_fn
        batch_size = config.batch_size
    else:
        # For H200 and other CUDA devices, use workers but keep tensors on CPU during collation
        num_workers = min(config.num_workers, 4)  # Limit workers to reduce memory usage
        persistent_workers = True
        prefetch_factor = 3
        pin_memory = True  # Use pinned memory for faster CPU->GPU transfer
        
        # Use H200 optimized collate function for CUDA devices but keep on CPU
        use_h200_optimized = device.type == "cuda" and hasattr(torch.cuda.get_device_properties(0), 'name') and "H200" in torch.cuda.get_device_properties(0).name
        if use_h200_optimized:
            print("Using H200-optimized collate function for maximum throughput")
            collate_function = lambda batch: h200_optimized_collate_fn(batch, device=None)
        else:
            collate_function = lambda batch: optimized_collate_fn(batch, device=None)
            
        # Base dataloader settings
        batch_size = config.batch_size
        
        # Auto-tune batch size for H200 if enabled
        if use_h200_optimized and len(full_dataset) > 0:
            try:
                optimal_batch_size = get_optimal_batch_size(
                    model=DeadShowDatingModel(sample_rate=config.target_sr, device=device),
                    dataset=full_dataset,
                    initial_batch_size=batch_size,
                    max_memory_usage=0.8
                )
                if optimal_batch_size > batch_size:
                    print(f"Auto-tuned batch size: {optimal_batch_size} (up from {batch_size})")
                    batch_size = optimal_batch_size
                else:
                    print(f"Keeping original batch size: {batch_size}")
            except Exception as e:
                print(f"Error auto-tuning batch size: {e}")
                print(f"Using default batch size: {batch_size}")

    # Create a reusable dictionary of dataloader kwargs with device-specific optimizations
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": min(config.num_workers, 8),  # Limit workers to prevent file descriptor exhaustion
        "pin_memory": pin_memory,
        "collate_fn": collate_function,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "drop_last": False,
    }

    # Create base DataLoaders
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs["shuffle"] = False
    
    # For MPS devices, use an even smaller batch size for validation to prevent OOM
    if device.type == 'mps':
        val_dataloader_kwargs["batch_size"] = 32  # Use a reasonable validation batch size
        print(f"Using validation batch size: {val_dataloader_kwargs['batch_size']}")
    
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)
    
    # Create training dataloader with shuffling
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)

    # Initialize model
    model = DeadShowDatingModel(sample_rate=config.target_sr, device=device)
    
    # Force model to float32 precision, disable any half-precision parameters
    disable_half_precision(model)
    
    # Verify model parameters are all float32
    all_float32, num_mismatched, total, mismatched = verify_model_dtype(model, torch.float32)
    if not all_float32:
        print(f"Warning: {num_mismatched}/{total} parameters not in float32 after initialization")
        for name, dtype, shape in mismatched[:10]:  # Show first 10 mismatched parameters
            print(f"  {name}: {dtype} (shape: {shape})")
    else:
        print(f"All {total} model parameters verified to be float32")

    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=config.weight_decay,
        eps=1e-8,
    )

    # Run learning rate finder if configured
    if config.run_lr_finder:
        print("Running learning rate finder...")
        optimal_lr = find_learning_rate(
            model,
            DataLoader(train_dataset, **dataloader_kwargs),  # Temporary loader
            optimizer,
            combined_loss,
            device,
            start_lr=1e-6,
            end_lr=1e-1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = optimal_lr
        config.initial_lr = optimal_lr
        print(f"Updated learning rate to {optimal_lr}")

    # Set up Stochastic Weight Averaging
    swa_start = int(config.total_training_steps * 0.75)  # Start SWA at 75% of training
    swa_model = AveragedModel(model)
    
    # Set up learning rate scheduler with warmup
    # Calculate warmup steps (10% of total steps)
    warmup_steps = int(config.total_training_steps * 0.1)
    
    # Create custom LR lambda function for warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(step - warmup_steps) / float(max(1, config.total_training_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Set up SWA scheduler that will take over after swa_start
    swa_scheduler = SWALR(
        optimizer, 
        anneal_strategy="cos", 
        anneal_epochs=int(config.total_training_steps * 0.05),  # 5% of total training
        swa_lr=config.initial_lr * 0.1  # SWA typically uses lower LR
    )

    # Initialize mixed precision training if on supported device (CUDA only)
    use_mixed_precision = config.use_mixed_precision and device.type == 'cuda'
    scaler = GradScaler() if use_mixed_precision else None
    
    print(f"Mixed precision training: {use_mixed_precision}")

    # Initialize training state
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Load checkpoint if resuming training
    if config.resume_checkpoint:
        if os.path.exists(config.resume_checkpoint):
            print(f"Loading checkpoint from {config.resume_checkpoint}")
            checkpoint = torch.load(config.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            swa_model = AveragedModel(model)
            
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
                # Move optimizer states to the target device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            if "global_step" in checkpoint:
                global_step = checkpoint["global_step"]
            
            if "best_val_loss" in checkpoint:
                best_val_loss = checkpoint["best_val_loss"]
            
            if "patience_counter" in checkpoint:
                patience_counter = checkpoint["patience_counter"]
            
            print(f"Resumed training from step {global_step}")
        else:
            print(f"Warning: Checkpoint file {config.resume_checkpoint} not found. Starting from scratch.")
    
    # Training loop
    num_epochs = 100  # Use a large number, we'll stop based on steps
    
    # Progress tracking with tqdm
    pbar = tqdm(total=config.total_training_steps, initial=global_step)
    
    # Train until global_step reaches total_training_steps
    early_stopping_triggered = False
    
    # Move model to device
    model = model.to(device)
    swa_model = swa_model.to(device)
    
    # Ensure model is in training mode
    model.train()
    
    for epoch in range(num_epochs):
        if global_step >= config.total_training_steps or early_stopping_triggered:
            break
        
        model.train()
        train_losses = []

        for batch in train_loader:
            try:
                # Skip empty batches (where all items were filtered out due to errors)
                if 'empty_batch' in batch and batch['empty_batch']:
                    print("Skipping empty batch (all items had errors)")
                    continue
                    
                # Explicitly move all tensors to the target device
                batch = ensure_tensor_consistency(batch, target_dtype=torch.float32, device=device)
                
                # For MPS, aggressive memory management
                if device.type == 'mps':
                    # For MPS, ensure only the necessary tensors are kept in GPU memory
                    required_keys = ['mel_spec', 'mel_spec_percussive', 'label', 'era', 'year']
                    # Keep only required keys to minimize memory footprint
                    batch = {k: v for k, v in batch.items() if k in required_keys}
                    
                    # Clear active MPS memory before forward pass
                    try:
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                
                # Convert label and era to tensors if necessary
                if 'label' in batch and not isinstance(batch['label'], torch.Tensor):
                    batch['label'] = torch.tensor(batch['label'], device=device)
                if 'era' in batch and not isinstance(batch['era'], torch.Tensor):
                    batch['era'] = torch.tensor(batch['era'], device=device)
                
                # Verify that all data is on the correct device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.device != device:
                        batch[key] = value.to(device)
                
                # Extract label and era from batch
                label = batch.get('label')
                era = batch.get('era')
                
                # Reset gradients
                optimizer.zero_grad()
                
                if use_mixed_precision:
                    with torch.amp.autocast(device_type=device.type):
                        # Ensure model parameters are consistent with autocast dtype
                        try:
                            outputs = model(batch)
                            
                            # Verify batch size consistency before loss calculation
                            if outputs['days'].shape[0] != label.shape[0]:
                                print(f"Batch size mismatch: model output {outputs['days'].shape[0]}, label {label.shape[0]}")
                                continue
                                
                            # Use combined_loss with global_step and total_steps
                            loss = combined_loss(
                                outputs, label, era, global_step, config.total_training_steps
                            )
                        except IndexError as idx_err:
                            print(f"IndexError in model forward pass or loss calculation: {idx_err}")
                            print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)]}")
                            # Skip this batch and continue training
                            continue
                        except Exception as e:
                            print(f"Unexpected error during forward pass: {e}")
                            continue
                        
                    # Scale gradients and backward pass
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("Warning: NaN/Inf detected in loss - skipping batch")
                        continue
                        
                    scaler.scale(loss).backward()
                    
                    # Check for valid gradients
                    valid_gradients = True
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if (
                                torch.isnan(param.grad).any()
                                or torch.isinf(param.grad).any()
                            ):
                                print(f"NaN/Inf gradient detected in {name}")
                                valid_gradients = False
                                break

                    if valid_gradients:
                        # Unscale before gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip_value
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        print("Skipping optimizer step due to invalid gradients")
                else:
                    # Standard training without mixed precision
                    try:
                        outputs = model(batch)
                        
                        # Verify batch size consistency before loss calculation
                        if outputs['days'].shape[0] != label.shape[0]:
                            print(f"Batch size mismatch: model output {outputs['days'].shape[0]}, label {label.shape[0]}")
                            continue
                            
                        # Use combined_loss with global_step and total_steps
                        loss = combined_loss(
                            outputs, label, era, global_step, config.total_training_steps
                        )
                    except IndexError as idx_err:
                        print(f"IndexError in model forward pass or loss calculation: {idx_err}")
                        print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)]}")
                        # Skip this batch and continue training
                        continue
                    except Exception as e:
                        print(f"Unexpected error during forward pass: {e}")
                        continue
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("Warning: NaN/Inf detected in loss - skipping batch")
                        continue
                        
                    # Standard backward pass
                    loss.backward()
                    
                    # Check for valid gradients
                    valid_gradients = True
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if (
                                torch.isnan(param.grad).any()
                                or torch.isinf(param.grad).any()
                            ):
                                print(f"NaN/Inf gradient detected in {name}")
                                valid_gradients = False
                                break
                    
                    if valid_gradients:
                        # Clip gradients to avoid explosive gradients
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip_value
                        )
                        optimizer.step()
                    else:
                        print("Skipping optimizer step due to invalid gradients")
                
                # Update SWA if we've reached the SWA start point
                if global_step >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

                # For MPS, aggressively release all unnecessary tensors after the training step
                if device.type == 'mps':
                    # Delete any references to intermediate tensors
                    del outputs, loss
                    if 'label' in batch: del batch['label']
                    if 'era' in batch: del batch['era']
                    
                    # Free the whole batch once used
                    del batch
                    
                    # Force garbage collection after every step
                    import gc
                    gc.collect()
                    
                    # Aggressively clear MPS cache
                    try:
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except Exception:
                        pass

                # Logging
                if global_step % config.log_interval == 0:
                    # Make sure loss exists before trying to access it
                    if 'loss' in locals():
                        writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "train/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                
                # Keep track of losses for epoch average
                if 'loss' in locals():
                    train_losses.append(loss.item())
                
                # Update progress bar
                avg_loss = np.mean(train_losses[-100:]) if train_losses else 0
                pbar.set_description(
                    f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                pbar.update(1)
                
                # Clean up memory for MPS devices
                if device.type == 'mps':
                    # Free up memory
                    try:
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except Exception as e:
                        print(f"Warning: Could not clear MPS cache during training: {e}")
                    # Force garbage collection
                    import gc
                    gc.collect()
                
                # Update global step
                global_step += 1
                
                # Check if we've reached the total training steps
                if global_step >= config.total_training_steps:
                    break
                    
                # Check for graceful interruption
                if _interrupt_training:
                    print("Interrupting training due to signal...")
                    break
                    
            except Exception as e:
                print(f"Error during training step: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Validation after each epoch
        if epoch % config.validation_interval == 0:
            model.eval()
            val_losses = []
            
            # For MPS, validate on a smaller subset to avoid OOM
            if device.type == 'mps':
                # Sample a reasonable validation subset (25% of validation data)
                val_subset_size = min(5000, len(val_dataset) // 4)
                val_subset_indices = torch.randperm(len(val_dataset))[:val_subset_size]
                val_subset = Subset(val_dataset, val_subset_indices)
                print(f"Using validation subset of {val_subset_size} samples for MPS device")
                
                # Create a validation loader with reasonable batch size
                small_val_loader = DataLoader(
                    val_subset, 
                    batch_size=32,  # Use a reasonable batch size
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    collate_fn=collate_function
                )
                validation_loader = small_val_loader
            else:
                validation_loader = val_loader
            
            # Print memory usage if on MPS
            if device.type == 'mps' and hasattr(torch, 'mps'):
                try:
                    # Try to get memory info
                    print("MPS memory status before validation:")
                    print(f"  Current allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
                    if hasattr(torch.mps, 'driver_allocated_memory'):
                        print(f"  Driver allocated: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
                except Exception as e:
                    print(f"Could not get MPS memory info: {e}")
            
            with torch.no_grad():
                for batch in validation_loader:
                    try:
                        # Skip empty batches
                        if 'empty_batch' in batch and batch['empty_batch']:
                            continue
                            
                        # Move data to device
                        batch = ensure_tensor_consistency(batch, target_dtype=torch.float32, device=device)
                        
                        # Forward pass
                        try:
                            outputs = model(batch)
                            
                            # Extract label and era
                            label = batch.get('label')
                            era = batch.get('era')
                            
                            # Verify batch size consistency before loss calculation
                            if outputs['days'].shape[0] != label.shape[0]:
                                print(f"Validation batch size mismatch: model output {outputs['days'].shape[0]}, label {label.shape[0]}")
                                continue
                            
                            # Calculate loss
                            val_loss = combined_loss(
                                outputs, label, era, global_step, config.total_training_steps
                            )
                            
                            val_losses.append(val_loss.item())
                        except IndexError as idx_err:
                            print(f"IndexError in validation forward pass: {idx_err}")
                            print(f"Validation batch shapes: {[(k, v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)]}")
                            # Skip this batch and continue validation
                            continue
                        except Exception as e:
                            print(f"Unexpected error during validation: {e}")
                            continue
                        
                        # Clear memory after each validation batch for MPS devices
                        if device.type == 'mps':
                            # Free up memory
                            del outputs
                            try:
                                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                                    torch.mps.empty_cache()
                            except Exception as e:
                                print(f"Warning: Could not clear MPS cache during validation: {e}")
                            # Force garbage collection
                            import gc
                            gc.collect()
                    except Exception as e:
                        print(f"Error during validation: {e}")
                        continue
            
            # Calculate average validation loss
            avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
            
            # Log validation loss
            writer.add_scalar("val/loss", avg_val_loss, global_step)
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                # Save best model
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(
                    config.output_dir, "checkpoints", f"best_checkpoint.pt"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
                
                print(f"New best model with validation loss {best_val_loss:.4f} saved to {checkpoint_path}")
            else:
                patience_counter += 1
                
                if config.use_early_stopping and patience_counter >= config.patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    early_stopping_triggered = True
                    break
        
        # Save periodic checkpoint
        if epoch % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                config.output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
    
    # Final validation with SWA model if we did enough steps for SWA
    if global_step >= swa_start:
        print("Evaluating SWA model...")
        
        # Update batch norm statistics for SWA model
        with torch.no_grad():
            swa_model.eval()
            # Use training loader to update batch norm stats
            update_bn(train_loader, swa_model, device=device)
        
        # Validate SWA model
        swa_val_losses = []
        swa_model.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Skip empty batches
                    if 'empty_batch' in batch and batch['empty_batch']:
                        continue
                        
                    # Move data to device
                    batch = ensure_tensor_consistency(batch, target_dtype=torch.float32, device=device)
                    
                    # Forward pass
                    outputs = swa_model(batch)
                    
                    # Extract label and era
                    label = batch.get('label')
                    era = batch.get('era')
                    
                    # Calculate loss
                    val_loss = combined_loss(
                        outputs, label, era, global_step, config.total_training_steps
                    )
                    
                    swa_val_losses.append(val_loss.item())
                except Exception as e:
                    print(f"Error during SWA validation: {e}")
                    continue
        
        # Calculate average SWA validation loss
        avg_swa_val_loss = np.mean(swa_val_losses) if swa_val_losses else float("inf")
        
        print(f"SWA model validation loss: {avg_swa_val_loss:.4f}")
        writer.add_scalar("val/swa_loss", avg_swa_val_loss, global_step)
        
        # Save SWA model if it's better than the best model so far
        if avg_swa_val_loss < best_val_loss:
            print(f"SWA model is better than best model ({avg_swa_val_loss:.4f} < {best_val_loss:.4f})")
            
            # Save SWA checkpoint
            checkpoint_path = os.path.join(
                config.output_dir, "checkpoints", f"best_swa_checkpoint.pt"
            )
            torch.save(
                {
                    "model_state_dict": swa_model.state_dict(),
                    "global_step": global_step,
                    "best_val_loss": avg_swa_val_loss,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
    
    # Save final model
    final_checkpoint_path = os.path.join(
        config.output_dir, "checkpoints", f"final_checkpoint.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "epoch": epoch,
        },
        final_checkpoint_path,
    )
    
    # Print final training information
    training_time = time.time() - training_start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time / 3600:.2f} hours)")
    print(f"Reached step {global_step} of {config.total_training_steps}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Clean up any resources
    writer.close()
    pbar.close()
    cleanup_file_handles()  # Close any open file handles
