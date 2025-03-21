#!/usr/bin/env python3
"""
Unified training script for the Grateful Dead show dating model.
Implements optimizations for different hardware and batch sizes.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import gc
import numpy as np
from pathlib import Path
import time

# Add the src directory to the path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, src_dir)

from src.config import Config
from src.models.dead_model import DeadShowDatingModel
from src.models import CombinedDeadLoss
from src.data.dataset import (
    PreprocessedDeadShowDataset, 
    MemoryEfficientMPSDataset,
    HighPerformanceMPSDataset, 
    optimized_collate_fn, 
    h200_optimized_collate_fn
)
from src.utils.device_utils import print_system_info

# Define picklable collate functions at module level
def collate_with_device_none(batch):
    """Picklable collate function that doesn't move tensors to device."""
    return optimized_collate_fn(batch, device=None)
    
# Define high-performance collate function for better throughput
def high_performance_collate_with_device_none(batch):
    """High-performance collate function with optimizations for modern hardware."""
    if 'high_performance_collate_fn' in globals():
        return high_performance_collate_fn(batch, device=None)
    else:
        # Fall back to standard optimized collate function
        return optimized_collate_fn(batch, device=None)

def h200_collate_with_device_none(batch):
    """Picklable collate function for H100/H200 GPUs that doesn't move tensors to device."""
    return h200_optimized_collate_fn(batch, device=None)


def create_dataloaders(config, device, max_samples=None, auto_tune=True):
    """
    Create train and validation dataloaders optimized for the target device.
    
    Args:
        config: Configuration object
        device: Target device for training
        max_samples: Maximum number of samples to use (for testing)
        auto_tune: Whether to auto-calibrate resource usage (default: True)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print(f"Creating dataloaders for {device} device")
    
    # Auto-tuning system to calibrate resource usage
    if auto_tune and device.type in ['cuda', 'mps']:
        print("Auto-calibrating resource usage...")
        
        # Check if num_workers was manually specified
        user_specified_workers = hasattr(config, 'num_workers') and config.num_workers is not None
        if user_specified_workers:
            print(f"Using user-specified num_workers: {config.num_workers}")
        
        # Dynamically adjust batch size based on device memory
        if device.type == 'cuda':
            # For CUDA, get memory capacity
            total_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
            free_memory_gb = (torch.cuda.get_device_properties(device).total_memory - 
                            torch.cuda.memory_reserved()) / 1024**3
            
            print(f"CUDA memory: {free_memory_gb:.2f}GB free of {total_memory_gb:.2f}GB total")
            
            # Adjust config based on available memory
            # Use 95% of memory to maximize utilization
            usable_memory_gb = free_memory_gb * 0.95
            
            # Drastically more aggressive memory estimate to utilize more GPU
            estimated_mb_per_sample = 8  # Even lower estimate to increase batch size
            
            # Calculate max batch size - more aggressive to maximize utilization
            max_batch_size = int(usable_memory_gb * 1024 / estimated_mb_per_sample)
            
            # Apply constraints and set batch size - significantly increased caps
            optimal_batch_size = min(max_batch_size, 512)  # Cap increased to 512
            optimal_batch_size = max(optimal_batch_size, 32)  # Minimum increased to 32
            
            # Only set worker count if not specified by user
            if not user_specified_workers:
                # Optimize worker count to reduce CPU bottlenecks 
                # Fewer workers to prevent CPU spikes but ensure enough for data feeding
                cpu_count = os.cpu_count() or 4
                # Key change: Keep workers very low to prevent CPU spikes
                optimal_workers = min(4, max(2, cpu_count // 4))
                config.num_workers = optimal_workers
            
            # Critical change: Very low prefetch to prevent memory fluctuations
            # This is likely the main cause of memory swings
            optimal_prefetch = 2  # Fixed low value to prevent memory spikes
            
            # Enable pin memory for faster transfers
            config.pin_memory = True
            
            # Enable async data loading
            config.non_blocking = True
            
            print(f"Auto-tuned CUDA parameters: batch_size={optimal_batch_size}, workers={config.num_workers}, prefetch={optimal_prefetch}")
            
            # Update config with auto-tuned values
            config.batch_size = optimal_batch_size
            config.prefetch_factor = optimal_prefetch
            
        elif device.type == 'mps':
            # For MPS (Apple Silicon)
            # MPS doesn't have easy memory query, so use system memory as proxy
            import psutil
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            total_memory_gb = system_memory.total / 1024**3
            free_memory_gb = system_memory.available / 1024**3
            
            print(f"System memory: {free_memory_gb:.2f}GB free of {total_memory_gb:.2f}GB total")
            
            # Ultra high-performance MPS parameters for high-memory systems (128GB)
            if free_memory_gb > 100:
                # Ultra high-end Mac with 128GB RAM
                optimal_batch_size = min(64, config.batch_size * 2)  # Much larger batch size
                optimal_workers = 8  # Significantly more workers
                prefetch_multiplier = 8  # Aggressive prefetching
                print(f"Detected high-memory Mac with {free_memory_gb:.0f}GB RAM - using ultra high-performance settings")
            elif free_memory_gb > 64:
                # Very high-end Mac with 64GB+ RAM
                optimal_batch_size = min(48, config.batch_size * 2)  
                optimal_workers = 6  # More workers for throughput
                prefetch_multiplier = 6
            elif free_memory_gb > 32:
                # High-end Mac with lots of RAM (M1 Max/Ultra or higher)
                optimal_batch_size = min(32, config.batch_size)
                optimal_workers = 4
                prefetch_multiplier = 4
            elif free_memory_gb > 16:
                # Mid-range Mac (M1 Pro or similar)
                optimal_batch_size = min(24, config.batch_size)
                optimal_workers = 2
                prefetch_multiplier = 2
            else:
                # Base model Mac with limited RAM
                optimal_batch_size = min(16, config.batch_size)
                optimal_workers = 1
                prefetch_multiplier = 1
            
            # Only set worker count if not specified by user
            if not user_specified_workers:
                config.num_workers = optimal_workers
                
            print(f"Auto-tuned MPS parameters: batch_size={optimal_batch_size}, workers={config.num_workers}")
            
            # Update config, respecting user specified values
            config.batch_size = optimal_batch_size
            config.prefetch_factor = prefetch_multiplier
            
            # Set aggressive memory management for all MPS devices
            config.aggressive_memory = True
    
    # Determine which dataset class to use based on device
    if device.type == 'mps' and config.aggressive_memory:
        print("Using memory-efficient MPS dataset")
        dataset_class = MemoryEfficientMPSDataset
    else:
        dataset_class = PreprocessedDeadShowDataset
    
    # Load dataset
    try:
        full_dataset = dataset_class(
            config.data_dir,
            augment=config.use_augmentation,
            target_sr=config.target_sr,
            # Keep tensors on CPU initially for dataloader compatibility
            device=None 
        )
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        print(f"Checking alternative data paths...")
        
        # Try alternative paths
        alt_paths = [
            os.path.join(config.data_dir, "preprocessed"),
            os.path.join(os.path.dirname(config.data_dir), "preprocessed"),
            os.path.join(os.path.dirname(os.path.dirname(config.data_dir)), "preprocessed")
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                print(f"Found alternative data path: {path}")
                full_dataset = dataset_class(
                    path,
                    augment=config.use_augmentation,
                    target_sr=config.target_sr,
                    device=None
                )
                break
        else:
            print("Could not find valid data directory. Available directories:")
            for path in [config.data_dir] + alt_paths:
                print(f"- {path}: {'exists' if os.path.exists(path) else 'does not exist'}")
            raise ValueError(f"No valid preprocessed data found. Please check data paths.")
    
    # Limit dataset size if requested (for testing/debugging)
    if max_samples is not None and max_samples < len(full_dataset):
        print(f"Limiting dataset to {max_samples} samples (from {len(full_dataset)})")
        indices = torch.randperm(len(full_dataset))[:max_samples]
        subset_indices = indices.tolist()
        
        # Create a subset with only the specified indices
        from torch.utils.data import Subset
        full_dataset = Subset(full_dataset, subset_indices)
    
    # Split dataset
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size
    
    # Create train/validation splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Training with {len(train_dataset)} samples")
    print(f"Validation with {len(val_dataset)} samples")
    
    # Determine batch size and other parameters based on device type
    if device.type == 'mps':
        # MPS (Apple Silicon) optimizations
        # Limited optimizations while respecting user choices
        pin_memory = False  # No pin_memory for MPS
        persistent_workers = False
        prefetch_factor = 2 if config.num_workers > 0 else None
        # Use user-specified batch size and workers
        batch_size = config.batch_size
        num_workers = config.num_workers
        
        print(f"Using MPS settings: workers={num_workers}, batch_size={batch_size}")
        
    elif device.type == 'cuda':
        # CUDA optimizations - dynamically scale based on available memory
        available_memory = 0
        try:
            # Try to get available GPU memory
            available_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        except Exception:
            pass
            
        # Scale workers and batch size based on available memory
        # Key changes: Much larger batch sizes, fewer workers to prevent CPU spikes
        if available_memory > 40:  # Very high-end GPU (>40GB)
            num_workers = 2  # Drastically reduced workers
            pin_memory = True  
            batch_size = min(512, config.batch_size)  # Allow much larger batches
            persistent_workers = True
            prefetch_factor = 2  # Lower prefetching to prevent memory fluctuations
        elif available_memory > 24:  # High-end GPU (24-40GB)
            num_workers = 2  # Drastically reduced workers
            pin_memory = True
            batch_size = min(384, config.batch_size)  # Much larger batches
            persistent_workers = True
            prefetch_factor = 2  # Lower prefetching to prevent memory fluctuations
        elif available_memory > 12:  # Mid-range GPU (12-24GB)
            num_workers = 2  # Drastically reduced workers
            pin_memory = True
            batch_size = min(256, config.batch_size)  # Much larger batches
            persistent_workers = True
            prefetch_factor = 2
        else:  # Low-end GPU (<12GB)
            num_workers = 2  # Drastically reduced workers
            pin_memory = True
            batch_size = min(128, config.batch_size)  # Larger batches
            persistent_workers = True
            prefetch_factor = 2
            
        print(f"Using optimized CUDA settings: workers={num_workers}, batch_size={batch_size}")
            
    else:
        # CPU fallback
        cpu_count = os.cpu_count() or 2
        num_workers = min(cpu_count - 1, 4)  # Leave 1 core free for system
        pin_memory = False
        batch_size = min(16, config.batch_size)
        persistent_workers = True if num_workers > 0 else False
        prefetch_factor = 2
        
        print(f"Using CPU settings: workers={num_workers}, batch_size={batch_size}")
    
    # Choose collate function based on device
    # Select the appropriate collate function for maximum performance
    if device.type == 'cuda' and hasattr(torch.cuda, 'get_device_name'):
        device_name = torch.cuda.get_device_name(device)
        if 'h100' in device_name.lower() or 'h200' in device_name.lower():
            print(f"Using H100/H200 optimized collate function")
            selected_collate_fn = h200_collate_with_device_none
        else:
            # Use high-performance collate for all other CUDA devices
            print(f"Using high-performance optimized collate function for CUDA")
            selected_collate_fn = high_performance_collate_with_device_none
    elif device.type == 'mps':
        # For Apple Silicon with plenty of RAM, use our high-performance variant
        print(f"Using high-performance optimized collate function for MPS")
        selected_collate_fn = high_performance_collate_with_device_none
    else:
        # Standard collate for CPU
        selected_collate_fn = collate_with_device_none
    
    print(f"Using {selected_collate_fn.__name__} collate function")
    
    # Keep the user-specified number of workers even for MPS
    
    # Configure dataloader kwargs with optimized settings
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': selected_collate_fn,
        'persistent_workers': persistent_workers if num_workers > 0 else False,
        'drop_last': True,  # Drop last incomplete batch to ensure consistent sizes
    }
    
    # Add PyTorch 2.0+ options if available
    if hasattr(torch, '__version__') and int(torch.__version__.split('.')[0]) >= 2:
        # These options only work in PyTorch 2.0+
        if pin_memory and device.type == 'cuda':
            dataloader_kwargs['pin_memory_device'] = str(device)
            
        if device.type == 'cuda' and num_workers > 0:
            dataloader_kwargs['multiprocessing_context'] = 'spawn'
    
    # Add prefetch_factor if num_workers > 0 and we have it defined
    if num_workers > 0 and 'prefetch_factor' in locals():
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        print(f"Using prefetch_factor={prefetch_factor}")
    
    # Special optimization for CUDA - enable timeout for deadlock prevention
    if device.type == 'cuda':
        # default timeout in seconds
        dataloader_kwargs['timeout'] = 300  # 5 minutes
    
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset,
        **dataloader_kwargs
    )
    
    # Create validation dataloader - use same batch size as training
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs['shuffle'] = False
    
    val_loader = DataLoader(
        val_dataset,
        **val_dataloader_kwargs
    )
    
    return train_loader, val_loader


class OptimizedTrainer:
    """
    Unified trainer with optimizations for different hardware.
    
    Args:
        config: Training configuration
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
        tensorboard_dir: Directory for TensorBoard logs
        auto_tune: Whether to automatically tune resource usage during training
    """
    def __init__(self, config, train_loader, val_loader, device, tensorboard_dir=None, auto_tune=True):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.auto_tune = auto_tune
        self.tensorboard_dir = tensorboard_dir
        
        # Set up output directories
        self.output_dir = self.config.output_dir
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        # Create tensorboard writer
        if self.tensorboard_dir:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join(self.tensorboard_dir, current_time)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
        
        # Resource monitoring data
        self.gpu_util_samples = []
        self.memory_util_samples = []
        self.batch_time_samples = []
        self.last_resource_check = time.time()
        self.resource_check_interval = 30  # seconds
        
        # Dynamic training parameters
        self.dynamic_batch_size = self.config.batch_size
        self.prefetching_factor = 2
        
        # Enable memory-efficient optimizations
        self.setup_memory_optimizations()
        
        # Initialize model
        print(f"Initializing model on {self.device}")
        self.model = DeadShowDatingModel(sample_rate=self.config.target_sr, device=self.device)
        self.model.to(self.device)
        
        # Setup resource monitoring for auto-tuning
        if self.auto_tune:
            # Import monitoring library if available
            try:
                # For CUDA, we can use pynvml for direct GPU monitoring
                if self.device.type == 'cuda':
                    import pynvml
                    try:
                        pynvml.nvmlInit()
                        self.nvml_initialized = True
                        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index if hasattr(self.device, 'index') else 0)
                        print("NVIDIA Management Library initialized for GPU monitoring")
                    except:
                        self.nvml_initialized = False
                        print("Could not initialize NVIDIA Management Library")
                
                # For MPS, use psutil for system memory monitoring
                elif self.device.type == 'mps':
                    import psutil
                    self.psutil_available = True
                    print("Using psutil for system resource monitoring")
                else:
                    self.nvml_initialized = False
                    self.psutil_available = False
            except ImportError:
                self.nvml_initialized = False
                self.psutil_available = False
                print("Resource monitoring libraries not available, auto-tuning will be limited")
        
        # Initialize optimizer with best parameters for device
        if self.device.type == 'mps':
            # For MPS, Adam works better than AdamW
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.initial_lr,
                weight_decay=self.config.weight_decay,
                eps=1e-8,
            )
        else:
            # For CUDA and CPU, AdamW is better
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.initial_lr,
                weight_decay=self.config.weight_decay,
                eps=1e-8,
            )
        
        # LR scheduler with warmup and cosine decay
        self.warmup_steps = int(self.config.total_training_steps * 0.1)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: self._lr_schedule(step)
        )
        
        # Initialize loss function
        self.criterion = CombinedDeadLoss()
        
        # Set up scaler for mixed precision if available and enabled
        self.use_amp = (
            self.config.use_mixed_precision and
            self.device.type == 'cuda' and
            torch.cuda.is_available()
        )
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training")
        
        # Store training config summary
        self.print_training_config()
        
    def setup_memory_optimizations(self):
        """Set up memory optimizations based on device type."""
        # For CUDA devices, set memory allocation options
        if self.device.type == 'cuda':
            # Enable tensor cores for mixed precision if available
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                
            # Enable cudnn benchmarking for faster convolutions
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                
            # Set memory allocation strategy for less fragmentation
            if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'memory'):
                # Use more aggressive allocation strategy
                if hasattr(torch.cuda.memory, 'set_allocator_settings'):
                    try:
                        # These settings minimize fragmentation
                        torch.cuda.memory.set_allocator_settings('max_split_size_mb:128')
                    except:
                        pass
                        
            # Enable tensor memory management for torch >= 2.0
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
                
        # For MPS, set memory options
        elif self.device.type == 'mps':
            # MPS-specific optimizations can be added here
            pass
    
    def _lr_schedule(self, step):
        """Custom learning rate schedule with warmup and cosine decay."""
        if step < self.warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            decay_steps = max(1, self.config.total_training_steps - self.warmup_steps)
            progress = float(step - self.warmup_steps) / float(decay_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    def print_training_config(self):
        """Print a summary of the training configuration."""
        print("\n======= TRAINING CONFIGURATION =======")
        print(f"Device: {self.device}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Batch Size: {self.train_loader.batch_size}")
        print(f"Learning Rate: {self.config.initial_lr}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Training Steps: {self.config.total_training_steps}")
        print(f"Checkpoint Dir: {os.path.join(self.output_dir, 'checkpoints')}")
        if self.writer:
            print(f"TensorBoard: {self.log_dir}")
        print("======================================\n")
    
    def move_batch_to_device(self, batch):
        """Move batch data to the target device efficiently with memory optimizations."""
        # Skip empty batches
        if batch.get('empty_batch', False):
            return batch
        
        # Process core features first, one at a time to prevent memory spikes
        core_features = ['harmonic', 'percussive', 'chroma', 'spectral_contrast', 'label', 'era']
        
        result = {}
        # Move core feature tensors to device
        for key in core_features:
            if key in batch and isinstance(batch[key], torch.Tensor):
                # Ensure float32 (or appropriate) precision
                if batch[key].dtype != torch.float32 and batch[key].dtype not in [torch.long, torch.int64]:
                    # Convert to float32 before moving to device
                    result[key] = batch[key].to(dtype=torch.float32, device=self.device, non_blocking=True)
                else:
                    # Keep original dtype for integer-type tensors
                    result[key] = batch[key].to(self.device, non_blocking=True)
                
                # Ensure tensor is contiguous for better performance
                if not result[key].is_contiguous():
                    result[key] = result[key].contiguous()
        
        # Process remaining tensors
        for key, value in batch.items():
            if key not in core_features:
                if isinstance(value, torch.Tensor):
                    # Optimize memory usage by using appropriate data types
                    if value.dtype != torch.float32 and value.dtype not in [torch.long, torch.int64]:
                        result[key] = value.to(dtype=torch.float32, device=self.device, non_blocking=True)
                    else:
                        result[key] = value.to(self.device, non_blocking=True)
                else:
                    # For non-tensor values, just copy
                    result[key] = value
        
        # Clean up to free memory
        del batch
        
        return result
    
    def save_checkpoint(self, path, is_best=False):
        """Save a checkpoint of the model and training state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": self.config,
        }
        
        # Save to temp file first to prevent corruption if interrupted
        temp_path = f"{path}.tmp"
        torch.save(checkpoint, temp_path)
        
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
        
        # Load checkpoint to CPU first (safer for MPS compatibility)
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Move model to device after loading weights
        self.model.to(self.device)
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]
        
        print(f"Resumed training from step {self.global_step}")
        return True
    
    def train(self, max_steps=None, max_epochs=None):
        """
        Run the training loop.
        
        Args:
            max_steps: Maximum number of training steps (optional)
            max_epochs: Maximum number of epochs (optional)
        """
        # Use provided max_steps or config value
        total_steps = max_steps or self.config.total_training_steps
        
        # If both max_epochs and max_steps are specified, prioritize max_epochs
        steps_are_primary = max_epochs is None
        
        # Track training start time
        start_time = time.time()
        epoch = 0
        
        # Progress tracking variables
        last_time_check = time.time()
        samples_processed = 0
        step_times = []
        preload_next_batch = True  # Enable batch preloading
        
        # Import tqdm for progress bar
        from tqdm import tqdm
        
        # For GPU memory monitoring
        gpu_mem_allocated = []
        
        try:
            # Keep training until we reach max_epochs (if specified) or total_steps
            # When max_epochs is specified, it takes priority over total_steps
            while ((steps_are_primary and self.global_step < total_steps) or
                  (not steps_are_primary and epoch < max_epochs)):
                
                epoch += 1
                print(f"\n=== Epoch {epoch}/{max_epochs if max_epochs else 'inf'} ===")
                
                # Training loop
                self.model.train()
                epoch_losses = []
                
                # Create progress bar for this epoch
                batch_count = len(self.train_loader)
                if steps_are_primary:
                    # Steps-based progress tracking
                    progress_total = min(batch_count, total_steps - self.global_step)
                else:
                    # Epoch-based progress tracking (just show progress within current epoch)
                    progress_total = batch_count
                    
                progress_bar = tqdm(total=progress_total, desc=f"Epoch {epoch}")
                
                # Preload first batch asynchronously if DataLoader supports it
                if hasattr(self.train_loader, '_prefetch_factor') and self.train_loader._prefetch_factor > 1:
                    batch_iter = iter(self.train_loader)
                    try:
                        next_batch = next(batch_iter)
                    except StopIteration:
                        # Dataset is empty
                        break
                        
                # Main training loop
                for batch_idx, batch in enumerate(self.train_loader):
                    # Safety check for memory utilization (only for MPS/CUDA)
                    if self.device.type in ['mps', 'cuda']:
                        if self.device.type == 'cuda':
                            mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                            mem_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                        else:  # mps
                            try:
                                # Not all PyTorch versions support these functions for MPS
                                if hasattr(torch.mps, 'current_allocated_memory'):
                                    mem_allocated = torch.mps.current_allocated_memory() / 1024**3  # GB
                                else:
                                    mem_allocated = 0
                                    
                                if hasattr(torch.mps, 'driver_allocated_memory'):
                                    mem_cached = torch.mps.driver_allocated_memory() / 1024**3  # GB
                                else:
                                    mem_cached = 0
                            except Exception:
                                mem_allocated = 0
                                mem_cached = 0
                                
                        # Store for monitoring
                        gpu_mem_allocated.append(mem_allocated)
                        
                        # Log memory stats periodically
                        if self.global_step % 10 == 0:
                            # Calculate memory efficiency metrics
                            avg_mem = np.mean(gpu_mem_allocated[-10:]) if len(gpu_mem_allocated) >= 10 else 0
                            max_mem = np.max(gpu_mem_allocated[-10:]) if len(gpu_mem_allocated) >= 10 else 0
                            min_mem = np.min(gpu_mem_allocated[-10:]) if len(gpu_mem_allocated) >= 10 else 0
                            
                            # Log to tensorboard
                            if self.writer:
                                self.writer.add_scalar("system/gpu_mem_allocated_gb", mem_allocated, self.global_step)
                                self.writer.add_scalar("system/gpu_mem_cached_gb", mem_cached, self.global_step)
                                self.writer.add_scalar("system/gpu_mem_fluctuation", max_mem - min_mem, self.global_step)
                    
                    # Skip empty batches
                    if batch.get('empty_batch', False):
                        progress_bar.update(1)
                        continue
                    
                    try:
                        # Record batch start time for throughput calculation
                        batch_start_time = time.time()
                        
                        # Move batch to device
                        batch = self.move_batch_to_device(batch)
                        
                        # Prepare labels
                        label = batch.get('label')
                        era = batch.get('era')
                        
                        # Skip if missing required data
                        if label is None:
                            progress_bar.write(f"Skipping batch {batch_idx}: missing label")
                            progress_bar.update(1)
                            continue
                        
                        # Update batch size for throughput calculation
                        current_batch_size = label.shape[0]
                        samples_processed += current_batch_size
                            
                        # Clear previous gradients - use set_to_none for improved memory efficiency
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        try:
                            # Forward pass with optimized memory handling
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    # Process with memory-optimized forward pass
                                    outputs = self.model(batch, global_step=self.global_step)
                                    loss_dict = self.criterion(
                                        outputs,
                                        {'days': label, 'era': era},
                                        self.global_step,
                                        total_steps
                                    )
                                    loss = loss_dict['loss']
                            else:
                                # Enable gradient checkpointing for larger models if available
                                # This trades computation for memory
                                if hasattr(self.model, 'gradient_checkpointing_enable') and current_batch_size > 16:
                                    self.model.gradient_checkpointing_enable()
                                    
                                outputs = self.model(batch, global_step=self.global_step)
                                loss_dict = self.criterion(
                                    outputs,
                                    {'days': label, 'era': era},
                                    self.global_step,
                                    total_steps
                                )
                                loss = loss_dict['loss']
                            
                            # Skip if NaN/Inf loss
                            if torch.isnan(loss) or torch.isinf(loss):
                                progress_bar.write(f"Skipping batch {batch_idx}: NaN/Inf loss")
                                progress_bar.update(1)
                                # Clear unnecessary tensors
                                del outputs, loss_dict, loss
                                continue
                            
                            # Backward pass with optimized memory handling
                            if self.use_amp:
                                # Scale the loss to prevent underflow
                                self.scaler.scale(loss).backward()
                                
                                # Unscale weights for gradient clipping
                                self.scaler.unscale_(self.optimizer)
                                
                                # Clip gradients to prevent training instability
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_value)
                                
                                # Step with scaler to maintain mixed precision benefits
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                # Regular backward pass
                                loss.backward()
                                
                                # Clip gradients for stability 
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_value)
                                
                                # Apply gradient updates
                                self.optimizer.step()
                            
                            # Update learning rate
                            self.scheduler.step()
                            
                            # Store the loss value for logging
                            loss_value = loss.item()
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                # Handle OOM error by freeing memory and reducing batch size
                                progress_bar.write(f"WARNING: Out of memory error in batch {batch_idx}. Clearing cache and reducing batch size.")
                                if self.device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                elif self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                                    torch.mps.empty_cache()
                                    
                                # Reduce batch size for future batches
                                self.dynamic_batch_size = max(4, int(self.dynamic_batch_size * 0.8))
                                progress_bar.write(f"Reduced batch size to {self.dynamic_batch_size} for future epochs")
                                
                                # Log recommendation
                                with open(os.path.join(self.output_dir, "resource_recommendations.txt"), 'a') as f:
                                    f.write(f"Step {self.global_step}: OOM error. Recommend batch_size={self.dynamic_batch_size} or lower\n")
                                
                                # Skip this batch
                                progress_bar.update(1)
                                continue
                            else:
                                # Re-raise other errors
                                raise
                        
                        # Calculate batch processing time
                        batch_time = time.time() - batch_start_time
                        step_times.append(batch_time)
                        
                        # Compute throughput metrics and check resource utilization
                        if time.time() - last_time_check >= 10:  # Every 10 seconds
                            elapsed = time.time() - last_time_check
                            steps_done = len(step_times)
                            if steps_done > 0 and elapsed > 0:
                                samples_per_sec = samples_processed / elapsed
                                avg_step_time = np.mean(step_times)
                                
                                # Capture GPU utilization if available
                                gpu_util = 0
                                memory_util = 0
                                
                                if self.device.type == 'cuda':
                                    if hasattr(self, 'nvml_initialized') and self.nvml_initialized:
                                        try:
                                            import pynvml
                                            # Get GPU utilization
                                            util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                                            gpu_util = util.gpu  # 0-100%
                                            memory_util = util.memory  # 0-100%
                                            
                                            # Get memory info
                                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                                            mem_used = mem_info.used / mem_info.total * 100.0
                                            
                                            # Store metrics for auto-tuning
                                            self.gpu_util_samples.append(gpu_util)
                                            self.memory_util_samples.append(mem_used)
                                            self.batch_time_samples.append(avg_step_time)
                                            
                                            # Log GPU metrics to TensorBoard
                                            if self.writer:
                                                self.writer.add_scalar("gpu/utilization_pct", gpu_util, self.global_step)
                                                self.writer.add_scalar("gpu/memory_used_pct", mem_used, self.global_step)
                                                
                                        except Exception as e:
                                            if self.config.debug:
                                                print(f"Error monitoring GPU: {e}")
                                    else:
                                        # Fallback to PyTorch's memory stats
                                        mem_allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(self.device).total_memory * 100
                                        self.memory_util_samples.append(mem_allocated)
                                        if self.writer:
                                            self.writer.add_scalar("gpu/memory_used_pct", mem_allocated, self.global_step)
                                
                                elif self.device.type == 'mps' and hasattr(self, 'psutil_available') and self.psutil_available:
                                    try:
                                        import psutil
                                        # For MPS, we use system memory as proxy for GPU memory
                                        memory = psutil.virtual_memory()
                                        mem_used_pct = memory.percent
                                        self.memory_util_samples.append(mem_used_pct)
                                        if self.writer:
                                            self.writer.add_scalar("system/memory_used_pct", mem_used_pct, self.global_step)
                                    except Exception as e:
                                        if self.config.debug:
                                            print(f"Error monitoring system memory: {e}")
                                
                                # Enhanced auto-tuning for resource usage optimization
                                if self.auto_tune and self.global_step > 50 and time.time() - self.last_resource_check >= self.resource_check_interval:
                                    self.last_resource_check = time.time()
                                    
                                    if self.device.type == 'cuda':
                                        # More aggressive CUDA resource optimization
                                        if len(self.gpu_util_samples) >= 2:
                                            # Get average GPU utilization with more weight on recent samples
                                            recent_gpu_util = np.mean(self.gpu_util_samples[-2:])
                                            recent_mem_util = np.mean(self.memory_util_samples[-2:])
                                            
                                            # Get overall trend
                                            gpu_util_trend = 0
                                            mem_util_trend = 0
                                            if len(self.gpu_util_samples) >= 4:
                                                gpu_util_trend = np.mean(self.gpu_util_samples[-2:]) - np.mean(self.gpu_util_samples[-4:-2])
                                                mem_util_trend = np.mean(self.memory_util_samples[-2:]) - np.mean(self.memory_util_samples[-4:-2])
                                            
                                            # Log detailed resource information
                                            progress_bar.write(f"Resource check: GPU: {recent_gpu_util:.1f}% (trend: {gpu_util_trend:+.1f}%), " +
                                                             f"Memory: {recent_mem_util:.1f}% (trend: {mem_util_trend:+.1f}%)")
                                            
                                            # Target 90-95% memory utilization instead of previous 80%
                                            target_min_mem = 85
                                            target_max_mem = 95
                                            
                                            # Auto-tune based on GPU utilization and memory trend
                                            if recent_gpu_util < 70 and recent_mem_util < target_min_mem and mem_util_trend <= 0:
                                                # GPU is underutilized, memory is available, and memory usage is stable/decreasing
                                                if len(self.train_loader.dataset) > 500:  # Only adjust if dataset is large enough
                                                    # More aggressive batch size increase (up to 50%)
                                                    increase_factor = 1.2 if recent_mem_util < 75 else 1.1
                                                    new_batch_size = min(int(self.train_loader.batch_size * increase_factor), 512)
                                                    
                                                    # Only change if increase is significant
                                                    if new_batch_size > self.train_loader.batch_size + 4:
                                                        progress_bar.write(f"Auto-tuning: Increasing batch size from {self.train_loader.batch_size} to {new_batch_size}")
                                                        
                                                        # Update dynamic batch size
                                                        self.dynamic_batch_size = new_batch_size
                                                        
                                                        # Also adjust workers based on new batch size
                                                        cpu_count = os.cpu_count() or 4
                                                        new_workers = min(cpu_count - 1, max(2, new_batch_size // 8 + 1))
                                                        
                                                        # Store recommendations
                                                        with open(os.path.join(self.output_dir, "resource_recommendations.txt"), 'a') as f:
                                                            f.write(f"Step {self.global_step}: Recommend batch_size={new_batch_size}, workers={new_workers} " +
                                                                  f"- GPU util: {recent_gpu_util:.1f}%, Mem util: {recent_mem_util:.1f}%\n")
                                            
                                            # Memory-first scaling policy - prioritize high memory utilization
                                            elif recent_mem_util < 80 and mem_util_trend < 1.0:
                                                # Memory utilization is low and stable - can try larger batches
                                                new_batch_size = min(int(self.train_loader.batch_size * 1.1), 512)
                                                if new_batch_size > self.train_loader.batch_size:
                                                    progress_bar.write(f"Auto-tuning: Slightly increasing batch size from {self.train_loader.batch_size} to {new_batch_size}")
                                                    self.dynamic_batch_size = new_batch_size
                                            
                                            # Risk mitigation for high memory usage or increasing trend
                                            elif recent_mem_util > target_max_mem or (recent_mem_util > 85 and mem_util_trend > 2.0):
                                                # Memory utilization is very high or increasing rapidly - reduce batch size
                                                # More aggressive reduction (25% instead of 20%)
                                                reduce_factor = 0.75 if recent_mem_util > 98 else 0.85
                                                new_batch_size = max(int(self.train_loader.batch_size * reduce_factor), 8)
                                                
                                                # Only reduce if significant
                                                if new_batch_size < self.train_loader.batch_size - 2:
                                                    progress_bar.write(f"Auto-tuning: Decreasing batch size from {self.train_loader.batch_size} to {new_batch_size} " +
                                                                    f"to prevent OOM (Memory: {recent_mem_util:.1f}%)")
                                                    self.dynamic_batch_size = new_batch_size
                                                    
                                                    # Also update prefetch factor to be more conservative
                                                    prefetch_factor = max(1, int(self.config.prefetch_factor * 0.75))
                                                    
                                                    # Force cache clearing
                                                    gc.collect()
                                                    torch.cuda.empty_cache()
                                                    
                                                    # Store recommendation for next run
                                                    with open(os.path.join(self.output_dir, "resource_recommendations.txt"), 'a') as f:
                                                        f.write(f"Step {self.global_step}: WARNING - High memory usage. " +
                                                              f"Recommend batch_size={new_batch_size}, prefetch_factor={prefetch_factor}\n")
                                    
                                    elif self.device.type == 'mps':
                                        # MPS-specific optimizations
                                        if len(self.memory_util_samples) >= 2:
                                            # Get recent memory usage (system memory as proxy)
                                            recent_mem_util = np.mean(self.memory_util_samples[-2:])
                                            
                                            # Check trend
                                            mem_util_trend = 0
                                            if len(self.memory_util_samples) >= 4:
                                                mem_util_trend = np.mean(self.memory_util_samples[-2:]) - np.mean(self.memory_util_samples[-4:-2])
                                                
                                            progress_bar.write(f"Resource check: System memory: {recent_mem_util:.1f}% (trend: {mem_util_trend:+.1f}%)")
                                            
                                            # MPS tuning - be more aggressive with batch size on Apple Silicon
                                            if recent_mem_util < 70 and mem_util_trend <= 0:
                                                # System has plenty of memory and stable/decreasing usage
                                                new_batch_size = min(int(self.train_loader.batch_size * 1.25), 64)
                                                if new_batch_size > self.train_loader.batch_size:
                                                    progress_bar.write(f"Auto-tuning: Increasing MPS batch size from {self.train_loader.batch_size} to {new_batch_size}")
                                                    self.dynamic_batch_size = new_batch_size
                                                    
                                            elif recent_mem_util > 90 or (recent_mem_util > 80 and mem_util_trend > 3.0):
                                                # System memory is high or increasing rapidly
                                                new_batch_size = max(int(self.train_loader.batch_size * 0.7), 4)
                                                progress_bar.write(f"Auto-tuning: Decreasing MPS batch size from {self.train_loader.batch_size} to {new_batch_size}")
                                                self.dynamic_batch_size = new_batch_size
                                                
                                                # Force memory cleanup
                                                gc.collect()
                                                if hasattr(torch.mps, 'empty_cache'):
                                                    torch.mps.empty_cache()
                                    
                                    # Keep more samples for better trend analysis, but limit to avoid memory issues
                                    self.gpu_util_samples = self.gpu_util_samples[-10:] if self.gpu_util_samples else []
                                    self.memory_util_samples = self.memory_util_samples[-10:] if self.memory_util_samples else []
                                    
                                    # Log the current dynamic batch size to TensorBoard
                                    if self.writer:
                                        self.writer.add_scalar("system/dynamic_batch_size", self.dynamic_batch_size, self.global_step)
                                
                                # Update progress bar description with throughput and resource info
                                postfix_dict = {
                                    'loss': f"{loss_value:.4f}",
                                    'samples/sec': f"{samples_per_sec:.1f}",
                                    'step_time': f"{avg_step_time:.3f}s",
                                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                                }
                                
                                # Add GPU info if available
                                if gpu_util > 0:
                                    postfix_dict['GPU%'] = f"{gpu_util:.0f}"
                                if memory_util > 0:
                                    postfix_dict['Mem%'] = f"{memory_util:.0f}"
                                    
                                progress_bar.set_postfix(postfix_dict)
                                
                                # Log throughput to TensorBoard
                                if self.writer:
                                    self.writer.add_scalar("perf/samples_per_sec", samples_per_sec, self.global_step)
                                    self.writer.add_scalar("perf/step_time_sec", avg_step_time, self.global_step)
                                    # Log the dynamic batch size
                                    self.writer.add_scalar("perf/dynamic_batch_size", self.dynamic_batch_size, self.global_step)
                                
                                # Reset counters
                                last_time_check = time.time()
                                samples_processed = 0
                                step_times = []
                        
                        # Log progress
                        epoch_losses.append(loss_value)
                        
                        # Log to tensorboard
                        if self.writer and self.global_step % 10 == 0:
                            # Log basic metrics
                            self.writer.add_scalar("train/loss", loss_value, self.global_step)
                            self.writer.add_scalar("train/learning_rate", 
                                                 self.optimizer.param_groups[0]['lr'], 
                                                 self.global_step)
                            
                            # Log individual loss components
                            for k, v in loss_dict.items():
                                if k != 'loss':
                                    self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)
                            
                            # Log model predictions vs ground truth (every 20 steps to avoid too much data)
                            if self.global_step % 20 == 0:
                                # Sample a few random items from the batch for visualization
                                num_samples = min(5, label.shape[0])
                                indices = torch.randperm(label.shape[0])[:num_samples]
                                
                                # Days prediction
                                if 'days' in outputs:
                                    pred_days = outputs['days']
                                    # Unscale predictions if they are in [0,1] range
                                    if torch.max(pred_days) <= 1.0 and 'days_unscaled' in outputs:
                                        pred_days = outputs['days_unscaled']
                                    
                                    # Calculate error
                                    true_days = label.view(-1) * 10000.0  # Assuming scaling factor same as in model
                                    for i, idx in enumerate(indices):
                                        self.writer.add_scalar(f"predictions/days_sample_{i}", 
                                                            pred_days[idx].item(), 
                                                            self.global_step)
                                        self.writer.add_scalar(f"ground_truth/days_sample_{i}", 
                                                            true_days[idx].item(), 
                                                            self.global_step)
                                
                                # Year prediction (if available)
                                if 'year_logits' in outputs and outputs['year_logits'] is not None:
                                    year_logits = outputs['year_logits']
                                    if year_logits.shape[0] > 0:
                                        # Convert logits to predictions
                                        year_preds = torch.argmax(year_logits, dim=1) + 1965  # Adjust for year offset
                                        
                                        # Calculate true years from label if needed
                                        for i, idx in enumerate(indices):
                                            if idx < year_preds.shape[0]:
                                                self.writer.add_scalar(f"predictions/year_sample_{i}", 
                                                                    year_preds[idx].item(), 
                                                                    self.global_step)
                                
                                # Era prediction (if available)
                                if 'era_logits' in outputs and outputs['era_logits'] is not None and era is not None:
                                    era_logits = outputs['era_logits']
                                    if era_logits.shape[0] > 0:
                                        # Convert logits to predictions
                                        era_preds = torch.argmax(era_logits, dim=1)
                                        
                                        # Log era ground truth and predictions
                                        for i, idx in enumerate(indices):
                                            if idx < era_preds.shape[0] and idx < era.shape[0]:
                                                self.writer.add_scalar(f"predictions/era_sample_{i}", 
                                                                    era_preds[idx].item(), 
                                                                    self.global_step)
                                                self.writer.add_scalar(f"ground_truth/era_sample_{i}", 
                                                                    era[idx].item(), 
                                                                    self.global_step)
                                
                                # Uncertainty (log_variance) prediction
                                if 'log_variance' in outputs:
                                    log_var = outputs['log_variance']
                                    if log_var.shape[0] > 0:
                                        # Convert to standard deviation for easier interpretation
                                        std_dev = torch.exp(0.5 * log_var)
                                        for i, idx in enumerate(indices):
                                            if idx < log_var.shape[0]:
                                                self.writer.add_scalar(f"predictions/uncertainty_std_sample_{i}", 
                                                                    std_dev[idx].item(), 
                                                                    self.global_step)
                        
                        # Optimized memory cleanup strategy to prevent memory spikes
                        # Store loss value before clearing tensors
                        loss_value = loss.item() if 'loss' in locals() and loss is not None else 0.0
                        
                        # Clear variables from highest to lowest memory consumption
                        if 'outputs' in locals() and outputs is not None:
                            # Clear individual tensors inside outputs dictionary for finer control
                            if isinstance(outputs, dict):
                                for key in list(outputs.keys()):
                                    if isinstance(outputs[key], torch.Tensor):
                                        outputs[key] = None
                            del outputs
                            
                        if 'loss_dict' in locals() and loss_dict is not None:
                            for key in list(loss_dict.keys()):
                                if isinstance(loss_dict[key], torch.Tensor):
                                    loss_dict[key] = None
                            del loss_dict
                            
                        if 'loss' in locals() and loss is not None:
                            del loss
                        
                        # Device-specific memory management strategies
                        if self.device.type == 'mps':
                            # Aggressive memory management for MPS
                            
                            # Clear batch tensors individually to prevent fragmentation
                            if 'batch' in locals() and batch is not None:
                                for k in list(batch.keys()):
                                    if isinstance(batch[k], torch.Tensor):
                                        batch[k] = None
                                del batch
                                
                            # Clear specific tensors
                            if 'label' in locals() and label is not None:
                                del label
                            if 'era' in locals() and era is not None:
                                del era
                                
                            # More frequent garbage collection for MPS - every 2 steps
                            if self.global_step % 2 == 0:
                                # Force garbage collection
                                gc.collect()
                                
                                # Clear MPS cache if available
                                if hasattr(torch.mps, 'empty_cache'):
                                    torch.mps.empty_cache()
                                    
                        elif self.device.type == 'cuda':
                            # More strategic memory management for CUDA
                            
                            # Clear batch tensors to free GPU memory
                            if 'batch' in locals() and batch is not None:
                                for k in list(batch.keys()):
                                    if isinstance(batch[k], torch.Tensor):
                                        batch[k] = None
                                del batch
                                
                            # Release specific tensors
                            if 'label' in locals() and label is not None:
                                del label
                            if 'era' in locals() and era is not None:
                                del era
                                
                            # Clear CUDA cache periodically - less frequently than MPS
                            # Use dynamic frequency based on memory utilization
                            if self.global_step % 5 == 0:
                                # Check current memory utilization
                                mem_allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(self.device).total_memory
                                
                                # If memory usage is high (>75%), clean more aggressively
                                if mem_allocated > 0.75:
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                # Otherwise, do lighter cleaning less frequently
                                elif self.global_step % 20 == 0:
                                    gc.collect()
                        
                        # Increment global step
                        self.global_step += 1
                        progress_bar.update(1)
                        
                        # Removed per-step checkpointing (too frequent)
                        
                        # Run validation periodically
                        if self.global_step % 100 == 0:
                            progress_bar.write(f"Running validation at step {self.global_step}...")
                            val_loss = self.validate()
                            
                            # Check for improvement
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.patience_counter = 0
                                self.save_checkpoint(
                                    os.path.join(self.output_dir, "checkpoints", "best_checkpoint.pt"),
                                    is_best=True
                                )
                                progress_bar.write(f"New best validation loss: {val_loss:.4f}")
                            else:
                                self.patience_counter += 1
                                progress_bar.write(f"Validation loss: {val_loss:.4f} (patience: {self.patience_counter}/{self.config.patience})")
                                
                                # Early stopping
                                if (self.config.use_early_stopping and 
                                    self.patience_counter >= self.config.patience):
                                    progress_bar.write(f"Early stopping triggered after {self.patience_counter} validations without improvement")
                                    break
                        
                        # Check if we've reached max_steps
                        if self.global_step >= total_steps:
                            break
                        
                        # Calculate ETA
                        if len(step_times) > 5:  # Need enough data for stable estimate
                            if steps_are_primary:
                                steps_remaining = total_steps - self.global_step
                                target = f"step {self.global_step}/{total_steps}"
                            else:
                                steps_per_epoch = len(self.train_loader)
                                epochs_remaining = max_epochs - epoch
                                steps_remaining = (steps_per_epoch * epochs_remaining) - batch_idx
                                target = f"epoch {epoch}/{max_epochs}"
                                
                            avg_time_per_step = np.mean(step_times)
                            eta_seconds = steps_remaining * avg_time_per_step
                            # Fix: Use correct datetime import
                            from datetime import timedelta
                            eta_str = str(timedelta(seconds=int(eta_seconds)))
                            
                            # Update progress bar with ETA and target info
                            progress_bar.set_postfix({
                                'loss': f"{loss_value:.4f}",
                                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                                'target': target,
                                'ETA': eta_str
                            })
                            
                    except Exception as e:
                        progress_bar.write(f"Error in training step: {e}")
                        # Print full traceback regardless of debug setting
                        import traceback
                        traceback.print_exc()
                        progress_bar.update(1)
                        continue
                
                # Close progress bar for this epoch
                progress_bar.close()
                
                # End of epoch
                avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                print(f"Epoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")
                
                # Save checkpoint at the end of each epoch
                print(f"Saving checkpoint after epoch {epoch}...")
                self.save_checkpoint(
                    os.path.join(self.output_dir, "checkpoints", "latest_checkpoint.pt")
                )
                
                # Run validation at the end of each epoch
                print(f"End-of-epoch validation...")
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(
                        os.path.join(self.output_dir, "checkpoints", "best_checkpoint.pt"),
                        is_best=True
                    )
                    print(f"New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"Validation loss: {val_loss:.4f} (patience: {self.patience_counter}/{self.config.patience})")
                    
                    # Early stopping check
                    if (self.config.use_early_stopping and 
                        self.patience_counter >= self.config.patience):
                        print(f"Early stopping triggered after {self.patience_counter} validations without improvement")
                        break
                
                # Print memory statistics if available
                if self.device.type == 'cuda':
                    print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, "
                          f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved")
                elif self.device.type == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
                    print(f"MPS Memory: {torch.mps.current_allocated_memory() / 1024**3:.2f}GB allocated")
            
            # Training complete
            elapsed_time = time.time() - start_time
            print(f"\n=== Training Summary ===")
            print(f"Completed {self.global_step} steps in {epoch} epochs")
            print(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            
            # Calculate and print performance metrics
            if step_times:
                print(f"Average step time: {np.mean(step_times):.4f} seconds")
                print(f"Throughput: {1.0 / np.mean(step_times):.2f} steps/second")
            
            # Save final checkpoint
            self.save_checkpoint(
                os.path.join(self.output_dir, "checkpoints", "final_checkpoint.pt")
            )
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self.save_checkpoint(
                os.path.join(self.output_dir, "checkpoints", "interrupted_checkpoint.pt")
            )
        
        finally:
            # Clean up
            if self.writer:
                self.writer.close()
    
    def validate(self):
        """Run validation and return the average validation loss."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # Skip empty batches
                    if batch.get('empty_batch', False):
                        continue
                    
                    # Move batch to device
                    batch = self.move_batch_to_device(batch)
                    
                    # Prepare labels
                    label = batch.get('label')
                    era = batch.get('era')
                    
                    # Skip if missing required data
                    if label is None:
                        continue
                    
                    # Forward pass
                    outputs = self.model(batch)
                    loss_dict = self.criterion(
                        outputs,
                        {'days': label, 'era': era},
                        self.global_step,
                        self.config.total_training_steps
                    )
                    loss = loss_dict['loss']
                    
                    # Store loss
                    val_losses.append(loss.item())
                    
                    # Clear memory for MPS
                    if self.device.type == 'mps':
                        del outputs, loss, loss_dict
                        gc.collect()
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
        
        # Calculate average validation loss
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar("val/loss", avg_val_loss, self.global_step)
            
            # Log validation predictions and ground truth
            # Only do this periodically to avoid cluttering TensorBoard
            if self.global_step % 100 == 0:
                # Collect validation predictions for TensorBoard visualization
                with torch.no_grad():
                    try:
                        # Sample batch from validation set
                        val_iter = iter(self.val_loader)
                        val_batch = next(val_iter)
                        
                        # Skip empty batches
                        if not val_batch.get('empty_batch', False):
                            # Move to device
                            val_batch = self.move_batch_to_device(val_batch)
                            
                            # Get predictions
                            val_outputs = self.model(val_batch)
                            
                            # Get labels
                            val_label = val_batch.get('label')
                            val_era = val_batch.get('era')
                            
                            if val_label is not None:
                                # Sample a few random items for visualization
                                num_samples = min(3, val_label.shape[0])
                                indices = torch.randperm(val_label.shape[0])[:num_samples]
                                
                                # Days prediction
                                if 'days' in val_outputs:
                                    pred_days = val_outputs['days']
                                    # Unscale predictions if they are in [0,1] range
                                    if torch.max(pred_days) <= 1.0 and 'days_unscaled' in val_outputs:
                                        pred_days = val_outputs['days_unscaled']
                                    
                                    # Log predictions vs ground truth
                                    true_days = val_label.view(-1) * 10000.0
                                    for i, idx in enumerate(indices):
                                        self.writer.add_scalar(f"val_predictions/days_sample_{i}", 
                                                            pred_days[idx].item(), 
                                                            self.global_step)
                                        self.writer.add_scalar(f"val_ground_truth/days_sample_{i}", 
                                                            true_days[idx].item(), 
                                                            self.global_step)
                                        
                                        # Calculate and log error
                                        day_error = abs(pred_days[idx].item() - true_days[idx].item())
                                        self.writer.add_scalar(f"val_metrics/days_error_sample_{i}", 
                                                            day_error, 
                                                            self.global_step)
                                
                                # Year and era predictions
                                if 'year_logits' in val_outputs and val_outputs['year_logits'] is not None:
                                    year_logits = val_outputs['year_logits']
                                    year_preds = torch.argmax(year_logits, dim=1) + 1965
                                    
                                    for i, idx in enumerate(indices):
                                        if idx < year_preds.shape[0]:
                                            self.writer.add_scalar(f"val_predictions/year_sample_{i}", 
                                                                year_preds[idx].item(), 
                                                                self.global_step)
                                
                                # Uncertainty
                                if 'log_variance' in val_outputs:
                                    log_var = val_outputs['log_variance']
                                    # Convert to standard deviation
                                    std_dev = torch.exp(0.5 * log_var)
                                    
                                    for i, idx in enumerate(indices):
                                        if idx < std_dev.shape[0]:
                                            self.writer.add_scalar(f"val_predictions/uncertainty_std_sample_{i}", 
                                                                std_dev[idx].item(), 
                                                                self.global_step)
                                            
                                # Add histograms for model weights periodically
                                if self.global_step % 500 == 0:
                                    # Add histograms for model weights
                                    for name, param in self.model.named_parameters():
                                        if param.requires_grad:
                                            self.writer.add_histogram(f"weights/{name}", param.data, self.global_step)
                                            if param.grad is not None:
                                                self.writer.add_histogram(f"gradients/{name}", param.grad, self.global_step)
                    
                    except (StopIteration, Exception) as e:
                        # Don't fail validation if visualization has issues
                        print(f"Warning: Could not log validation visualizations: {e}")
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        return avg_val_loss


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Unified training script for Grateful Dead show dating model")
    
    # Data and model parameters
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for model outputs and checkpoints")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate")
    
    # Print all command line arguments for debugging
    print("Command line arguments will be displayed after parsing")
    
    # Training control
    parser.add_argument("--steps", type=int, default=10000, help="Total number of training steps")
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs (optional)")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (validations without improvement)")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--validation-interval", type=int, default=100, help="Run validation every N steps")
    
    # Hardware and performance
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda, mps, cpu, auto)")
    parser.add_argument("--num-workers", type=int, help="Number of dataloader workers")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--aggressive-memory", action="store_true", help="Enable aggressive memory management for MPS")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to use (for testing)")
    
    # Logging
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tensorboard-dir", type=str, default="./runs", help="TensorBoard log directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    
    args = parser.parse_args()
    
    # Print all command line arguments after parsing
    print("Command line arguments:")
    for arg in vars(args):
        print(f"  --{arg}: {getattr(args, arg)}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Print system information
    device = print_system_info()
    
    # Override device if specified
    if args.device != "auto":
        device = torch.device(args.device)
        print(f"Using specified device: {device}")
    
    # Set up config
    config = Config()
    config.total_training_steps = args.steps
    config.batch_size = args.batch_size
    config.initial_lr = args.learning_rate
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.use_early_stopping = args.early_stopping
    config.patience = args.patience
    config.use_mixed_precision = args.fp16
    config.aggressive_memory = args.aggressive_memory
    config.checkpoint_interval = args.checkpoint_interval
    config.validation_interval = args.validation_interval
    
    # Set num_workers if specified
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    
    # Ensure all required config attributes exist (add defaults if needed)
    if not hasattr(config, 'weight_decay'):
        config.weight_decay = 1e-6
    if not hasattr(config, 'use_augmentation'):
        config.use_augmentation = False
    if not hasattr(config, 'grad_clip_value'):
        config.grad_clip_value = 1.0
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, device, max_samples=args.max_samples)
    
    # Set up tensorboard
    tensorboard_dir = args.tensorboard_dir if args.tensorboard else None
    
    # Create and configure trainer with auto-tuning enabled by default
    trainer = OptimizedTrainer(
        config, 
        train_loader, 
        val_loader, 
        device, 
        tensorboard_dir,
        auto_tune=True  # Enable auto-tuning of resources
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Run training
    trainer.train(max_steps=args.steps, max_epochs=args.epochs)


if __name__ == "__main__":
    main()