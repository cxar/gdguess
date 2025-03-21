"""
Advanced optimization utilities for PyTorch models.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List

# Import helpers if they exist
try:
    import torch._dynamo
    HAS_TORCH_COMPILE = True
except ImportError:
    HAS_TORCH_COMPILE = False


def optimize_model_for_inference(
    model: nn.Module, 
    device: torch.device,
    inference_batch_size: Optional[int] = None,
    optimize_memory: bool = True,
    quantize: bool = False
) -> nn.Module:
    """
    Optimize a model for inference using device-specific optimizations.
    
    Args:
        model: The PyTorch model to optimize
        device: The target device
        inference_batch_size: Expected batch size for inference (for optimizations)
        optimize_memory: Whether to optimize for memory usage
        quantize: Whether to apply int8 quantization for CUDA devices
        
    Returns:
        Optimized model for inference
    """
    # Move model to device first
    model = model.to(device)
    model.eval()
    
    # Apply device-specific optimizations
    if device.type == 'cuda':
        # CUDA-specific optimizations
        if torch.cuda.is_available():
            # Enable TF32 for compute (faster with minimal precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN benchmark mode for optimized kernel selection
            torch.backends.cudnn.benchmark = True
            
            # Apply quantization for int8 inference if requested
            if quantize:
                try:
                    # Dynamic quantization (works for most models)
                    from torch.quantization import quantize_dynamic
                    model = quantize_dynamic(
                        model, 
                        {nn.Linear, nn.Conv1d, nn.Conv2d}, 
                        dtype=torch.qint8
                    )
                    print("Applied dynamic quantization for faster inference")
                except Exception as e:
                    print(f"Quantization failed: {e}")
            
            # Try to apply TorchScript for faster execution
            try:
                # Use scripting instead of tracing for better coverage
                model = torch.jit.script(model)
                print("Applied TorchScript optimization")
            except Exception as e:
                print(f"TorchScript optimization failed: {e}")
                # Try partial scripting if full scripting fails
                try:
                    # Identify and script compatible submodules
                    for name, submodule in model.named_children():
                        if any(isinstance(submodule, t) for t in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.RNN, nn.LSTM]):
                            try:
                                scripted_submodule = torch.jit.script(submodule)
                                setattr(model, name, scripted_submodule)
                            except Exception:
                                pass
                    print("Applied partial TorchScript optimization to compatible submodules")
                except Exception:
                    print("Partial TorchScript optimization failed")
                    
            # Compile with torch.compile if available (PyTorch 2.0+)
            if HAS_TORCH_COMPILE:
                try:
                    compile_mode = "reduce-overhead"  # Options: default, reduce-overhead, max-autotune
                    model = torch._dynamo.optimize(compile_mode)(model)
                    print(f"Applied torch.compile optimization with mode: {compile_mode}")
                except Exception as e:
                    print(f"torch.compile optimization failed: {e}")
    
    elif device.type == 'mps':
        # MPS-specific optimizations for Apple Silicon
        if optimize_memory:
            # MPS memory management optimizations
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        # MPS doesn't support TorchScript well yet
        # Just ensure float32 precision for stability
        try:
            for param in model.parameters():
                if param.data.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)
            print("Ensured float32 precision for MPS stability")
        except Exception as e:
            print(f"MPS precision optimization failed: {e}")
            
    elif device.type == 'cpu':
        # CPU-specific optimizations
        try:
            # Try to use TorchScript for CPU
            model = torch.jit.script(model)
            print("Applied TorchScript optimization for CPU")
        except Exception:
            print("TorchScript optimization failed for CPU")
            
        if quantize:
            try:
                # Apply dynamic quantization for CPU
                from torch.quantization import quantize_dynamic
                model = quantize_dynamic(
                    model, 
                    {nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU}, 
                    dtype=torch.qint8
                )
                print("Applied dynamic quantization for CPU")
            except Exception as e:
                print(f"CPU quantization failed: {e}")
                
        # Enable parallel CPU execution if multiple cores available
        if hasattr(torch, 'set_num_threads'):
            num_threads = os.cpu_count() or 2
            torch.set_num_threads(num_threads)
            print(f"Set PyTorch to use {num_threads} CPU threads")
            
    # Set to eval mode again to ensure all optimizations are applied
    model.eval()
    return model


def optimize_training_pipeline(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixed_precision: bool = True,
    channels_last: bool = True,
    compile_model: bool = False
) -> Tuple[nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    """
    Optimize the entire training pipeline for maximum performance.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        device: Target device
        mixed_precision: Whether to use mixed precision training
        channels_last: Whether to use channels-last memory format for 4D tensors
        compile_model: Whether to use torch.compile for PyTorch 2.0+
    
    Returns:
        Tuple of (optimized_model, optimized_optimizer, training_tools)
    """
    # Initialize training tools
    training_tools = {
        "scaler": None,
        "use_amp": False,
    }
    
    # Device-specific pipeline optimizations
    if device.type == 'cuda':
        # Check for TF32 support (faster matmul with negligible precision loss)
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            print("Enabled TF32 matmul for faster computation")
            
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 for cuDNN operations")
        
        # Enable mixed precision (AMP) for faster training
        if mixed_precision:
            training_tools["use_amp"] = True
            training_tools["scaler"] = torch.cuda.amp.GradScaler()
            print("Enabled automatic mixed precision training")
            
        # Use channels-last memory format for 4D tensors (potentially faster for convolutions)
        if channels_last:
            try:
                # Only apply to model with 4D tensors (like CNNs)
                has_4d = False
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                        has_4d = True
                        break
                
                if has_4d:
                    model = model.to(memory_format=torch.channels_last)
                    print("Using channels-last memory format for potential convolution speedup")
            except Exception as e:
                print(f"Warning: Failed to convert to channels_last: {e}")
                
        # Optimize CUDA graphs for repeated operations
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("Enabled cuDNN benchmark mode for optimized convolution algorithms")
            
        # Enhanced GPU-specific optimizer settings
        if isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW):
            # Try to use fused implementation for Adam/AdamW
            try:
                if hasattr(torch.optim, 'FusedAdam'):
                    new_optimizer = torch.optim.FusedAdam(
                        model.parameters(),
                        lr=optimizer.param_groups[0]['lr'],
                        betas=optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                        eps=optimizer.param_groups[0].get('eps', 1e-8),
                        weight_decay=optimizer.param_groups[0].get('weight_decay', 0),
                    )
                    optimizer = new_optimizer
                    print("Using FusedAdam for faster optimizer performance")
                elif hasattr(torch.optim.adam, '_FusedAdam'):
                    # Try apex fused Adam if available
                    from torch.optim.adam import _FusedAdam
                    new_optimizer = _FusedAdam(
                        model.parameters(),
                        lr=optimizer.param_groups[0]['lr'],
                        betas=optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                        eps=optimizer.param_groups[0].get('eps', 1e-8),
                        weight_decay=optimizer.param_groups[0].get('weight_decay', 0),
                    )
                    optimizer = new_optimizer
                    print("Using _FusedAdam for faster optimizer performance")
            except Exception as e:
                print(f"Fused optimizer initialization failed: {e}")
                
    elif device.type == 'mps':
        # MPS optimizations for Apple Silicon
        # Set appropriate default settings
        training_tools["use_amp"] = False  # MPS doesn't fully support AMP yet
        
        # Ensure model is in float32 for stability on MPS
        for param in model.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        
        # Apple-specific optimizations
        if hasattr(torch, 'set_float32_matmul_precision'):
            # Set higher matmul precision for stability
            torch.set_float32_matmul_precision('high')
            print("Set float32 matmul precision to 'high' for MPS stability")
    
    # Apply torch.compile() if available (PyTorch 2.0+) and requested
    if compile_model and HAS_TORCH_COMPILE:
        try:
            if device.type == 'cuda':
                # For CUDA, prioritize speed
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                print("Applied torch.compile with reduce-overhead mode for CUDA")
            elif device.type == 'cpu':
                # For CPU, prioritize overhead reduction
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                print("Applied torch.compile with reduce-overhead mode for CPU")
            else:
                # MPS compilation is usually not beneficial yet
                pass
        except Exception as e:
            print(f"torch.compile failed: {e}")
    
    return model, optimizer, training_tools


def create_optimized_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    device: torch.device,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create optimized DataLoaders for the given device.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        device: Target device
        num_workers: Number of workers (if None, will be automatically determined)
        pin_memory: Whether to use pinned memory (if None, will be automatically determined)
        persistent_workers: Whether to use persistent workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Configure optimal settings based on device
    if device.type == 'cuda':
        # For CUDA, use more workers and pinned memory
        if num_workers is None:
            num_workers = min(os.cpu_count() or 4, 8)  # Use up to 8 workers for CUDA
        if pin_memory is None:
            pin_memory = True  # Use pinned memory for faster CPU->GPU transfer
        prefetch_factor = 2  # Prefetch 2 batches per worker
    elif device.type == 'mps':
        # For MPS, use fewer workers to avoid memory issues
        if num_workers is None:
            num_workers = min(os.cpu_count() or 2, 4)  # Use up to 4 workers for MPS
        if pin_memory is None:
            pin_memory = False  # MPS doesn't benefit from pinned memory as much
        prefetch_factor = 2  # Lower prefetch to reduce memory pressure
    else:
        # For CPU, use all available cores
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 2) - 1)  # Use all but one core
        if pin_memory is None:
            pin_memory = False  # No need for pinned memory on CPU
        prefetch_factor = 2  # Standard prefetch
    
    # Configure DataLoader settings
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'persistent_workers': persistent_workers if num_workers > 0 else False,
    }
    
    # Additional optimization for CUDA
    if device.type == 'cuda':
        # Try setting CUDA stream priority for data loading
        if hasattr(torch.cuda, 'Stream'):
            try:
                # Create a higher priority stream for data transfer
                stream = torch.cuda.Stream(priority=-1)  # High priority
                dataloader_kwargs['generator'] = torch.Generator().manual_seed(42)
                # We'd use a custom collate_fn that uses this stream, but that's more complex
            except Exception:
                pass
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch for better performance
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,  # Keep all validation samples
        **dataloader_kwargs
    )
    
    print(f"Created optimized DataLoaders with {num_workers} workers, "
          f"pin_memory={pin_memory}, prefetch_factor={prefetch_factor if num_workers > 0 else 'N/A'}")
    
    return train_loader, val_loader


def apply_gradient_accumulation(
    model: nn.Module,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    accumulation_steps: int,
    current_step: int,
    clip_grad_norm: Optional[float] = None,
) -> None:
    """
    Apply gradient accumulation for efficient large batch training.
    
    Args:
        model: The model
        loss: The current loss tensor
        optimizer: The optimizer
        scaler: Optional gradient scaler for mixed precision
        accumulation_steps: Number of steps to accumulate gradients
        current_step: Current step within the accumulation cycle
        clip_grad_norm: Maximum norm for gradient clipping (or None to disable)
    """
    # Normalize loss to account for accumulation
    loss = loss / accumulation_steps
    
    # Backward pass with or without scaler
    if scaler is not None:
        scaler.scale(loss).backward()
        
        # Update weights at the end of accumulation cycle
        if (current_step + 1) % accumulation_steps == 0:
            if clip_grad_norm is not None:
                # Unscale before clipping for accurate values
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Update with scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    else:
        # Standard backward pass
        loss.backward()
        
        # Update weights at the end of accumulation cycle
        if (current_step + 1) % accumulation_steps == 0:
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad() 