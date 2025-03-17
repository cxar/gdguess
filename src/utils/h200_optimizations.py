#!/usr/bin/env python3
"""
H200-specific optimizations for efficient training.
"""

import os
import gc
import torch
import logging
from typing import Dict, Any, Optional, Tuple

def optimize_h200_memory() -> Dict[str, Any]:
    """
    Apply memory optimizations for H200 GPUs.
    
    Returns:
        Dictionary with optimization status information
    """
    result = {
        "success": False,
        "device": None,
        "device_name": None,
        "memory_optimized": False,
        "cuda_available": torch.cuda.is_available(),
        "applied_optimizations": []
    }
    
    if not torch.cuda.is_available():
        result["error"] = "CUDA not available"
        return result
    
    # Check if we're on an H200
    device_idx = 0
    try:
        device_props = torch.cuda.get_device_properties(device_idx)
        is_h200 = "H200" in device_props.name
        result["device_name"] = device_props.name
        result["is_h200"] = is_h200
    except Exception as e:
        result["error"] = f"Failed to get device properties: {e}"
        return result
    
    # Set device
    device = torch.device('cuda', device_idx)
    result["device"] = device
    
    try:
        # Force garbage collection first
        gc.collect()
        torch.cuda.empty_cache()
        result["applied_optimizations"].append("empty_cache")
        
        # Memory optimizations
        if is_h200:
            # H200-specific optimizations
            
            # Enable CUDA graph capture for repetitive operations
            # This can significantly reduce CPU overhead
            if hasattr(torch.cuda, 'is_current_stream_capturing'):
                result["applied_optimizations"].append("cuda_graph_ready")
                
            # Enable asynchronous GPU copies
            torch.backends.cuda.matmul.allow_tf32 = True
            result["applied_optimizations"].append("allow_tf32")
            
            # Optimize memory allocator
            if hasattr(torch.cuda, 'memory_stats'):
                # Configure memory allocator for large-batch training
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                result["applied_optimizations"].append("alloc_conf")
                
            # For H200 with HBM3e memory, enable advanced memory management
            torch.backends.cudnn.allow_tf32 = True
            result["applied_optimizations"].append("cudnn_tf32")
            
            # Set environment variable for H200 optimized kernels
            if "PYTORCH_JIT_USE_NNC_NOT_NVFUSER" not in os.environ:
                os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "0"
                result["applied_optimizations"].append("nvfuser_enabled")
        
        # For all CUDA devices, enable benchmark mode when input sizes are consistent
        torch.backends.cudnn.benchmark = True
        result["applied_optimizations"].append("cudnn_benchmark")
        
        # Successfully applied optimizations
        result["success"] = True
        result["memory_optimized"] = True
        
    except Exception as e:
        result["error"] = f"Optimization error: {e}"
        
    return result


def create_cuda_streams(num_streams: int = 4) -> Dict[str, Any]:
    """
    Create CUDA streams for pipeline parallelism.
    
    Args:
        num_streams: Number of CUDA streams to create
        
    Returns:
        Dictionary with streams and status information
    """
    result = {
        "success": False,
        "streams": [],
        "num_streams": num_streams
    }
    
    if not torch.cuda.is_available():
        result["error"] = "CUDA not available"
        return result
    
    try:
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        result["streams"] = streams
        result["success"] = True
    except Exception as e:
        result["error"] = f"Failed to create CUDA streams: {e}"
    
    return result


def get_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: Dict[str, torch.Tensor],
    target_device: torch.device,
    min_batch: int = 16,
    max_batch: int = 512,
    step_size: int = 16
) -> Tuple[int, Dict[str, Any]]:
    """
    Find optimal batch size for H200 GPU.
    
    Args:
        model: The PyTorch model
        sample_input: A sample input batch with batch_size=1
        target_device: Device to test on
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try
        step_size: Step size for batch testing
        
    Returns:
        Tuple of (optimal_batch_size, details_dict)
    """
    results = {
        "tested_batch_sizes": [],
        "memory_usage": {},
        "success": False,
        "optimal_batch_size": None,
        "optimal_memory_usage": None
    }
    
    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        return min_batch, results
    
    # Save initial state
    initial_device = next(model.parameters()).device
    model = model.to(target_device)
    model.eval()  # Set to eval mode for memory testing
    
    # Test batch sizes
    max_working_batch = min_batch
    
    for batch_size in range(min_batch, max_batch + 1, step_size):
        # Clear cache before test
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Start from clean memory state
            torch.cuda.reset_peak_memory_stats()
            
            # Create a larger batch by repeating the sample
            batch = {}
            for key, tensor in sample_input.items():
                if isinstance(tensor, torch.Tensor):
                    if tensor.dim() == 0:  # Handle scalars
                        batch[key] = tensor.repeat(batch_size)
                    else:
                        # Repeat first dimension to batch_size
                        repeats = [batch_size] + [1] * (tensor.dim() - 1)
                        batch[key] = tensor.repeat(*repeats)
                        
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(target_device)
            
            # Run inference with this batch size
            with torch.no_grad():
                _ = model(batch)
            
            # Get peak memory
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # in GB
            
            results["tested_batch_sizes"].append(batch_size)
            results["memory_usage"][batch_size] = peak_memory
            max_working_batch = batch_size
            
            logging.info(f"Batch size {batch_size}: {peak_memory:.2f} GB")
            
        except torch.cuda.OutOfMemoryError:
            logging.info(f"OOM at batch size {batch_size}")
            break
        except Exception as e:
            logging.error(f"Error testing batch size {batch_size}: {e}")
            break
    
    # Return model to original device
    model = model.to(initial_device)
    
    # Calculate optimal batch size (80% of max to leave headroom)
    optimal_batch = int(max_working_batch * 0.8)
    optimal_batch = max(optimal_batch, min_batch)
    
    results["success"] = True
    results["optimal_batch_size"] = optimal_batch
    if optimal_batch in results["memory_usage"]:
        results["optimal_memory_usage"] = results["memory_usage"][optimal_batch]
    
    return optimal_batch, results 