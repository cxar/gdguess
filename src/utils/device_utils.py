#!/usr/bin/env python3
"""
Utility functions for device management and diagnostics,
especially for MPS (Metal Performance Shaders) on Apple Silicon.
"""

import os
import platform
import sys
import time
import torch
import numpy as np
import psutil
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

# Try to import NVIDIA management library
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

def get_cpu_info() -> Dict:
    """Get detailed CPU information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "num_physical_cores": psutil.cpu_count(logical=False),
        "num_logical_cores": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
        "memory_available": psutil.virtual_memory().available / (1024**3)  # GB
    }
    return info

def get_gpu_info() -> Dict:
    """Get detailed GPU information if available."""
    info = {"available": False}
    
    # Check for CUDA
    if torch.cuda.is_available():
        info["available"] = True
        info["type"] = "CUDA"
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(info["current_device"])
        info["cuda_version"] = torch.version.cuda
        
        # Add NVML info if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["memory_total"] = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)  # GB
                info["memory_free"] = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024**3)  # GB
                info["utilization"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                pynvml.nvmlShutdown()
            except:
                pass
        
    # Check for MPS
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["available"] = True
        info["type"] = "MPS"
        info["device_name"] = "Apple Silicon GPU"
        info["is_built"] = torch.backends.mps.is_built()
        
        # Check if we can query MPS memory info
        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                info["current_allocated"] = torch.mps.current_allocated_memory() / (1024**3)  # GB
            if hasattr(torch.mps, "driver_allocated_memory"):
                info["driver_allocated"] = torch.mps.driver_allocated_memory() / (1024**3)  # GB
        except:
            pass
            
    return info

def print_system_info():
    """Print detailed system information."""
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    
    print("=== System Information ===")
    print(f"Platform: {cpu_info['platform']}")
    print(f"Python: {cpu_info['python_version']}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU: {cpu_info['processor']}")
    print(f"Cores: {cpu_info['num_physical_cores']} physical, {cpu_info['num_logical_cores']} logical")
    print(f"Memory: {cpu_info['memory_total']:.2f} GB total, {cpu_info['memory_available']:.2f} GB available")
    
    print("\n=== GPU Information ===")
    if gpu_info["available"]:
        print(f"GPU Type: {gpu_info['type']}")
        print(f"Device: {gpu_info['device_name']}")
        
        if gpu_info["type"] == "CUDA":
            print(f"CUDA Version: {gpu_info['cuda_version']}")
            print(f"Device Count: {gpu_info['device_count']}")
            
            if "memory_total" in gpu_info:
                print(f"Memory: {gpu_info['memory_total']:.2f} GB total, {gpu_info['memory_free']:.2f} GB free")
            if "utilization" in gpu_info:
                print(f"Utilization: {gpu_info['utilization']}%")
                
        elif gpu_info["type"] == "MPS":
            if "current_allocated" in gpu_info:
                print(f"Current Allocated Memory: {gpu_info['current_allocated']:.2f} GB")
            if "driver_allocated" in gpu_info:
                print(f"Driver Allocated Memory: {gpu_info['driver_allocated']:.2f} GB")
    else:
        print("No GPU acceleration available")
    
    print("\n=== PyTorch Device Setup ===")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device

def test_basic_operations(device=None):
    """Test basic tensor operations on the specified device."""
    if device is None:
        device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
    
    print(f"\n=== Testing Basic Operations on {device} ===")
    
    # Create tensors on CPU first
    a_cpu = torch.randn(1000, 1000)
    b_cpu = torch.randn(1000, 1000)
    
    # Time CPU operation
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"CPU matrix multiplication: {cpu_time:.4f} seconds")
    
    # Move to device and perform operation
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    
    # Warmup
    c = torch.matmul(a, b)
    
    # Timed operation
    start = time.time()
    c = torch.matmul(a, b)
    device_time = time.time() - start
    print(f"{device} matrix multiplication: {device_time:.4f} seconds")
    
    if device_time > 0:
        speedup = cpu_time / device_time
        print(f"Speedup: {speedup:.2f}x")
    
    # Check correctness
    c_cpu_from_device = c.to("cpu")
    is_close = torch.allclose(c_cpu, c_cpu_from_device, rtol=1e-3, atol=1e-3)
    print(f"Results match: {is_close}")
    
    if not is_close:
        max_diff = torch.max(torch.abs(c_cpu - c_cpu_from_device))
        print(f"Maximum difference: {max_diff}")
    
    return is_close

def benchmark_matrix_operations(sizes, device, num_runs=5):
    """Benchmark matrix operations with different sizes."""
    results = []
    
    for size in tqdm(sizes, desc=f"Matrix ops on {device}"):
        times = []
        
        for _ in range(num_runs):
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warmup
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            c = torch.matmul(a, b)
            
            # Sync before timing
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            # Timed run
            start = time.time()
            c = torch.matmul(a, b)
            
            # Sync after timing
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            times.append(time.time() - start)
            
            # Clear cache
            del a, b, c
            if device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
        
        results.append(np.mean(times))
    
    return results

def run_benchmark(max_size=4000, step=500, device=None, plot=True):
    """Run comprehensive benchmarks and optionally plot results."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    print(f"Running benchmarks on {device}...")
    sizes = list(range(step, max_size + step, step))
    
    # Also benchmark on CPU for comparison
    cpu_device = torch.device("cpu")
    
    # Run benchmarks
    cpu_times = benchmark_matrix_operations(sizes, cpu_device)
    device_times = benchmark_matrix_operations(sizes, device)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, cpu_times, 'o-', label='CPU')
        plt.plot(sizes, device_times, 's-', label=str(device).upper())
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (seconds)')
        plt.title('Matrix Multiplication Performance')
        plt.legend()
        plt.grid(True)
        
        # Calculate and show speedup
        speedups = [c/d if d > 0 else 0 for c, d in zip(cpu_times, device_times)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, speedups, 'o-')
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup (x)')
        plt.title(f'Speedup using {str(device).upper()} vs CPU')
        plt.grid(True)
        plt.axhline(y=1, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.show()
    
    # Return results dictionary
    return {
        'sizes': sizes,
        'cpu_times': cpu_times,
        'device_times': device_times,
        'speedups': [c/d if d > 0 else 0 for c, d in zip(cpu_times, device_times)]
    }

def main():
    """Main function when script is run directly."""
    device = print_system_info()
    test_basic_operations(device)
    
    # Run a small benchmark by default
    if device.type != "cpu":  # Only if GPU is available
        run_benchmark(max_size=2000, step=500, device=device)

if __name__ == "__main__":
    main()