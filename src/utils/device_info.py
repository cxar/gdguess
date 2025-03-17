#!/usr/bin/env python3
"""
Utility to print detailed information about available computing devices.
"""

import os
import platform
import sys
import torch
import psutil
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

def get_cpu_info():
    """Get detailed CPU information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "python_version": platform.python_version(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }
    
    # Add macOS specific info
    if platform.system() == 'Darwin':
        try:
            # Try to get Mac model information
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                info['cpu_model'] = result.stdout.strip()
                
            # Check if running on Apple Silicon
            result = subprocess.run(['sysctl', '-n', 'hw.optional.arm64'], 
                                   capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip() == '1':
                info['apple_silicon'] = True
        except Exception as e:
            print(f"Could not get detailed Mac info: {e}")
    
    return info


def get_cuda_info():
    """Get detailed CUDA GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": [],
        "version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else None,
    }
    
    # Get detailed per-device information
    for i in range(torch.cuda.device_count()):
        device_info = {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}",
        }
        
        # Add memory info if possible
        if hasattr(torch.cuda, 'get_device_properties'):
            props = torch.cuda.get_device_properties(i)
            device_info.update({
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "sm_count": props.multi_processor_count,
            })
        
        # Add NVML info if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_info.update({
                    "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                    "power_usage_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                    "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0,
                    "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                })
            except Exception as e:
                device_info["nvml_error"] = str(e)
        
        info["devices"].append(device_info)
    
    return info


def get_mps_info():
    """Get information about Metal Performance Shaders (MPS) on Apple Silicon."""
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "is_built": torch.backends.mps.is_built(),
        "fallback_to_cpu_enabled": getattr(torch.backends.mps, "fallback_to_cpu", False),
    }
    
    # Try to get M-series chip information on macOS
    if platform.system() == 'Darwin':
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.model'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                info['model'] = result.stdout.strip()
                
            # Try to get memory info
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                try:
                    mem_bytes = int(result.stdout.strip())
                    info['total_memory_gb'] = round(mem_bytes / (1024**3), 2)
                except ValueError:
                    pass
                
            # Try to get performance/efficiency core counts
            result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) == 2:
                    try:
                        p_cores = int(parts[0])
                        e_cores = int(parts[1])
                        info['performance_cores'] = p_cores
                        info['efficiency_cores'] = e_cores
                    except ValueError:
                        pass
                    
            # GPU core count (harder to get reliably)
            try:
                # This would require a helper tool in practice
                info['gpu_cores'] = "Not available via API"
            except Exception:
                pass
                
        except Exception as e:
            info['error'] = str(e)

    return info


def print_device_summary():
    """Print a summary of all available computing devices."""
    print("-" * 60)
    print("SYSTEM INFORMATION")
    print("-" * 60)
    
    cpu_info = get_cpu_info()
    print(f"Platform: {cpu_info['platform']}")
    print(f"Python: {cpu_info['python_version']}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU: {cpu_info.get('cpu_model', cpu_info['processor'])}")
    print(f"Cores: {cpu_info['cores_physical']} physical, {cpu_info['cores_logical']} logical")
    print(f"Memory: {cpu_info['memory_total_gb']} GB total, {cpu_info['memory_available_gb']} GB available")
    
    print("\n" + "-" * 60)
    print("GPU INFORMATION")
    print("-" * 60)
    
    # Check for CUDA
    cuda_info = get_cuda_info()
    if cuda_info["available"]:
        print(f"CUDA: Available (version {cuda_info['version']})")
        print(f"cuDNN: {cuda_info['cudnn_version']}")
        print(f"GPU count: {cuda_info['device_count']}")
        
        for i, device in enumerate(cuda_info["devices"]):
            print(f"\nGPU {i}: {device['name']}")
            print(f"  Compute capability: {device['capability']}")
            if 'total_memory_gb' in device:
                print(f"  Memory: {device['total_memory_gb']} GB")
            if 'sm_count' in device:
                print(f"  Multiprocessors: {device['sm_count']}")
            if 'temperature' in device:
                print(f"  Temperature: {device['temperature']}°C")
            if 'utilization' in device:
                print(f"  Utilization: {device['utilization']}%")
            if 'power_usage_w' in device:
                print(f"  Power usage: {device['power_usage_w']:.1f}W / {device['power_limit_w']:.1f}W")
    else:
        print("CUDA: Not available")
    
    # Check for Apple Silicon MPS
    mps_info = get_mps_info()
    if mps_info["available"]:
        print("\nMPS (Apple Silicon): Available")
        print(f"Model: {mps_info.get('model', 'Unknown')}")
        if 'total_memory_gb' in mps_info:
            print(f"Memory: {mps_info['total_memory_gb']} GB")
        if 'performance_cores' in mps_info:
            print(f"CPU cores: {mps_info['performance_cores']} performance, {mps_info['efficiency_cores']} efficiency")
        print(f"Fallback to CPU: {'Enabled' if mps_info['fallback_to_cpu_enabled'] else 'Disabled'}")
    else:
        print("\nMPS (Apple Silicon): Not available")
    
    print("\n" + "-" * 60)
    print("RECOMMENDED CONFIGURATION")
    print("-" * 60)
    
    # Make device recommendations
    if cuda_info["available"]:
        print("Recommended device: CUDA")
        print("Example command: python train.py --device cuda")
    elif mps_info["available"]:
        print("Recommended device: MPS (Apple Silicon)")
        print("Example command: python train.py --device mps --mps-fallback")
    else:
        print("Recommended device: CPU (no GPU acceleration available)")
        print("Example command: python train.py --device cpu")
    
    # Additional flags
    if mps_info["available"]:
        print("\nAdditional MPS options:")
        print("  --mps-fallback   Enable CPU fallback for unsupported MPS operations")
        print("  --mps-optimize   Enable MPS-specific optimizations")
    
    # Batch size recommendations
    if cuda_info["available"]:
        memory_gb = cuda_info["devices"][0].get("total_memory_gb", 0)
        if memory_gb > 20:
            batch_size = 512
        elif memory_gb > 10:
            batch_size = 256
        elif memory_gb > 6:
            batch_size = 128
        else:
            batch_size = 64
    elif mps_info["available"]:
        memory_gb = mps_info.get("total_memory_gb", 0)
        if memory_gb > 16:
            batch_size = 128
        else:
            batch_size = 64
    else:
        # CPU recommendation
        batch_size = 32
    
    print(f"\nRecommended batch size: {batch_size}")
    print(f"Example: python train.py --batch-size {batch_size}")


if __name__ == "__main__":
    print_device_summary() 