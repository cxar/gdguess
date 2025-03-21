"""
Utility functions for the Grateful Dead show dating project.
"""

import os
import torch


def get_device():
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device: The optimal device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_system_info():
    """
    Print system information including PyTorch version and available devices.
    
    Returns:
        torch.device: The optimal device
    """
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Apple Silicon) is available")
    
    # Fall back to CPU
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def save_checkpoint(model, optimizer, scheduler, epoch, step, best_loss, path):
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        epoch: Current epoch
        step: Current step
        best_loss: Best validation loss
        path: Path to save the checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss
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


def load_checkpoint(model, optimizer, scheduler, path, device=None):
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        path: Path to load the checkpoint from
        device: Device to load the checkpoint to
        
    Returns:
        tuple: (epoch, step, best_loss)
    """
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found")
        return 0, 0, float("inf")
    
    # Map checkpoint to specified device or model's device
    map_location = device if device else next(model.parameters()).device
    checkpoint = torch.load(path, map_location=map_location)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler if available
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Get training info
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))
    
    print(f"Loaded checkpoint from {path} (epoch {epoch}, step {step})")
    
    return epoch, step, best_loss