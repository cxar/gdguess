#!/usr/bin/env python3
"""
Utility functions for loading models and checkpoints.
"""

import datetime
import torch
import torch.nn as nn

from models.dead_model import DeadShowDatingModel


def load_model(checkpoint_path, device="cpu"):
    """
    Load the model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on (cpu, cuda, mps)
        
    Returns:
        model: Loaded model
        base_date: Base date used for day offset calculations
    """
    print(f"Loading model from {checkpoint_path}")

    # Handle different possible checkpoint formats
    try:
        # First try with PyTorch 2.0+ approach using safe globals
        import torch.serialization

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([datetime.date])
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            # Fall back to older PyTorch versions
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
    except (TypeError, AttributeError):
        # Final fallback for even older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    model = DeadShowDatingModel()

    # Handle different key formats in checkpoints
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint

    # Load state dict, handling DataParallel prefix if present
    if list(model_state.keys())[0].startswith("module."):
        # Model was saved using DataParallel
        model = nn.DataParallel(model)
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # Move to the specified device
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Get config from checkpoint if it exists
    config = checkpoint.get("config", {})
    base_date = config.get("base_date", datetime.date(1968, 1, 1))

    return model, base_date 