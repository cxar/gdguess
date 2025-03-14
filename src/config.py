#!/usr/bin/env python3
"""
Configuration settings for the Grateful Dead show dating model.
"""

import datetime
from typing import Dict


def get_default_config() -> Dict:
    """
    Get the default configuration for training.

    Returns:
        Dictionary containing default configuration values
    """
    base_date = datetime.date(1968, 1, 1)

    return {
        # Data settings
        "input_dir": "../../data/audsnippets-all",
        "target_sr": 24000,
        "base_date": base_date,
        # Training settings
        "batch_size": 64,  # Reduced from 128 for better stability
        "initial_lr": 3e-5,  # Slightly lower learning rate
        "weight_decay": 0.02,  # Increased regularization
        "num_workers": 8,
        "valid_split": 0.15,  # Larger validation set
        "use_augmentation": True,
        # Checkpoint settings
        "resume_checkpoint": "./checkpoints/checkpoint_latest.pt",
        "latest_checkpoint": "checkpoint_latest.pt",
        "best_checkpoint": "checkpoint_best.pt",
        "save_every_n_epochs": 1,
        # Training schedule
        "total_training_steps": 200000,  # More training steps
        # Learning rate finder
        "run_lr_finder": True,  # Enable by default for new model
        # Early stopping
        "use_early_stopping": True,
        "patience": 15,  # Increased patience
        "min_delta": 0.3,  # More sensitive to improvements
        # Performance optimization
        "use_jit": False,
        # Gradient clipping
        "grad_clip_value": 2.0,  # Add explicit gradient clipping
    }


def get_training_config(**kwargs) -> Dict:
    """
    Get training configuration with optional overrides.

    Args:
        **kwargs: Overrides for default configuration values

    Returns:
        Dictionary containing configuration values
    """
    config = get_default_config()
    config.update(kwargs)
    return config
