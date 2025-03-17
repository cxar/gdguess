#!/usr/bin/env python3
"""
Configuration settings for the Grateful Dead show dating model.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch

# Import helpers for device consistency
from utils.helpers import ensure_device_consistency, prepare_batch  # Import for API compatibility


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 24000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    feature_cache_size: int = 128
    

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Feature dimensions
    harmonic_dim: int = 1
    percussive_dim: int = 1
    chroma_dim: int = 12
    spectral_contrast_dim: int = 6
    
    # Network architecture
    feature_dims: List[int] = field(default_factory=lambda: [32, 64])
    attention_heads: int = 4
    attention_dropout: float = 0.1
    attention_chunk_size: int = 128
    
    # Seasonal pattern module
    seasonal_hidden_dim: int = 256
    num_seasonal_components: int = 4
    
    # Final layers
    final_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.1])


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 128
    num_epochs: int = 100
    
    # Loss weights
    periodicity_loss_weight: float = 0.1
    uncertainty_loss_weight: float = 1.0
    
    # Learning rate scheduling
    lr_warmup_epochs: int = 5
    lr_decay_factor: float = 0.1
    lr_decay_epochs: List[int] = field(default_factory=lambda: [60, 80])


@dataclass
class Config:
    """Main configuration class combining all settings."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Dataset
    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training control
    total_training_steps: int = 180000
    use_early_stopping: bool = True
    patience: int = 10
    use_jit: bool = True
    use_augmentation: bool = True
    resume_checkpoint: Optional[str] = None
    run_lr_finder: bool = False
    disable_preprocessing: bool = True  # Disable preprocessing by default
    
    # Logging and checkpoints
    log_dir: str = "logs"
    output_dir: str = "output"  # Output directory for model outputs and checkpoints
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10
    log_interval: int = 100
    validation_interval: int = 1
    checkpoint_interval: int = 5
    
    # Device and performance
    device: str = "auto"  # Will automatically choose best available device (cuda > mps > cpu)
    force_device_consistency: bool = True  # Force tensors to be on the same device
    force_inputs_to_device: bool = True  # Move inputs to the target device
    disable_device_consistency: bool = False  # Disable device consistency checks (for torch.compile)
    num_workers: int = 16
    pin_memory: bool = True
    use_mixed_precision: bool = True
    grad_clip_value: float = 1.0
    
    # MPS specific settings
    mps_fallback_to_cpu: bool = True  # If MPS operation not supported, fallback to CPU
    mps_enable_all_optimizations: bool = True  # Enable all MPS optimizations
    
    # H200 specific optimizations
    use_torch_compile: bool = False
    use_cudnn_benchmark: bool = True
    compile_mode: str = "reduce-overhead"
    
    # Audio processing
    target_sr: int = 24000
    
    # Optimization
    initial_lr: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 128
    
    def __post_init__(self):
        """Setup after initialization to handle device configuration."""
        # Auto-detect device if set to "auto"
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                # MPS doesn't support mixed precision yet
                self.use_mixed_precision = False
                # Adjust workers for MPS
                import os
                self.num_workers = min(self.num_workers, os.cpu_count() or 4)
            else:
                self.device = "cpu"
                self.use_mixed_precision = False
        
        # Apply device-specific settings
        if self.device == "mps":
            # MPS doesn't support mixed precision
            self.use_mixed_precision = False
            # JIT compilation is experimental on MPS
            self.use_jit = False
            # Configure MPS fallback behavior
            if hasattr(torch.backends, "mps"):
                torch.backends.mps.fallback_to_cpu = self.mps_fallback_to_cpu


# Default configuration
config = Config()

def get_training_config(**overrides):
    """Get training configuration with optional overrides."""
    config = Config()
    
    # Apply overrides
    for key, value in overrides.items():
        if "." in key:
            # Handle nested parameters (e.g., "training.batch_size")
            section, param = key.split(".", 1)
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, param):
                    setattr(section_config, param, value)
                else:
                    raise ValueError(f"Unknown parameter '{param}' in section '{section}'")
            else:
                raise ValueError(f"Unknown configuration section: {section}")
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    # Apply post-initialization logic
    config.__post_init__()
    
    return config
