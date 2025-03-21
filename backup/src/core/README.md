# Simplified Grateful Dead Show Dating Model

This is a simplified version of the Grateful Dead show dating project with a focus on readability and clarity. The code has been significantly streamlined to make it easier to understand the model architecture and training process.

## Project Structure

```
src/core/
├── data/
│   ├── dataset.py      # Simplified dataset and data loading
│   └── __init__.py
├── model/
│   ├── model.py        # Core model architecture
│   ├── loss.py         # Simplified loss function
│   └── __init__.py
├── train/
│   ├── train.py        # Training script
│   ├── trainer.py      # Trainer class
│   └── __init__.py
├── utils/
│   ├── utils.py        # Utility functions
│   └── __init__.py
├── main.py             # Simple command line interface
└── README.md           # This file
```

## Key Components

### Model Architecture

The model processes multiple audio features in parallel branches:
- Harmonic features
- Percussive features  
- Chroma features
- Spectral contrast features

These are processed through separate convolutional branches, then combined and enhanced with a seasonal pattern module to provide a final date prediction with uncertainty estimation.

### Training

Training is handled by the `SimpleTrainer` class which implements:
- Basic training and validation loops
- Checkpointing
- TensorBoard logging
- Early stopping

### CLI

The simplified CLI provides three main commands:
- `train`: Train the model on a dataset
- `infer`: Run inference on audio files
- `sysinfo`: Display system information

## Usage

### Training

```bash
# Basic training
python -m src.core.main train --data-dir /path/to/data --batch-size 16

# Resume from checkpoint
python -m src.core.main train --data-dir /path/to/data --checkpoint /path/to/checkpoint.pt
```

### System Information

```bash
python -m src.core.main sysinfo
```

## Customization

The simplified codebase is designed to be easy to understand and modify. Key parameters can be adjusted in:
- `src/core/train/train.py`: Training hyperparameters
- `src/core/model/model.py`: Model architecture
- `src/core/model/loss.py`: Loss function