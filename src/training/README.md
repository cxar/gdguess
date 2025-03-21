# Training Module

This module contains all code related to model training for the Grateful Dead show dating project.

## Directory Structure

- **`training/`**: Main training module
  - **`run_training.py`**: Standard training script
  - **`run_unified_training.py`**: Optimized unified training script
  - **`trainer.py`**: Main Trainer class implementation
  - **`lr_finder.py`**: Learning rate finder utility
  - **`experimental/`**: Experimental training approaches
    - **`fast_synthetic_training.py`**: Training with synthetic data
    - **`minimal_training_test.py`**: Minimal test implementation
    - **`optimized_mps_training.py`**: Optimized training for Apple Silicon
  - **`legacy/`**: Legacy trainer implementations
    - **`trainer_legacy.py`**: Previous trainer version

## Key Components

### Trainer Class

The `Trainer` class in `trainer.py` is the central component that handles:

- Training and validation loops
- Learning rate scheduling
- Checkpointing and model saving
- TensorBoard logging
- Early stopping
- Device-specific optimizations

### Training Scripts

Multiple training scripts are provided:

- `run_training.py`: Standard training script with basic options
- `run_unified_training.py`: Optimized training with advanced device handling
- Experimental scripts for testing and specialized training scenarios

## Usage

### Through Command Line

The recommended way to train is through the command-line interface:

```bash
# Basic training with auto-optimized settings
./gdguess.py train --data-dir /path/to/data --batch-size 16 --tensorboard

# Training with specific settings
./gdguess.py train --data-dir /path/to/data --batch-size 16 --learning-rate 0.0005 --steps 10000 --early-stopping --tensorboard
```

### Programmatically

```python
from src.training.trainer import Trainer
from src.models.dead_model import DeadModel
from src.data.dataset import create_dataloaders

# Create model and dataloaders
model = DeadModel()
train_loader, val_loader = create_dataloaders(data_dir="/path/to/data", batch_size=16)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=0.0005,
    device="cuda"
)

# Train model
trainer.train(max_steps=10000)
```