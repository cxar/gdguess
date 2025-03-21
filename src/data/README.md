# Data Processing Module

This module handles all aspects of data processing and dataset management for the Grateful Dead show dating model.

## Directory Structure

- **`data/`**: Main data processing module
  - **`dataset.py`**: Core dataset implementation with PyTorch DataLoader integration
  - **`augmentation.py`**: Data augmentation techniques for training
  - **`preprocessing.py`**: Functions for preprocessing data before training
  - **`create_small_dataset.py`**: Utility to create smaller dataset for testing

## Key Components

### Dataset Classes

- `GratefulDeadDataset`: The primary dataset class for loading processed audio features
- `ValidationDataset`: Specialized dataset for validation data handling

### Augmentation

The module provides various augmentation techniques to improve model robustness:
- Time stretch and pitch shift
- Noise injection
- Feature masking
- Mixup augmentation

## Usage

```python
from src.data.dataset import GratefulDeadDataset, create_dataloaders

# Create training and validation dataloaders
train_loader, val_loader = create_dataloaders(
    data_dir="/path/to/data",
    batch_size=16,
    num_workers=4
)
```

For creating a smaller dataset for testing:

```bash
python -m src.data.create_small_dataset --input-dir /path/to/full/dataset --output-dir /path/to/small/dataset --samples 100
```