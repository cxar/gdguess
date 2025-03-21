# Tools Module

This module contains debugging tools and utilities for the Grateful Dead show dating project.

## Directory Structure

- **`tools/`**: Main tools module
  - **`check_batch.py`**: Batch inspection utility
  - **`check_dataloader.py`**: DataLoader inspection and validation
  - **`check_model.py`**: Model inspection and testing
  - **`debug_loss.py`**: Loss function debugging
  - **`minimal_test.py`**: Minimal test implementation
  - **`profile_training.py`**: Training profile and optimization
  - **`real_data_test.py`**: Testing with real-world data

## Key Components

### Debugging Tools

The module provides various debugging utilities:

- `check_batch.py`: Inspect batch shapes, values, and statistics
- `check_dataloader.py`: Verify DataLoader outputs and consistency
- `check_model.py`: Test model forward pass and parameter initialization

### Profiling

The `profile_training.py` tool profiles the training loop to identify bottlenecks:

- Memory usage tracking
- Step timing analysis
- Component-level profiling (data loading, forward pass, backward pass)

## Usage

### Check DataLoader

```bash
python -m src.tools.check_dataloader --data-dir /path/to/data --batch-size 16 --num-batches 10
```

### Profile Training

```bash
python -m src.tools.profile_training --data-dir /path/to/data --batch-size 16 --steps 100
```

### Debug Loss

```bash
python -m src.tools.debug_loss --data-dir /path/to/data --batch-size 8
```