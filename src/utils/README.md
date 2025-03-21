# Utilities Module

This module provides utility functions and helper tools for the Grateful Dead show dating project.

## Directory Structure

- **`utils/`**: Main utilities module
  - **`device_utils.py`**: Device detection and optimization utilities
  - **`helpers.py`**: General helper functions
  - **`inspection.py`**: Checkpoint inspection tools
  - **`visualization.py`**: Visualization utilities
  - **`h200_optimizations.py`**: Optimizations for H200 hardware

## Key Components

### Device Utilities

The `device_utils.py` file provides functions for device management:

- `get_device`: Detect and select the best available device
- `print_system_info`: Display system information
- `run_benchmark`: Run a device performance benchmark
- Device-specific optimizations for CUDA and MPS

### Visualization

The `visualization.py` file contains functions for visualizing:

- Training progress and metrics
- Model predictions
- Audio features and spectrograms
- Uncertainty visualization

### Inspection

The `inspection.py` file provides tools for checkpoint inspection:

- Detailed model information
- Parameter statistics
- Layer structure visualization
- Checkpoint comparison

## Usage

### Device Management

```python
from src.utils.device_utils import get_device, print_system_info

# Print system information
print_system_info()

# Get the best available device
device = get_device()
print(f"Using device: {device}")
```

### Checkpoint Inspection

```python
from src.utils.inspection import inspect_pt_file

# Inspect a checkpoint file
inspect_pt_file("/path/to/checkpoint.pt")
```