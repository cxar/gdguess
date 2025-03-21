# Inference Utilities

This directory contains utility functions for inference with the Grateful Dead show dating model.

## Directory Structure

- **`model_loader.py`**: Functions for loading model checkpoints
- **`tta.py`**: Test-time augmentation utilities
- **`uncertainty.py`**: Utilities for handling prediction uncertainty

## Usage

```python
from src.inference.utils.model_loader import load_model
from src.inference.utils.uncertainty import get_uncertainty_bounds

# Load a model from checkpoint
model, base_date = load_model('path/to/checkpoint.pt', device='cuda')

# Get uncertainty bounds for a prediction
lower_bound, upper_bound = get_uncertainty_bounds(prediction, confidence=0.95)
```

## Functions

- `load_model`: Load a model from a checkpoint
- `apply_tta`: Apply test-time augmentation to increase prediction robustness
- `get_uncertainty_bounds`: Calculate confidence intervals for predictions