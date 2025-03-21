# Evaluation Module

This module contains tools and utilities for evaluating the Grateful Dead show dating model.

## Directory Structure

- **`evaluation/`**: Main evaluation module
  - **`evaluate.py`**: Core evaluation functionality
  - **`metrics.py`**: Evaluation metrics implementations

## Key Components

### Evaluation Functions

The module provides functions for evaluating model performance:

- `evaluate_model`: Comprehensive model evaluation on a dataset
- `evaluate_batch`: Single batch evaluation
- `evaluate_uncertainty`: Evaluate uncertainty estimation quality

### Metrics

The `metrics.py` file implements various metrics for model evaluation:

- Date regression metrics (MAE, RMSE)
- Year and month accuracy
- Calibration metrics for uncertainty estimation
- Seasonal pattern recognition accuracy

## Usage

```python
from src.evaluation.evaluate import evaluate_model
from src.data.dataset import create_dataloaders

# Create a validation dataloader
_, val_loader = create_dataloaders(
    data_dir="/path/to/data",
    batch_size=32,
    val_only=True
)

# Evaluate a trained model
results = evaluate_model(
    model=model,
    val_loader=val_loader,
    device="cuda"
)

print(f"Mean Absolute Error: {results['mae']:.2f} days")
print(f"Year Accuracy: {results['year_acc']:.2%}")
print(f"Month Accuracy: {results['month_acc']:.2%}")
```