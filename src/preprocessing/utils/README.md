# Preprocessing Utilities

This directory contains utility functions for preprocessing audio files for the Grateful Dead show dating model.

## Directory Structure

- **`compare_outputs.py`**: Tools to compare outputs from different preprocessing methods
- **`validate_preprocessing.py`**: Validation tools for preprocessing output

## Usage

```python
from src.preprocessing.utils.compare_outputs import compare_feature_outputs
from src.preprocessing.utils.validate_preprocessing import validate_dataset

# Compare features from different preprocessing methods
compare_feature_outputs('turbo_output.pt', 'fast_output.pt', plot=True)

# Validate a preprocessed dataset
valid, issues = validate_dataset('/path/to/dataset')
```

## Functions

- `compare_feature_outputs`: Compare feature outputs from different preprocessing methods
- `validate_dataset`: Check a dataset for common issues
- `fix_dataset_issues`: Attempt to fix common dataset issues