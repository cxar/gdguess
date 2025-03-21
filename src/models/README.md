# Models Module

This module contains the model architecture components for the Grateful Dead show dating project.

## Directory Structure

- **`models/`**: Main models module
  - **`components/`**: Reusable model building blocks
    - **`attention.py`**: Attention mechanisms
    - **`audio_processing.py`**: Audio-specific processing layers
    - **`blocks.py`**: Common building blocks (Conv, ResNet, etc.)
  - **`feature_extractors.py`**: Feature extraction components
  - **`losses.py`**: Loss functions for model training
  - **`dead_model.py`**: Main model architecture implementation
  - **`dtype_helpers.py`**: Helper functions for handling different data types

## Key Components

### Model Architecture

The core model architecture in `dead_model.py` processes multiple audio features through parallel branches:

1. **Feature Extraction**: Extracts features from raw audio including:
   - Mel spectrograms
   - Harmonic and percussive components
   - Chroma features
   - Spectral contrast

2. **Feature Processing**: Each feature is processed through its own network branch

3. **Feature Fusion**: Results from all branches are combined

4. **Date Prediction**: Final layers estimate the date with uncertainty

### Loss Functions

The model uses several specialized loss functions defined in `losses.py`:

- Date regression loss with uncertainty
- Seasonal pattern regularization
- Dynamic loss weighting

## Usage

```python
from src.models.dead_model import DeadModel

# Create model instance
model = DeadModel(
    input_channels={
        "harmonic": 1,
        "percussive": 1,
        "chroma": 12,
        "contrast": 7
    },
    emb_dim=128
)

# Forward pass with a batch of features
outputs = model({
    "harmonic": harmonic_features,
    "percussive": percussive_features,
    "chroma": chroma_features,
    "contrast": contrast_features
})
```