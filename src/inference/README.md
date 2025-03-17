# Inference Module

This module contains all the scripts and utilities for performing inference with the Grateful Dead show dating model.

## Directory Structure

- **`inference/`**: Main inference module
  - **`base_inference.py`**: Core inference functionality
  - **`interactive/`**: Interactive inference tools
    - **`mic_capture.py`**: Microphone audio capture for real-time inference
    - **`visualization.py`**: Visualization utilities for interactive mode
    - **`mic_inference.py`**: Interactive inference with microphone input
  - **`utils/`**: Utility functions
    - **`model_loader.py`**: Utilities for loading model checkpoints

## Usage

### Basic Inference

To run inference on an audio file:

```python
from inference import predict_date
from inference.utils import load_model

# Load model
model, base_date = load_model('path/to/checkpoint.pt', device='cuda')

# Run inference
prediction = predict_date(model, 'path/to/audio.mp3', base_date)
print(f"Predicted date: {prediction['predicted_date']}")
```

### Interactive Inference

For interactive inference with microphone input:

```bash
python -m src.inference.interactive.mic_inference --checkpoint path/to/checkpoint.pt
```

## Function Reference

- `extract_audio_features`: Extract audio features for the model
- `predict_date`: Predict the date of a Grateful Dead performance
- `load_model`: Load a model from a checkpoint 