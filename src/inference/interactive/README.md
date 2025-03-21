# Interactive Inference Module

This module provides tools for real-time, interactive inference with the Grateful Dead show dating model.

## Directory Structure

- **`mic_capture.py`**: Microphone audio capture functionality
- **`mic_inference.py`**: Main script for interactive inference
- **`mic_interface.py`**: User interface for microphone-based inference
- **`visualization.py`**: Real-time visualization of model predictions

## Usage

```bash
# Start interactive inference with microphone input
python -m src.inference.interactive.mic_inference --checkpoint path/to/model.pt
```

## Key Features

- Real-time audio processing from microphone
- Instant feedback on prediction quality
- Visualization of prediction uncertainty
- Time travel visualization (prediction changes over time)

## Requirements

- PyAudio
- Matplotlib
- NumPy
- A trained model checkpoint