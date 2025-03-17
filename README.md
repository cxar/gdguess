# Grateful Dead Show Dating Model

A deep learning model for estimating the date of a Grateful Dead concert recording based on audio features.

## Project Organization

The project is organized as follows:

```
├── config/              # Configuration files
├── data/                # Data handling and datasets
├── models/              # Model architecture
│   ├── components/      # Model building blocks
│   └── feature_extractors/ # Feature extraction modules
├── output/              # Output directory for checkpoints
├── src/                 # Source directory
├── training/            # Training code
└── utils/               # Utility functions
```

## Model Architecture

The model processes audio recordings through several feature extraction steps and combines them to predict the date of a Grateful Dead concert.

### Key Components:

- **Audio Feature Extraction**: Processes raw audio to extract mel spectrograms, harmonic and percussive components, chroma features, and spectral contrast.
- **Feature Network**: Parallel branches process different audio features, which are then concatenated and fused.
- **Seasonal Pattern Module**: Models the seasonal patterns in concert dates.
- **Output Heads**: Predicts the date and provides uncertainty estimation.

## Training

The training process uses Stochastic Weight Averaging (SWA) to improve model generalization and supports multiple device types (CUDA, MPS, CPU).

### Training Features:

- Learning rate warmup and cosine decay
- Dynamic loss weighting
- Uncertainty-aware loss function
- Seasonal pattern recognition
- Early stopping and checkpointing

## Usage

The project includes a single integrated command-line interface for all operations:

```bash
# Training
./gdguess.py train --data-dir /path/to/data --batch-size 16 --steps 10000

# Inference
./gdguess.py infer --model /path/to/model --input /path/to/audio

# Inspect model checkpoints
./gdguess.py inspect /path/to/checkpoint.pt

# Check system information
./gdguess.py sysinfo --test-device --benchmark
```

### Apple Silicon GPU Acceleration

The model supports training and inference on Apple Silicon GPUs using Metal Performance Shaders (MPS).

Requirements:
- Apple Silicon Mac (M1/M2/M3 series)
- macOS 12.3+ (Monterey or newer)
- PyTorch 1.13+ (PyTorch 2.0+ recommended)

To use Apple Silicon GPU acceleration, use the `--device mps` flag:

```bash
./gdguess.py train --device mps --data-dir /path/to/data
```

## Recent Improvements

- Reorganized the model architecture for better modularity
- Simplified device handling code to reduce redundancy
- Created a Trainer class to encapsulate the training loop
- Improved error handling and validation
- Enhanced memory management, especially for MPS (Apple Silicon) devices
- Consolidated utility scripts into a more coherent structure
- Added comprehensive inspection tools for PyTorch checkpoints
- Created a unified command-line interface for all operations