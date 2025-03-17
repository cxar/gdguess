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

To train the model:

```bash
python -m training.run_training --data-dir /path/to/data --batch-size 16 --steps 10000
```

Optional arguments:
- `--config`: Path to a config file
- `--lr`: Initial learning rate
- `--checkpoint`: Path to checkpoint to resume from
- `--device`: Device to use (cuda, mps, cpu, auto)

## Recent Improvements

- Reorganized the model architecture for better modularity
- Simplified device handling code to reduce redundancy
- Created a Trainer class to encapsulate the training loop
- Improved error handling and validation
- Enhanced memory management, especially for MPS (Apple Silicon) devices