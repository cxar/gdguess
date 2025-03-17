# Preprocessing Module

This module contains all the scripts and utilities for preprocessing audio files for the Grateful Dead show dating model.

## Directory Structure

- **`preprocessing/`**: Main preprocessing module
  - **`turboprocess.py`**: High-performance preprocessing implementation with GPU acceleration
  - **`fastprocess.py`**: Faster preprocessing implementation focused on Apple Silicon
  - **`preprocess.py`**: Original preprocessing script (simpler but slower)
  - **`utils/`**: Utility functions for preprocessing
    - **`compare_outputs.py`**: Tools to compare outputs from different preprocessing methods
  - **`profiling/`**: Profiling and optimization tools
    - **`profile_turboprocess.py`**: Profiling for turboprocess
    - **`profile_chroma.py`**: Dedicated profiling for chroma feature extraction
    - **`optimize_chroma.py`**: Optimized implementations of chroma extraction

## Usage

### Basic Preprocessing

For normal preprocessing, use `turboprocess.py`:

```bash
python -m src.preprocessing.turboprocess --input-dir /path/to/audio --use-gpu
```

### Faster Preprocessing on Apple Silicon

For optimized performance on Apple Silicon:

```bash
python -m src.preprocessing.turboprocess --input-dir /path/to/audio --use-gpu --mps-optimize
```

### Profiling

To profile the preprocessing performance:

```bash
python -m src.preprocessing.profiling.profile_turboprocess
```

## Function Reference

Main functions exposed by the module:

- `find_audio_files`: Find audio files with valid dates in a directory tree
- `get_feature_extractor`: Create a feature extractor for audio processing
- `process_batch`: Process a batch of files in parallel
- `init_gpu`, `init_mps`: Initialize GPU/MPS device with optimized settings 