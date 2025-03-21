# Preprocessing Profiling

This directory contains tools for profiling and optimizing the preprocessing pipeline for the Grateful Dead show dating model.

## Directory Structure

- **`profile_turboprocess.py`**: Profiling for the high-performance preprocessing implementation
- **`profile_chroma.py`**: Dedicated profiling for chroma feature extraction
- **`optimize_chroma.py`**: Optimized implementations of chroma extraction

## Usage

### Profile Turboprocess

```bash
# Profile the turboprocess preprocessing implementation
python -m src.preprocessing.profiling.profile_turboprocess --sample-count 10
```

### Profile Chroma Feature Extraction

```bash
# Profile chroma feature extraction specifically
python -m src.preprocessing.profiling.profile_chroma --device mps
```

### Test Optimized Implementations

```bash
# Compare optimized vs standard implementations
python -m src.preprocessing.profiling.optimize_chroma --compare
```

## Profiling Results

The profiling tools generate performance reports that can be used to identify bottlenecks in the preprocessing pipeline. 
Reports are saved to the `profiling_results/` directory by default.