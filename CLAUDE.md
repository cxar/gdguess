# Claude Helper for Grateful Dead Show Dating Project

This file contains useful information for Claude to assist with the project.

## Project Structure

- `src/`: Main source code directory
  - `models/`: Model architecture
    - `components/`: Model building blocks
    - `feature_extractors/`: Feature extraction modules
  - `data/`: Data handling and loading
  - `training/`: Training logic
  - `utils/`: Utility functions

## Command-Line Interface

The project uses a unified command-line interface:

```bash
# Training
./gdguess.py train --data-dir /path/to/data --batch-size 16 --steps 10000

# Inference
./gdguess.py infer --model /path/to/model --input /path/to/audio

# Inspect checkpoints
./gdguess.py inspect /path/to/checkpoint.pt

# System info and benchmarking
./gdguess.py sysinfo --test-device --benchmark

# Running tests
./gdguess.py test --all
```

## Development Guidelines

1. **Code Style**:
   - Use consistent Python naming conventions (snake_case for functions, PEP8)
   - Include docstrings for all functions and classes
   - Use type hints

2. **Model Architecture**:
   - Place reusable components in `models/components/`
   - Keep device-specific code to a minimum
   - Implement proper error checking

3. **Training**:
   - Use the Trainer class in `training/trainer.py`
   - Support multiple devices (CPU, CUDA, MPS)
   - Implement proper checkpointing

4. **Testing**:
   - Add tests to the `tests/` directory
   - Run tests with `./gdguess.py test --all`