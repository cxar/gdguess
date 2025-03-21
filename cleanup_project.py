#!/usr/bin/env python3
"""
Script to clean up project structure by removing alternate implementations and 
consolidating code to a single clear implementation.
"""

import os
import shutil
import sys

def ensure_dir(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def backup_dir(directory):
    """Backup a directory."""
    backup_dir = f"{directory}_backup"
    if os.path.exists(directory):
        print(f"Backing up {directory} to {backup_dir}")
        shutil.copytree(directory, backup_dir)

# Files to remove (relative to project root)
FILES_TO_REMOVE = [
    # Alternate entry points
    "gdguess_core.py",
    "minimal_core.py",
    "minimal_test.py",
    "train_simple.py",
    
    # Deprecated implementation guides
    "implementation_guide.py",
    
    # Core implementation directory
    "src/core",
]

# Ensure we're in the project root
project_root = os.getcwd()

# Create backup directory
backup_dir = os.path.join(project_root, "backup_implementations")
ensure_dir(backup_dir)

# Process each file to remove
for file_path in FILES_TO_REMOVE:
    full_path = os.path.join(project_root, file_path)
    if os.path.exists(full_path):
        print(f"Moving {file_path} to backup directory")
        
        # Create the target directory structure in the backup
        target_dir = os.path.dirname(os.path.join(backup_dir, file_path))
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        # Move to backup if it's a file
        if os.path.isfile(full_path):
            shutil.copy2(full_path, os.path.join(backup_dir, file_path))
            os.remove(full_path)
        # Move to backup if it's a directory
        elif os.path.isdir(full_path):
            shutil.copytree(full_path, os.path.join(backup_dir, file_path))
            shutil.rmtree(full_path)
            
        print(f"Removed {file_path}")
    else:
        print(f"Warning: {file_path} not found")

# Update the README to reflect changes
readme_path = os.path.join(project_root, "README.md")
with open(readme_path, "r") as f:
    readme_content = f.read()

# Replace mentions of alternative implementations
updated_readme = readme_content.replace("- **Core Implementation** (`gdguess_core.py`)", "")
updated_readme = updated_readme.replace("- **Minimal Implementation** (`minimal_core.py`)", "")
updated_readme = updated_readme.replace("├── minimal_core.py      # Ultra-minimal implementation", "")
updated_readme = updated_readme.replace("├── gdguess_core.py", "")
updated_readme = updated_readme.replace("├── train_simple.py", "")
updated_readme = updated_readme.replace("├── minimal_test.py", "")
updated_readme = updated_readme.replace("│   ├── core/            # Simplified core implementation", "")
updated_readme = updated_readme.replace("│   │   ├── data/        # Core data handling", "")
updated_readme = updated_readme.replace("│   │   ├── model/       # Core model implementation", "")
updated_readme = updated_readme.replace("│   │   └── train/       # Core training loop", "")

# Write updated README
with open(readme_path, "w") as f:
    f.write(updated_readme)
print("Updated README.md")

# Update Architecture document if it exists
arch_path = os.path.join(project_root, "ARCHITECTURE.md")
if os.path.exists(arch_path):
    with open(arch_path, "r") as f:
        arch_content = f.read()
    
    # Remove alternate implementation sections
    updated_arch = arch_content.replace(
        """## Code Variants

The project includes several variants of the implementation with different complexity levels:

1. **Full Implementation**: Complete feature set with all optimizations
   - Entry point: `gdguess.py`
   - Main model: `src/models/dead_model.py`

2. **Core Implementation**: Simplified implementation with essential features
   - Entry point: `gdguess_core.py`
   - Main model: `src/core/model/model.py`

3. **Minimal Implementation**: Ultra-minimal implementation for testing
   - Entry point: `minimal_core.py`
   - Contained in a single file""",
        
        """## Implementation

The project uses a single, robust implementation:

- Entry point: `gdguess.py`
- Main model: `src/models/dead_model.py`
- Training: `unified_training.py`"""
    )
    
    # Write updated architecture doc
    with open(arch_path, "w") as f:
        f.write(updated_arch)
    print("Updated ARCHITECTURE.md")

# Create a new, simplified TRAINING.md
training_path = os.path.join(project_root, "TRAINING.md")
with open(training_path, "w") as f:
    f.write("""# Training the Grateful Dead Show Dating Model

This document provides a guide to training the Grateful Dead show dating model.

## Training the Model

To train the model, use the main CLI interface:

```bash
# Basic training with automatic settings
./gdguess.py train --data-dir /path/to/data --tensorboard

# Training with specific parameters
./gdguess.py train --data-dir /path/to/data --batch-size 16 --learning-rate 0.0005 --tensorboard
```

## Hardware-Specific Training

### CUDA (NVIDIA GPUs)

```bash
# Training on CUDA with large batch size
./gdguess.py train --data-dir /path/to/data --batch-size 32 --device cuda --fp16
```

### MPS (Apple Silicon)

```bash
# For M1/M2/M3 regular models
./gdguess.py train --data-dir /path/to/data --batch-size 8 --device mps

# For M1/M2/M3 Pro/Max/Ultra models
./gdguess.py train --data-dir /path/to/data --batch-size 16 --device mps --aggressive-memory
```

### CPU

```bash
# Multi-core CPU training
./gdguess.py train --data-dir /path/to/data --batch-size 4 --device cpu
```

## Monitoring Training

### TensorBoard

```bash
# Start training with TensorBoard
./gdguess.py train --data-dir /path/to/data --tensorboard --tensorboard-dir ./runs

# In another terminal
tensorboard --logdir=./runs
```

## Advanced Training Options

### Checkpoint Management

```bash
# Resume training from checkpoint
./gdguess.py train --data-dir /path/to/data --checkpoint path/to/checkpoint.pt

# Save checkpoints more frequently
./gdguess.py train --data-dir /path/to/data --checkpoint-interval 50
```

### Early Stopping

```bash
# Train with early stopping (recommended)
./gdguess.py train --data-dir /path/to/data --early-stopping --patience 5
```

### Mixed Precision Training

```bash
# Use FP16 for faster training on supported devices
./gdguess.py train --data-dir /path/to/data --fp16
```

## Complete Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data-dir` | Directory with preprocessed data | (required) |
| `--batch-size` | Batch size | Auto-selected |
| `--learning-rate` or `--lr` | Learning rate | 0.0005 |
| `--steps` | Training steps | 10000 |
| `--epochs` | Training epochs | Unlimited |
| `--max-epochs` | Maximum epochs | 100 |
| `--device` | Device to use | Auto-detected |
| `--checkpoint` | Checkpoint to resume from | None |
| `--checkpoint-interval` | Steps between checkpoints | 100 |
| `--validation-interval` | Steps between validation | 100 |
| `--early-stopping` | Enable early stopping | False |
| `--patience` | Early stopping patience | 5 |
| `--fp16` | Use half precision | False |
| `--aggressive-memory` | Aggressive memory management | False |
| `--tensorboard` | Enable TensorBoard | False |
| `--tensorboard-dir` | TensorBoard directory | ./runs |
| `--output-dir` | Output directory | ./output |
| `--max-samples` | Limit dataset size | All samples |
| `--debug` | Enable debug mode | False |
""")
print("Created simplified TRAINING.md")

# Create simplified UPGRADE_GUIDE.md
upgrade_path = os.path.join(project_root, "UPGRADE_GUIDE.md")
with open(upgrade_path, "w") as f:
    f.write("""# Upgrade Guide

This document outlines the changes made to simplify the project structure and how to migrate existing code.

## Overview of Changes

The project structure has been simplified to:

1. Remove multiple implementations and maintain a single, robust implementation
2. Consolidate entry points to just `gdguess.py`
3. Use a single training script (`unified_training.py`) that handles all device types
4. Maintain comprehensive documentation with README files in each directory

## Migration Guide

### Command Line Usage

Continue using `gdguess.py` as before:

```bash
# Training
./gdguess.py train --data-dir /path/to/data --batch-size 16 --tensorboard

# Inference
./gdguess.py infer --model /path/to/model --input /path/to/audio
```

### Script Changes

If you were using any of the following scripts, you should migrate to the main implementation:

- `gdguess_core.py` → Use `gdguess.py` instead
- `minimal_core.py` → Use `gdguess.py` instead
- `train_simple.py` → Use `gdguess.py train` instead

### Code Structure

The `src/core/` directory has been removed. If you had custom code in this directory, you should:

1. Adapt your code to work with the main implementation in `src/models/dead_model.py`
2. Update imports to use the main implementation rather than the core implementation

## Future Development

All new features and improvements should be added to the main implementation:

1. Model enhancements → `src/models/dead_model.py`
2. Training improvements → `unified_training.py`
3. Data processing → `src/data/dataset.py`

## Benefits of Simplified Structure

1. **Easier Maintenance**: A single codebase is easier to maintain and enhance
2. **Clearer Organization**: Developers can quickly understand the project structure
3. **Better Performance**: The unified implementation includes optimizations for all devices
4. **Simplified Documentation**: Documentation is focused on a single implementation
""")
print("Created simplified UPGRADE_GUIDE.md")

print("\nProject cleanup complete. The following files have been moved to the backup directory:")
for file_path in FILES_TO_REMOVE:
    print(f"- {file_path}")
print("\nThe project now uses a single, unified implementation with gdguess.py as the entry point.")
print("To test the cleanup, try running: ./gdguess.py sysinfo --test-device")