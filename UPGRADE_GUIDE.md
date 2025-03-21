# Upgrade Guide

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

### Best Practices for Development

1. **Start with Documentation**: Read the README in the directory you're working on

2. **Follow the Component Pattern**:
   - Place reusable components in `src/models/components/`
   - Keep device-specific code in utility functions
   - Use type hints consistently

3. **Testing Modified Code**:
   - Run tests with `./gdguess.py test --all`
   - Use the test_setup.py script to verify basic functionality

## Common Upgrade Issues

### Missing Imports

If you encounter import errors:

```python
# Old structure - core implementation
from src.core.model.model import CoreModel

# New structure - use main implementation
from src.models.dead_model import DeadShowDatingModel
```

### Script Path Changes

If scripts fail to locate files:

```python
# Update paths in your custom scripts
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# To the more robust:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(project_root, "data")
```

## Getting Help

If you encounter issues with the new structure:

1. Run the test_setup.py script to verify basic functionality
2. Check the appropriate README file for the component you're working with
3. Refer to `ARCHITECTURE.md` for the overall design

## Benefits of Simplified Structure

1. **Easier Maintenance**: A single codebase is easier to maintain and enhance
2. **Clearer Organization**: Developers can quickly understand the project structure
3. **Better Performance**: The unified implementation includes optimizations for all devices
4. **Simplified Documentation**: Documentation is focused on a single implementation