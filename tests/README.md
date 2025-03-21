# Tests for Grateful Dead Show Dating Project

This module contains all test suites for the Grateful Dead show dating project.

## Test Categories

The tests focus on various aspects of the system:

- **Model Structure Tests**: Validate the model architecture 
- **Transformer Tests**: Specifically test the transformer encoder component
- **Training Tests**: Validate the training process
- **Real Data Tests**: Test the model with real audio files from the `audsnippets-all` collection
- **Synthetic Data Tests**: Test with programmatically generated data

## Directory Structure

- **`tests/`**: Main tests module
  - **`run_all_tests.py`**: Advanced test runner with real data options
  - **`run_tests.py`**: Legacy test runner script
  - **`run_transformer_tests.py`**: Transformer-specific test runner
  - **`real_data_test.py`**: Tests with real audio data
  - **`test_transformer_with_real_data.py`**: Tests transformer with real audio
  - **`test_transformer.py`**: Transformer model tests
  - **`test_transformer_detailed.py`**: Detailed transformer tests
  - **`test_transformer_simple.py`**: Simplified transformer tests

## New Real Data Tests

The newest additions to the test suite focus on using real audio data to ensure the system works properly with actual input. These tests use audio snippets from the `data/audsnippets-all` directory.

Key files:
- `real_data_test.py`: Tests the full model with multiple real audio files
- `test_transformer_with_real_data.py`: Specifically tests the transformer with real audio input

## Running Tests

### Using the CLI

The simplest way to run tests is using the main CLI:

```bash
# Run all tests
./gdguess.py test --all

# Run only transformer tests
./gdguess.py test --transformer

# Run only model tests
./gdguess.py test --model

# Test with real data only (no synthetic data)
./gdguess.py test --real-data

# Run a quick test suite (faster, less thorough)
./gdguess.py test --quick

# Run only synthetic data tests (no real data)
./gdguess.py test --synthetic
```

### Run Scripts Directly

For more control, you can run test scripts directly:

```bash
# Run all tests
python tests/run_all_tests.py

# Run only real data tests
python tests/run_all_tests.py --real-data

# Run only transformer tests
python tests/run_all_tests.py --transformer

# Run a specific test
python tests/real_data_test.py
python tests/test_transformer_with_real_data.py
```

## Data Sources

Tests use two types of data:

1. **Synthetic Data**: Generated programmatically for component-level testing
2. **Real Data**: Audio files from the `data/audsnippets-all` directory

Real data tests rely on the existence of the `data/audsnippets-all` directory which should contain MP3 files of Grateful Dead concert recordings. The file structure should include year-based subdirectories.