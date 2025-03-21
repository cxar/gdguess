# Model Components

This directory contains reusable components for the Grateful Dead show dating model architecture.

## Directory Structure

- **`attention.py`**: Attention mechanism implementations
- **`audio_processing.py`**: Audio processing components
- **`blocks.py`**: Standard neural network building blocks

## Usage

These components are used in the main model architecture:

```python
from models.components.blocks import ConvBlock
from models.components.attention import SelfAttention

# Create a convolutional block
conv_block = ConvBlock(in_channels=64, out_channels=128)

# Use attention mechanism
attention = SelfAttention(dim=128)
```

## Design Principles

- Components should be reusable across different model architectures
- Each component should handle a specific transformation
- Components should be well-tested and robust to input variations