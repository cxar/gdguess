# Training the Grateful Dead Show Dating Model

This document provides a guide to training the Grateful Dead show dating model.

## Training the Model

To train the model, use the main CLI interface:

```bash
# Basic training with automatic settings
./gdguess.py train --data-dir /path/to/data --tensorboard

# Training with specific parameters
./gdguess.py train --data-dir /path/to/data --batch-size 16 --learning-rate 0.0005 --tensorboard
```

## Optimized GPU-Accelerated Training

The project now supports an optimized training pipeline specifically designed for GPU acceleration that delivers substantial performance improvements.

```bash
# Use the optimized GPU-accelerated training pipeline
./gdguess.py train --data-dir /path/to/data --optimized --device auto

# With specific batch size and learning rate
./gdguess.py train --data-dir /path/to/data --optimized --batch-size 32 --lr 0.0003 
```

The optimized training pipeline includes:
- Automatic device detection and configuration
- Memory-efficient tensor handling
- Gradient accumulation for effective batch size optimization
- Automatic Mixed Precision (AMP) for CUDA devices
- Optimized DataLoaders with prefetching
- Learning rate scheduler with warmup and cosine decay
- Enhanced TensorBoard logging
- Proper validation and evaluation

## Hardware-Specific Training

### CUDA (NVIDIA GPUs)

```bash
# Standard training on CUDA
./gdguess.py train --data-dir /path/to/data --batch-size 32 --device cuda --fp16

# Optimized training on CUDA (recommended)
./gdguess.py train --data-dir /path/to/data --optimized --batch-size 64 --device cuda
```

### MPS (Apple Silicon)

```bash
# Standard training for M1/M2/M3 regular models
./gdguess.py train --data-dir /path/to/data --batch-size 8 --device mps

# Standard training for M1/M2/M3 Pro/Max/Ultra models
./gdguess.py train --data-dir /path/to/data --batch-size 16 --device mps --aggressive-memory

# Optimized training for Apple Silicon (recommended)
./gdguess.py train --data-dir /path/to/data --optimized --batch-size 16 --device mps
```

### CPU

```bash
# Multi-core CPU training (standard)
./gdguess.py train --data-dir /path/to/data --batch-size 4 --device cpu

# Optimized CPU training
./gdguess.py train --data-dir /path/to/data --optimized --batch-size 8 --device cpu
```

## Monitoring Training

### TensorBoard

```bash
# Start training with TensorBoard
./gdguess.py train --data-dir /path/to/data --tensorboard --tensorboard-dir ./runs

# In another terminal
tensorboard --logdir=./runs
```

### Training Metrics

The training process displays the following metrics:

- Training loss
- Validation loss
- Learning rate schedule
- Prediction accuracy (in days)
- Training speed (samples/second)

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

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size: `--batch-size 2`
   - Enable aggressive memory management: `--aggressive-memory`
   - Use CPU instead: `--device cpu`

2. **Data Loading Errors**
   - Verify dataset format:
     ```bash
     python -c "import torch; data = torch.load('/path/to/data/file.pt'); print(data.keys())"
     ```
   - Try the minimal implementation which is robust to format variations:
     ```bash
     ./minimal_core.py --data-dir /path/to/data
     ```

3. **Model Errors**
   - Check the model's device compatibility:
     ```bash
     ./gdguess.py sysinfo --test-device
     ```
   - Try the simplified implementation:
     ```bash
     ./gdguess_core.py --data-dir /path/to/data
     ```

## Complete Parameter Reference

| Parameter | Description | Default | Optimized Mode Support |
|-----------|-------------|---------|------------------------|
| `--data-dir` | Directory with preprocessed data | (required) | ✅ |
| `--batch-size` | Batch size | Auto-selected | ✅ |
| `--learning-rate` or `--lr` | Learning rate | 0.0005 | ✅ (use `--lr`) |
| `--steps` | Training steps | Unlimited | ✅ |
| `--epochs` | Training epochs | Unlimited | ✅ |
| `--max-epochs` | Maximum epochs | 100 | ✅ (as `--epochs`) |
| `--device` | Device to use | Auto-detected | ✅ |
| `--checkpoint` | Checkpoint to resume from | None | ✅ |
| `--checkpoint-interval` | Steps between checkpoints | 100 | ✅ (as `--save-frequency`) |
| `--validation-interval` | Steps between validation | 1 epoch | ✅ |
| `--early-stopping` | Enable early stopping | False | ✅ (enabled by default) |
| `--patience` | Early stopping patience | 3 | ✅ |
| `--fp16` | Use half precision | False | ✅ (enabled by default) |
| `--aggressive-memory` | Aggressive memory management | False | ✅ (auto-handled) |
| `--tensorboard` | Enable TensorBoard | False | ✅ (enabled by default) |
| `--tensorboard-dir` | TensorBoard directory | ./runs | ✅ |
| `--output-dir` | Output directory | ./output | ✅ |
| `--max-samples` | Limit dataset size | All samples | ✅ |
| `--debug` | Enable debug mode | False | ✅ |
| `--optimized` | Use optimized training pipeline | False | - |
| `--num-workers` | DataLoader workers | Auto-selected | ✅ |