# Apple Silicon GPU Acceleration Guide

This guide explains how to use Metal Performance Shaders (MPS) to accelerate training on Apple Silicon Macs (M1/M2/M3 series).

## Requirements

- Apple Silicon Mac (M1/M2/M3 series)
- macOS 12.3+ (Monterey or newer)
- PyTorch 1.13+ (PyTorch 2.0+ recommended for best performance)

## Quick Start

To use Apple Silicon GPU acceleration, simply add the `--device mps` flag when running the training script:

```bash
python src/main.py --device mps
```

For best compatibility with operations not fully supported by MPS, add the `--mps-fallback` flag:

```bash
python src/main.py --device mps --mps-fallback
```

## Checking MPS Support

You can check if your system supports MPS and run performance tests with the included utilities:

```bash
# Check device compatibility and details
python src/utils/device_info.py

# Run MPS functionality tests
python src/utils/check_mps.py

# Run performance benchmarks
python src/utils/benchmark_mps.py --save
```

## Performance Expectations

On Apple Silicon Macs, MPS acceleration typically provides:

- 2-5x speedup for matrix operations
- 3-8x speedup for convolution operations
- 2-4x speedup for overall neural network training

Actual performance will vary depending on your specific Mac model and the model architecture.

## Troubleshooting

### MPS Not Available

If MPS is not available, check the following:

1. Make sure you're using PyTorch 1.13 or later
2. Ensure you're on macOS 12.3 or later
3. Verify you're using an Apple Silicon Mac

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce batch size with the `--batch-size` flag
2. Enable memory-efficient operations with `--mps-fallback`

### Operation Not Implemented Errors

Some operations may not be fully supported in the MPS backend. If you encounter `"Operation not implemented"` errors:

1. Add the `--mps-fallback` flag to enable CPU fallback for unsupported operations
2. Alternatively, use `--device cpu` to run on CPU only

### Slow Performance

If MPS performance is slower than expected:

1. Make sure no other GPU-intensive applications are running
2. Close apps that might be using the GPU (browsers, video players, etc.)
3. Try the benchmark utility to see if performance is as expected

## Command Line Options

The training script supports the following MPS-related options:

- `--device mps`: Use Apple Silicon GPU acceleration
- `--mps-fallback`: Enable CPU fallback for unsupported operations
- `--mps-optimize`: Enable additional MPS-specific optimizations
- `--force-cpu`: Force CPU usage even on Apple Silicon Mac
- `--device auto`: Automatically choose the best available device (default)

## Advanced Configuration

### Environment Variables

You can set these environment variables for additional control:

```bash
# Disable Apple Silicon GPU acceleration entirely
export PYTORCH_ENABLE_MPS_FALLBACK=0

# Enable verbose MPS backend logging
export PYTORCH_MPS_LOG_LEVEL=3
```

### Benchmark Results

After running benchmarks with `python src/utils/benchmark_mps.py --save`, you'll find performance comparison charts in the `benchmark_results` directory showing:

- Raw performance comparison between CPU and MPS
- Speedup factors for various operations
- Performance scaling with different input sizes

## Known Limitations

1. MPS does not support mixed precision training
2. Some operations may not be fully optimized
3. Memory management is less sophisticated than CUDA

For more information on PyTorch MPS support, see the [official documentation](https://pytorch.org/docs/stable/notes/mps.html). 