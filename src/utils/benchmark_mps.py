#!/usr/bin/env python3
"""
Benchmark script to compare CPU vs MPS performance for model training.
This helps understand the potential speedup from Apple Silicon GPU acceleration.
"""

import torch
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def benchmark_matrix_operations(sizes, device, num_runs=5):
    """Benchmark matrix operations with different sizes."""
    results = []
    
    for size in tqdm(sizes, desc=f"Matrix ops on {device}"):
        times = []
        for _ in range(num_runs):
            # Create random matrices
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Force sync before timing
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            # Time matrix multiplication
            start = time.time()
            c = torch.matmul(a, b)
            
            # Force sync after operation
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'cpu':
                # Force materialization to ensure operation completes
                _ = c.sum().item()
                
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Use median time for more stable results
        results.append(np.median(times))
    
    return results


def benchmark_conv_operations(image_sizes, device, num_runs=3):
    """Benchmark convolution operations with different input sizes."""
    results = []
    
    for size in tqdm(image_sizes, desc=f"Conv ops on {device}"):
        times = []
        for _ in range(num_runs):
            # Create random input and filter
            x = torch.randn(1, 3, size, size, device=device)
            w = torch.randn(16, 3, 3, 3, device=device)
            
            # Force sync before timing
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            # Time convolution
            start = time.time()
            y = torch.nn.functional.conv2d(x, w, padding=1)
            
            # Force sync after operation
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'cpu':
                # Force materialization
                _ = y.sum().item()
                
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Use median time
        results.append(np.median(times))
    
    return results


def benchmark_neural_network(batch_sizes, device, num_runs=3, num_epochs=5):
    """Benchmark neural network training with different batch sizes."""
    results = []
    input_size = 128  # Fixed input size
    
    # Define a model similar to our actual model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.fc1 = torch.nn.Linear(32 * 32 * 32, 256)
            self.fc2 = torch.nn.Linear(256, 10)
            self.dropout = torch.nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv1(x)))
            x = self.pool(torch.nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 32 * 32 * 32)
            x = torch.nn.functional.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    for batch_size in tqdm(batch_sizes, desc=f"NN training on {device}"):
        times = []
        for _ in range(num_runs):
            # Create model and optimizer
            model = SimpleModel().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Create random data
            inputs = torch.randn(batch_size, 3, 128, 128, device=device)
            targets = torch.randint(0, 10, (batch_size,), device=device)
            
            # Force sync before timing
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            # Time training loop
            start = time.time()
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Force sync at each epoch for accurate timing
                if device == 'mps':
                    torch.mps.synchronize()
                elif device == 'cuda':
                    torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Average time per epoch
        avg_time = np.median(times) / num_epochs
        results.append(avg_time)
    
    return results


def plot_results(sizes, cpu_results, mps_results, cuda_results=None, title="", xlabel="", ylabel="Time (s)", 
                log_scale=True, save_path=None):
    """Plot benchmark results comparing CPU vs MPS vs CUDA (if available)."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(sizes, cpu_results, 'o-', label='CPU', linewidth=2)
    plt.plot(sizes, mps_results, 's-', label='MPS (Apple Silicon)', linewidth=2)
    
    if cuda_results is not None:
        plt.plot(sizes, cuda_results, '^-', label='CUDA (NVIDIA)', linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def calculate_speedups(cpu_results, mps_results, cuda_results=None):
    """Calculate speedups of MPS/CUDA over CPU."""
    mps_speedups = [cpu / mps for cpu, mps in zip(cpu_results, mps_results)]
    
    if cuda_results:
        cuda_speedups = [cpu / cuda for cpu, cuda in zip(cpu_results, cuda_results)]
        mps_vs_cuda = [cuda / mps for cuda, mps in zip(cuda_results, mps_results)]
        return mps_speedups, cuda_speedups, mps_vs_cuda
    
    return mps_speedups


def plot_speedups(sizes, mps_speedups, cuda_speedups=None, title="", xlabel="", 
                 save_path=None):
    """Plot speedups of MPS/CUDA over CPU."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(sizes, mps_speedups, 's-', label='MPS vs CPU', linewidth=2)
    
    if cuda_speedups is not None:
        plt.plot(sizes, cuda_speedups, '^-', label='CUDA vs CPU', linewidth=2)
    
    # Add a horizontal line at y=1 (parity with CPU)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Speedup Factor (higher is better)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def run_matrix_benchmarks(matrix_sizes, devices):
    """Run matrix multiplication benchmarks."""
    results = {}
    
    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            continue
            
        results[device] = benchmark_matrix_operations(matrix_sizes, device)
    
    print("\nMatrix Multiplication Results:")
    for i, size in enumerate(matrix_sizes):
        print(f"  Size {size}x{size}:", end=" ")
        for device in results:
            print(f"{device}: {results[device][i]:.4f}s", end="  ")
        print()
    
    # Calculate and print speedups
    if 'cpu' in results and 'mps' in results:
        mps_speedups = [results['cpu'][i] / results['mps'][i] for i in range(len(matrix_sizes))]
        print("\nMPS speedup over CPU:")
        for i, size in enumerate(matrix_sizes):
            print(f"  Size {size}x{size}: {mps_speedups[i]:.2f}x")
    
    return results


def run_conv_benchmarks(image_sizes, devices):
    """Run convolution benchmarks."""
    results = {}
    
    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            continue
            
        results[device] = benchmark_conv_operations(image_sizes, device)
    
    print("\nConvolution Results:")
    for i, size in enumerate(image_sizes):
        print(f"  Size {size}x{size}:", end=" ")
        for device in results:
            print(f"{device}: {results[device][i]:.4f}s", end="  ")
        print()
    
    # Calculate and print speedups
    if 'cpu' in results and 'mps' in results:
        mps_speedups = [results['cpu'][i] / results['mps'][i] for i in range(len(image_sizes))]
        print("\nMPS speedup over CPU:")
        for i, size in enumerate(image_sizes):
            print(f"  Size {size}x{size}: {mps_speedups[i]:.2f}x")
    
    return results


def run_nn_benchmarks(batch_sizes, devices):
    """Run neural network training benchmarks."""
    results = {}
    
    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            continue
            
        results[device] = benchmark_neural_network(batch_sizes, device)
    
    print("\nNeural Network Training Results (time per epoch):")
    for i, batch in enumerate(batch_sizes):
        print(f"  Batch size {batch}:", end=" ")
        for device in results:
            print(f"{device}: {results[device][i]:.4f}s", end="  ")
        print()
    
    # Calculate and print speedups
    if 'cpu' in results and 'mps' in results:
        mps_speedups = [results['cpu'][i] / results['mps'][i] for i in range(len(batch_sizes))]
        print("\nMPS speedup over CPU:")
        for i, batch in enumerate(batch_sizes):
            print(f"  Batch size {batch}: {mps_speedups[i]:.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs MPS performance")
    parser.add_argument("--matrix", action="store_true", help="Run matrix multiplication benchmarks")
    parser.add_argument("--conv", action="store_true", help="Run convolution benchmarks")
    parser.add_argument("--nn", action="store_true", help="Run neural network training benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--save", action="store_true", help="Save plots to files")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save results")
    args = parser.parse_args()
    
    # Default to all benchmarks if none specified
    run_all = args.all or not (args.matrix or args.conv or args.nn)
    
    # Create output directory if saving results
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Detect available devices
    devices = ['cpu']
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
        print("Apple Silicon MPS is available")
    else:
        print("Apple Silicon MPS is not available")
    
    if torch.cuda.is_available():
        devices.append('cuda')
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available")
    
    print(f"Running benchmarks on: {', '.join(devices)}")
    
    # Matrix multiplication benchmark
    if args.matrix or run_all:
        print("\n=== Matrix Multiplication Benchmark ===")
        matrix_sizes = [512, 1024, 2048, 4096]
        matrix_results = run_matrix_benchmarks(matrix_sizes, devices)
        
        if args.save and len(matrix_results) > 1:
            cuda_results = matrix_results.get('cuda')
            plot_results(
                matrix_sizes, 
                matrix_results['cpu'], 
                matrix_results['mps'],
                cuda_results,
                title="Matrix Multiplication Performance",
                xlabel="Matrix Size",
                save_path=os.path.join(args.output_dir, "matrix_perf.png")
            )
            
            # Plot speedups
            if 'mps' in matrix_results:
                mps_speedups = [matrix_results['cpu'][i] / matrix_results['mps'][i] 
                               for i in range(len(matrix_sizes))]
                cuda_speedups = None
                if 'cuda' in matrix_results:
                    cuda_speedups = [matrix_results['cpu'][i] / matrix_results['cuda'][i] 
                                    for i in range(len(matrix_sizes))]
                
                plot_speedups(
                    matrix_sizes,
                    mps_speedups,
                    cuda_speedups,
                    title="Matrix Multiplication Speedup vs CPU",
                    xlabel="Matrix Size",
                    save_path=os.path.join(args.output_dir, "matrix_speedup.png")
                )
    
    # Convolution benchmark
    if args.conv or run_all:
        print("\n=== Convolution Benchmark ===")
        image_sizes = [128, 256, 512, 1024]
        conv_results = run_conv_benchmarks(image_sizes, devices)
        
        if args.save and len(conv_results) > 1:
            cuda_results = conv_results.get('cuda')
            plot_results(
                image_sizes, 
                conv_results['cpu'], 
                conv_results['mps'],
                cuda_results,
                title="Convolution Performance",
                xlabel="Image Size",
                save_path=os.path.join(args.output_dir, "conv_perf.png")
            )
            
            # Plot speedups
            if 'mps' in conv_results:
                mps_speedups = [conv_results['cpu'][i] / conv_results['mps'][i] 
                               for i in range(len(image_sizes))]
                cuda_speedups = None
                if 'cuda' in conv_results:
                    cuda_speedups = [conv_results['cpu'][i] / conv_results['cuda'][i] 
                                    for i in range(len(image_sizes))]
                
                plot_speedups(
                    image_sizes,
                    mps_speedups,
                    cuda_speedups,
                    title="Convolution Speedup vs CPU",
                    xlabel="Image Size",
                    save_path=os.path.join(args.output_dir, "conv_speedup.png")
                )
    
    # Neural network benchmark
    if args.nn or run_all:
        print("\n=== Neural Network Training Benchmark ===")
        batch_sizes = [16, 32, 64, 128]
        nn_results = run_nn_benchmarks(batch_sizes, devices)
        
        if args.save and len(nn_results) > 1:
            cuda_results = nn_results.get('cuda')
            plot_results(
                batch_sizes, 
                nn_results['cpu'], 
                nn_results['mps'],
                cuda_results,
                title="Neural Network Training Performance",
                xlabel="Batch Size",
                ylabel="Time per Epoch (s)",
                save_path=os.path.join(args.output_dir, "nn_perf.png")
            )
            
            # Plot speedups
            if 'mps' in nn_results:
                mps_speedups = [nn_results['cpu'][i] / nn_results['mps'][i] 
                               for i in range(len(batch_sizes))]
                cuda_speedups = None
                if 'cuda' in nn_results:
                    cuda_speedups = [nn_results['cpu'][i] / nn_results['cuda'][i] 
                                    for i in range(len(batch_sizes))]
                
                plot_speedups(
                    batch_sizes,
                    mps_speedups,
                    cuda_speedups,
                    title="Neural Network Training Speedup vs CPU",
                    xlabel="Batch Size",
                    save_path=os.path.join(args.output_dir, "nn_speedup.png")
                )
    
    print("\nBenchmarks complete! 🎉")
    if args.save:
        print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main() 