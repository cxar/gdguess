#!/usr/bin/env python3
"""
Simple test script to check if MPS (Metal Performance Shaders) is working correctly.
This can help diagnose issues with Apple Silicon GPU acceleration.
"""

import torch
import time
import numpy as np

def test_basic_operations():
    """Test basic tensor operations on MPS device."""
    print("\n=== Testing Basic Operations ===")
    
    # Create tensors on CPU first
    a_cpu = torch.randn(1000, 1000)
    b_cpu = torch.randn(1000, 1000)
    
    # Time CPU operation
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"CPU matrix multiplication time: {cpu_time:.4f} seconds")
    
    # Check if MPS is available
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        print("MPS is not available on this system")
        return False
    
    # Try MPS operation
    try:
        # Move tensors to MPS
        device = torch.device("mps")
        a_mps = a_cpu.to(device)
        b_mps = b_cpu.to(device)
        
        # Time MPS operation
        start = time.time()
        c_mps = torch.matmul(a_mps, b_mps)
        # Force synchronization to get accurate timing
        c_mps_cpu = c_mps.to("cpu")
        mps_time = time.time() - start
        print(f"MPS matrix multiplication time: {mps_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / mps_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Check for correctness (allowing for small numerical differences)
        max_diff = torch.max(torch.abs(c_cpu - c_mps_cpu)).item()
        print(f"Maximum absolute difference: {max_diff:.6f}")
        
        success = speedup > 1.0 and max_diff < 1e-3
        print(f"Basic operations test {'PASSED' if success else 'FAILED'}")
        
        return success
    except Exception as e:
        print(f"Error during MPS test: {e}")
        return False


def test_neural_network():
    """Test a small neural network on MPS device."""
    print("\n=== Testing Neural Network ===")
    
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(100, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 10)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create random data
    batch_size = 64
    x = torch.randn(batch_size, 100)
    y = torch.randint(0, 10, (batch_size,))
    
    # CPU training
    model_cpu = SimpleNN()
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    start = time.time()
    # Run a few iterations
    for i in range(10):
        optimizer_cpu.zero_grad()
        output = model_cpu(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer_cpu.step()
    cpu_time = time.time() - start
    print(f"CPU training time (10 iterations): {cpu_time:.4f} seconds")
    
    # Check if MPS is available
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        print("MPS is not available on this system")
        return False
    
    try:
        # MPS training
        device = torch.device("mps")
        model_mps = SimpleNN().to(device)
        optimizer_mps = torch.optim.Adam(model_mps.parameters(), lr=0.001)
        x_mps = x.to(device)
        y_mps = y.to(device)
        
        start = time.time()
        # Run a few iterations
        for i in range(10):
            optimizer_mps.zero_grad()
            output = model_mps(x_mps)
            loss = loss_fn(output, y_mps)
            loss.backward()
            optimizer_mps.step()
        
        # Force synchronization
        torch.mps.synchronize()
        mps_time = time.time() - start
        print(f"MPS training time (10 iterations): {mps_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / mps_time
        print(f"Speedup: {speedup:.2f}x")
        
        success = speedup > 1.0
        print(f"Neural network test {'PASSED' if success else 'FAILED'}")
        
        return success
    except Exception as e:
        print(f"Error during MPS neural network test: {e}")
        return False


def test_operations_compatibility():
    """Test compatibility of different operations with MPS backend."""
    print("\n=== Testing Operations Compatibility ===")
    
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        print("MPS is not available on this system")
        return False
    
    device = torch.device("mps")
    
    operations = [
        ("Addition", lambda: torch.add(torch.randn(100, 100, device=device), 
                                      torch.randn(100, 100, device=device))),
        ("Subtraction", lambda: torch.sub(torch.randn(100, 100, device=device), 
                                         torch.randn(100, 100, device=device))),
        ("Multiplication", lambda: torch.mul(torch.randn(100, 100, device=device), 
                                           torch.randn(100, 100, device=device))),
        ("Division", lambda: torch.div(torch.randn(100, 100, device=device), 
                                      torch.randn(100, 100, device=device) + 0.1)),
        ("MatMul", lambda: torch.matmul(torch.randn(100, 100, device=device), 
                                       torch.randn(100, 100, device=device))),
        ("Exp", lambda: torch.exp(torch.randn(100, 100, device=device))),
        ("Log", lambda: torch.log(torch.abs(torch.randn(100, 100, device=device)) + 0.1)),
        ("Sigmoid", lambda: torch.sigmoid(torch.randn(100, 100, device=device))),
        ("Tanh", lambda: torch.tanh(torch.randn(100, 100, device=device))),
        ("ReLU", lambda: torch.nn.functional.relu(torch.randn(100, 100, device=device))),
        ("Softmax", lambda: torch.nn.functional.softmax(torch.randn(100, 100, device=device), dim=1)),
        ("Convolution", lambda: torch.nn.functional.conv2d(
            torch.randn(1, 3, 32, 32, device=device), 
            torch.randn(16, 3, 3, 3, device=device), 
            padding=1
        )),
        ("Max Pooling", lambda: torch.nn.functional.max_pool2d(
            torch.randn(1, 3, 32, 32, device=device), 
            kernel_size=2, 
            stride=2
        )),
        ("Batch Norm", lambda: torch.nn.functional.batch_norm(
            torch.randn(10, 10, device=device), 
            torch.randn(10, device=device), 
            torch.randn(10, device=device), 
            None, 
            None, 
            training=True, 
            momentum=0.1, 
            eps=1e-5
        )),
    ]
    
    results = []
    
    for name, op in operations:
        try:
            result = op()
            results.append((name, "SUCCESS", None))
            print(f"{name}: SUCCESS")
        except Exception as e:
            results.append((name, "FAILED", str(e)))
            print(f"{name}: FAILED - {e}")
    
    # Summarize results
    success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
    print(f"\nOperations compatibility: {success_count}/{len(operations)} passed")
    
    return success_count == len(operations)


def check_mps_version():
    """Check MPS version and compatibility information."""
    print("\n=== MPS Version Information ===")
    
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        print("MPS is not available on this system")
        return
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    try:
        # Create a small tensor on MPS to check if it's working
        device = torch.device("mps")
        test_tensor = torch.ones(1, device=device)
        print(f"MPS device test: SUCCESS")
        
        # Check fallback to CPU setting
        fallback = getattr(torch.backends.mps, "fallback_to_cpu", "Not available")
        print(f"MPS fallback to CPU: {fallback}")
        
        # Try to get macOS version
        import platform
        macos_version = platform.mac_ver()[0]
        print(f"macOS version: {macos_version}")
        
        # Display recommended PyTorch version for this macOS
        if macos_version:
            major, minor = map(int, macos_version.split('.')[:2])
            if major >= 13:  # macOS 13 (Ventura) or newer
                print("Recommendation: PyTorch 2.0+ recommended for best MPS support")
            elif major == 12:  # macOS 12 (Monterey)
                print("Recommendation: PyTorch 1.13+ recommended for basic MPS support")
            else:
                print("Warning: MPS requires macOS 12.3+ (Monterey or newer)")
    except Exception as e:
        print(f"Error checking MPS version: {e}")


def run_all_tests():
    """Run all MPS tests."""
    print("===================================")
    print(" MPS (Metal Performance Shaders) Test")
    print("===================================")
    
    # Check if MPS is available
    if not hasattr(torch.backends, "mps"):
        print("MPS is not supported in this PyTorch build")
        return False
    
    if not torch.backends.mps.is_available():
        print("MPS is not available on this system")
        print("Requirements:")
        print("1. PyTorch 1.13+ with MPS support")
        print("2. macOS 12.3+ (Monterey or newer)")
        print("3. Apple Silicon Mac (M1/M2/M3 series)")
        return False
    
    # Display version info
    check_mps_version()
    
    # Run tests
    basic_ops_result = test_basic_operations()
    nn_result = test_neural_network()
    compat_result = test_operations_compatibility()
    
    # Overall result
    print("\n=== Summary ===")
    print(f"Basic operations: {'PASSED' if basic_ops_result else 'FAILED'}")
    print(f"Neural network: {'PASSED' if nn_result else 'FAILED'}")
    print(f"Operations compatibility: {'PASSED' if compat_result else 'FAILED'}")
    
    overall = basic_ops_result and nn_result
    print(f"\nOverall MPS support: {'GOOD' if overall else 'PARTIAL'}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if overall:
        print("✅ MPS is working well on your system. You can use --device mps for accelerated training.")
        if not compat_result:
            print("⚠️ Some operations may not be supported. Consider using --mps-fallback for better compatibility.")
    else:
        print("⚠️ MPS support is limited on your system.")
        print("   Consider using --device cpu for better stability.")
        print("   Or try --device mps --mps-fallback to enable CPU fallback for unsupported operations.")
    
    return overall


if __name__ == "__main__":
    run_all_tests() 