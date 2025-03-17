"""
Utility functions for the Grateful Dead show dating project.
"""

from .helpers import debug_shape, reset_parameters, cleanup_file_handles, ensure_device_consistency
from .visualization import (
    log_era_confusion_matrix,
    log_error_by_era,
    log_prediction_samples,
)
from .device_utils import (
    print_system_info,
    test_basic_operations,
    run_benchmark,
    get_cpu_info,
    get_gpu_info,
)
from .h200_optimizations import (
    optimize_h200_memory,
    create_cuda_streams,
    get_optimal_batch_size,
)
from .inspection import (
    inspect_pt_file,
    print_pt_file_info,
    inspect_directory,
)

__all__ = [
    # Helpers
    "reset_parameters",
    "debug_shape",
    "cleanup_file_handles",
    "ensure_device_consistency",
    
    # Visualization
    "log_prediction_samples",
    "log_era_confusion_matrix",
    "log_error_by_era",
    
    # Device management
    "print_system_info",
    "test_basic_operations",
    "run_benchmark",
    "get_cpu_info",
    "get_gpu_info",
    
    # H200 optimizations
    "optimize_h200_memory",
    "create_cuda_streams",
    "get_optimal_batch_size",
    
    # Inspection tools
    "inspect_pt_file",
    "print_pt_file_info",
    "inspect_directory",
]
