"""
CPU Thread Limiter for PyTorch
Prevents high CPU usage during training

Add this at the beginning of your scripts:
    from cpu_limiter import set_cpu_limit
    set_cpu_limit(20)  # or any number you prefer
"""

import torch
import os


def set_cpu_limit(num_threads=20):
    """
    Limit CPU threads for PyTorch to prevent high CPU usage
    
    Args:
        num_threads: Number of threads to use (default: 20)
    """
    # Set PyTorch threading
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    
    # Set OpenMP threads (if available)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    
    print(f"CPU threads limited to: {num_threads}")
    print(f"PyTorch num_threads: {torch.get_num_threads()}")
    print(f"PyTorch num_interop_threads: {torch.get_num_interop_threads()}")


# Auto-set on import (optional)
# Uncomment the next line if you want automatic limiting when importing
# set_cpu_limit(20)