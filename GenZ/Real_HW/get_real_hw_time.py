
import warnings
import os
import subprocess
import yaml
from GenZ.system import System
from GenZ.unit import Unit
from GenZ.operator_base import Operator
import numpy as np
import contextlib
import io
import re
import pandas as pd
import csv

import os
import statistics
import time
from typing import List, Tuple
import gc
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import torch


def get_max_gpu_size():
    """Get the maximum tensor dimensions that can fit in GPU memory."""

    # Get GPU properties
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory

        # Calculate max square matrix size for FP16 (2 bytes per element)
        # Assuming we need 3 matrices (A, B, C) for GEMM
        bytes_per_element = 2  # FP16
        max_elements = total_memory / (3 * bytes_per_element)
        max_square_dim = int(
            (max_elements**0.5) * 0.8
        )  # Using 80% of memory to be safe

        return {
            "gpu_name": gpu_properties.name,
            "total_memory_gb": total_memory / (1024**3),
            "max_square_matrix_dim": max_square_dim,
            "max_batch_size": 32,  # Default reasonable batch size
        }
    else:
        return {
            "gpu_name": "No GPU available",
            "total_memory_gb": 0,
            "max_square_matrix_dim": 0,
            "max_batch_size": 0,
        }


def run_einsum(einsum_str, tensor_a_shape, tensor_b_shape, num_repeats: int) -> float:
    """
    Run a Einsum operation on GPU tensor cores and measure the latency.

    Args:
        einsum_str: Einsum string
        tensor_a_shape: Shape of tensor A
        tensor_b_shape: Shape of tensor B
        num_repeats: Number of repeats

    Returns:
        float: Latency in milliseconds

    Raises:
        RuntimeError: If PyTorch is not installed or CUDA is not available
    """
    # Import PyTorch here to avoid issues with static analysis
    # Note: The linter may still show an error for this import, but the code will
    # handle the case where PyTorch is not installed gracefully at runtime
    try:
        import torch  # type: ignore
    except ImportError:
        raise RuntimeError(
            "PyTorch is not installed. Please install PyTorch with CUDA support."
        )

    # Check for CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")

    # Create input tensors on GPU
    # For tensor cores, we use float16 data type
    a = torch.randn(tensor_a_shape, dtype=torch.float16, device="cuda")
    b = torch.randn(tensor_b_shape, dtype=torch.float16, device="cuda")
    # print(a.shape, b.shape)
    # Warm-up run to avoid initial overhead
    _ = torch.einsum(einsum_str, a, b)

    # Synchronize before timing
    torch.cuda.synchronize()

    latencies = []
    for i in range(num_repeats):
        # Measure execution time
        start_time = time.perf_counter()
        # We need to keep the result to ensure the operation is actually performed
        # but we use _ to indicate it's not used elsewhere
        _ = torch.einsum(einsum_str, a, b)
        torch.cuda.synchronize()  # Wait for the operation to complete
        end_time = time.perf_counter()
        del _

        # Convert to milliseconds
        latencies.append((end_time - start_time) * 1000)

    latency_ms = statistics.median(latencies)
    # Delete tensors to free up GPU memory
    del a, b
    torch.cuda.empty_cache()
    gc.collect()

    return latency_ms





def get_real_hw_time(op:Operator,  system: System):
    # Get the time taken by Scale-sim to simulate the system
    # system: System object
    # returns: time taken by Scale-sim to simulate the system
    # Note: This function is a wrapper around the Scale-sim simulator
    # and requires the Scale-sim simulator to be installed on the system
    op_type = op.get_op_type(op.dim)
    if op_type == 'GEMM':
        dim = op.get_dimensions()
        # print(dim)

        B, K, M = dim[0][0]
        W1, W2 = dim[0][1]
        assert K == W2
        return run_einsum('bik,kj->bij', (B, M, K), (W2, W1), 5 ) / 1000
    elif op_type == 'Logit':
        dim = op.get_dimensions()
        # print(dim)
        return run_einsum('bhmd,bknd->bhmn', dim[0], dim[1], 5 ) / 1000

    elif op_type == 'Attend':
        dim = op.get_dimensions()
        return run_einsum('bhmn,bknd->bhmd', dim[0], dim[1], 5 ) / 1000


    
    return op.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec