"""
entropy_gpu.py

GPU-accelerated entropy calculations using PyTorch.

This module provides CUDA-accelerated versions of entropy computations
for Wordle guess optimization. It uses PyTorch for GPU array operations.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def check_gpu_available():
    """Check if GPU and PyTorch CUDA are available."""
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is not installed. Install with: pip install torch"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU acceleration requires NVIDIA GPU with CUDA support.")

    try:
        # Test GPU access
        _ = torch.tensor([1, 2, 3], device='cuda')
    except Exception as e:
        raise RuntimeError(f"GPU not accessible: {e}") from e


def get_gpu_info():
    """Get GPU device information."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    mem_allocated = torch.cuda.memory_allocated(device)
    mem_total = torch.cuda.get_device_properties(device).total_memory

    return {
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'memory_total_gb': mem_total / 1e9,
        'memory_free_gb': (mem_total - mem_allocated) / 1e9,
    }


def entropy_from_counts_gpu(counts):
    """
    Compute Shannon entropy from bucket counts on GPU.

    Args:
        counts: PyTorch tensor of counts on GPU

    Returns:
        Entropy value (float)
    """
    total = torch.sum(counts)
    mask = counts > 0
    probs = counts[mask].float() / total.float()
    entropy = -torch.sum(probs * torch.log2(probs))
    return float(entropy.item())


def two_guess_entropy_gpu(row1_scaled, row2):
    """
    Joint entropy of two guesses on GPU.

    Args:
        row1_scaled: PyTorch tensor, first pattern row scaled by 243
        row2: PyTorch tensor, second pattern row

    Returns:
        Joint entropy H(guess1, guess2)
    """
    joint = row1_scaled + row2
    unique_vals, counts = torch.unique(joint, return_counts=True)
    return entropy_from_counts_gpu(counts)


def batch_two_guess_entropy_gpu(row1_scaled_batch, matrix, j_indices):
    """
    Compute entropy for a batch of pairs in parallel.

    Args:
        row1_scaled_batch: PyTorch tensor of scaled first rows, shape (batch_size, n_answers)
        matrix: Full pattern matrix on GPU, shape (n_allowed, n_answers)
        j_indices: PyTorch tensor of second word indices, shape (batch_size,)

    Returns:
        PyTorch tensor of entropies, shape (batch_size,)
    """
    batch_size = len(j_indices)

    # Fetch second rows
    row2_batch = matrix[j_indices]  # Shape: (batch_size, n_answers)

    # Compute joint patterns
    joint_batch = row1_scaled_batch + row2_batch  # Shape: (batch_size, n_answers)

    # Compute entropy for each pair
    entropies = torch.zeros(batch_size, dtype=torch.float32, device='cuda')

    for idx in range(batch_size):
        unique_vals, counts = torch.unique(joint_batch[idx], return_counts=True)
        entropies[idx] = entropy_from_counts_gpu(counts)

    return entropies


def decode_pair_from_thread_id(thread_id, n):
    """
    Decode linear thread ID to (i, j) pair in upper triangle.

    For n items, there are n*(n-1)/2 unique pairs where i < j.
    This function maps thread_id ∈ [0, n*(n-1)/2) to the corresponding (i, j).

    Args:
        thread_id: Linear thread index
        n: Total number of items

    Returns:
        (i, j) where 0 <= i < j < n
    """
    # Find which "row" of the triangle we're in
    i = 0
    cumsum = 0
    while cumsum + (n - i - 1) <= thread_id:
        cumsum += (n - i - 1)
        i += 1

    # Offset within row i
    offset = thread_id - cumsum
    j = i + 1 + offset

    return i, j


def decode_pair_from_thread_id_vectorized(thread_ids, n):
    """
    Vectorized version of pair decoding for PyTorch tensors.

    Args:
        thread_ids: PyTorch tensor of thread IDs
        n: Total number of items

    Returns:
        (i_tensor, j_tensor) - PyTorch tensors of decoded indices
    """
    # Solve for i using quadratic formula
    # i*(2*n - i - 1)/2 = thread_id
    # i = (2*n - 1 - sqrt((2*n-1)^2 - 8*thread_id)) / 2

    discriminant = (2*n - 1)**2 - 8*thread_ids
    i = ((2*n - 1) - torch.sqrt(discriminant.float())) / 2
    i = torch.floor(i).to(torch.int32)

    # Compute j from i and thread_id
    base = i * n - i * (i + 1) // 2
    j = thread_ids - base + i + 1

    return i, j
