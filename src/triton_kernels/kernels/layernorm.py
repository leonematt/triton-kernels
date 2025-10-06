#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    weight_ptr,     # Pointer to weight tensor (optional scaling)
    n_elements,     # Number of elements per row
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
    EPS: tl.constexpr,  # Small constant for numerical stability (changed to uppercase)
):
    """
    RMS (Root Mean Square) normalization kernel.
    
    RMS norm is computed as:
    output = x / sqrt(mean(x^2) + eps) * weight
    
    where mean(x^2) is computed over the last dimension.
    """
    # Get the program ID for the current row
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * n_elements
    
    # Generate offsets for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
    
    # Load the input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute x^2
    x_squared = x * x
    
    # Compute mean of x^2 across the row
    mean_x_squared = tl.sum(x_squared, axis=0) / n_elements
    
    # Compute RMS normalization denominator: sqrt(mean(x^2) + eps)
    rms = tl.sqrt(mean_x_squared + EPS)  # Changed to EPS
    
    # Normalize: x / rms
    normalized = x / rms
    
    # Apply weight scaling if weight is provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
        normalized = normalized * weight
    
    # Store the result
    tl.store(output_ptr + offsets, normalized, mask=mask)

# Variants with different block sizes and configurations
VARIANTS = [
    {'BLOCK_SIZE': 1024, 'EPS': 1e-6},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-6},
    {'BLOCK_SIZE': 4096, 'EPS': 1e-6},
    {'BLOCK_SIZE': 8192, 'EPS': 1e-6},
    {'BLOCK_SIZE': 1024, 'EPS': 1e-5},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-5},
]