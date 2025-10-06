#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_divide(
    a_ptr,          # Pointer to first input tensor
    b_ptr,          # Pointer to second input tensor  
    output_ptr,     # Pointer to output tensor
    n_elements,     # Number of elements
    BLOCK_SIZE: tl.constexpr, # Block size (must be power of 2)
):
    """
    Elementwise division kernel: output = a / b
    """
    # Get the program ID, which determines the block of data this instance processes
    pid = tl.program_id(axis=0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate a vector of offsets for the elements this block will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load the data from global memory
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division
    output = a / b
    
    # Store the result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)

# Variants with different block sizes
VARIANTS = [
    {'BLOCK_SIZE': 128},
    {'BLOCK_SIZE': 256},
    {'BLOCK_SIZE': 512},
    {'BLOCK_SIZE': 1024},
]