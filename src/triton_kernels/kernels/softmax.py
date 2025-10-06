#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def softmax(
    input_ptr,      # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    n_rows,         # Number of rows
    n_cols,         # Number of columns
    input_row_stride,  # Stride between rows in input
    output_row_stride, # Stride between rows in output
    BLOCK_SIZE: tl.constexpr, # Block size (must be power of 2)
):
    """
    Softmax kernel that computes softmax along the last dimension.
    Each program instance processes one row.
    Handles arbitrary column sizes by processing in tiles.
    """
    # Get the row index this program will process
    row_idx = tl.program_id(axis=0)
    
    # Compute starting pointers for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # First pass: find the maximum value across the entire row
    max_val = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(row_block, axis=0))
    
    # Second pass: compute sum of exponentials
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        row_block_minus_max = row_block - max_val
        # Set values that were masked to 0 in exp
        row_block_minus_max = tl.where(mask, row_block_minus_max, float('-inf'))
        exp_vals = tl.exp(row_block_minus_max)
        sum_exp += tl.sum(exp_vals, axis=0)
    
    # Third pass: compute softmax output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        row_block_minus_max = row_block - max_val
        row_block_minus_max = tl.where(mask, row_block_minus_max, float('-inf'))
        exp_vals = tl.exp(row_block_minus_max)
        softmax_output = exp_vals / sum_exp
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


# Variants with different block sizes
VARIANTS = [
    {'BLOCK_SIZE': 128},
    {'BLOCK_SIZE': 256},
    {'BLOCK_SIZE': 512},
    {'BLOCK_SIZE': 1024},
]