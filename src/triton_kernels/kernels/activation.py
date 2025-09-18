#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
import os


@triton.jit
def silu_and_mul_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SiLU and multiply kernel.
    Computes: output = silu(x[:half]) * x[half:]
    where silu(x) = x * sigmoid(x)
    
    Assumes: n_elements is the full length (half for gate, half for up)
            and (n_elements // 2) <= BLOCK_SIZE
    """
    row_idx = tl.program_id(0)
    half_elements = n_elements // 2
    row_start = row_idx * n_elements
    
    # Generate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_elements
    
    # Load gate (first half) and up (second half)
    gate_offsets = row_start + offsets
    up_offsets = row_start + half_elements + offsets
    
    gate = tl.load(x_ptr + gate_offsets, mask=mask, other=0.0)
    up = tl.load(x_ptr + up_offsets, mask=mask, other=0.0)
    
    # Convert to float32 for computation
    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)
    
    # Compute SiLU: x * sigmoid(x)
    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate
    
    # Multiply with up projection
    output = silu_gate * up_f32
    
    # Convert back to original dtype
    output_result = output.to(gate.dtype)
    
    # Store result (output is half the size of input)
    output_offsets = row_idx * half_elements + offsets
    tl.store(output_ptr + output_offsets, output_result, mask=mask)


# Variants with different block sizes
VARIANTS = [
    {'BLOCK_SIZE': 1024},
    {'BLOCK_SIZE': 2048},
    {'BLOCK_SIZE': 4096},
    {'BLOCK_SIZE': 8192},
    {'BLOCK_SIZE': 16384},
]