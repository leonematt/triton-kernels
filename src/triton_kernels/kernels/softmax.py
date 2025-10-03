#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
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
    """
    # Get the row index this program will process
    row_idx = tl.program_id(axis=0)
    
    # Compute starting pointers for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load input data with masking for out-of-bounds
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Subtract max for numerical stability (max trick)
    row_minus_max = row - tl.max(row, axis=0)
    
    # Compute exponentials
    numerator = tl.exp(row_minus_max)
    
    # Sum exponentials
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Write back result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


# Test variants with different block sizes
variants = [
    {'BLOCK_SIZE': 128},
    {'BLOCK_SIZE': 256},
    {'BLOCK_SIZE': 512},
    {'BLOCK_SIZE': 1024},
]

def run():
    n_rows = 128
    n_cols = 256
    
    print(f"--- Running Softmax Kernel Variants (Shape: {n_rows}x{n_cols}) ---")
    
    # Loop through and test each variant
    for variant in variants:
        block_size = variant['BLOCK_SIZE']
        
        print(f"Testing BLOCK_SIZE={block_size:4}... ", end="")
        
        try:
            # Data and Kernel Execution
            input_tensor = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
            output = torch.empty_like(input_tensor)
            
            grid = (n_rows,)
            
            softmax_kernel[grid](
                input_ptr=input_tensor,
                output_ptr=output,
                n_rows=n_rows,
                n_cols=n_cols,
                input_row_stride=input_tensor.stride(0),
                output_row_stride=output.stride(0),
                BLOCK_SIZE=block_size
            )
            
            # Verification
            expected_output = torch.softmax(input_tensor, dim=-1)
            is_correct = torch.allclose(output, expected_output, atol=1e-5)
            status = "PASSED" if is_correct else "FAILED"
            print(status)
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n--- All variants tested. ---")

if __name__ == "__main__":
    run()