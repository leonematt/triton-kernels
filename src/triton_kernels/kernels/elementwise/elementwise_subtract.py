#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_subtract_kernel(
    a_ptr,          # Pointer to first input tensor
    b_ptr,          # Pointer to second input tensor  
    output_ptr,     # Pointer to output tensor
    n_elements,     # Number of elements
    BLOCK_SIZE: tl.constexpr, # Block size (must be power of 2)
):
    """
    Elementwise subtraction kernel: output = a - b
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
    
    # Perform subtraction
    output = a - b
    
    # Store the result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)

# Test variants with different block sizes
variants = [
    {'BLOCK_SIZE': 128},
    {'BLOCK_SIZE': 256},
    {'BLOCK_SIZE': 512},
    {'BLOCK_SIZE': 1024},
]
    
def run():
    tensor_size = 4096
    
    print(f"--- Running Subtraction Kernel Variants (Tensor Size: {tensor_size}) ---")

    # Loop through and test each variant
    for variant in variants:
        block_size = variant['BLOCK_SIZE']

        print(f"Testing BLOCK_SIZE={block_size:4}... ", end="")

        try:
            # Data and Kernel Execution
            a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
            b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
            output = torch.empty_like(a)
            
            grid = (triton.cdiv(tensor_size, block_size),)
            
            elementwise_subtract_kernel[grid](
                a_ptr=a,
                b_ptr=b,
                output_ptr=output,
                n_elements=tensor_size,
                BLOCK_SIZE=block_size
            )

            # Verification
            expected_output = a - b
            is_correct = torch.allclose(output, expected_output)
            status = "PASSED" if is_correct else "FAILED"
            print(status)

        except Exception as e:
            print(f"ERROR: {e}")

    print("\n--- All variants tested. ---")
    
if __name__ == "__main__":
    run()