#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    weight_ptr,     # Pointer to weight tensor (optional scaling)
    n_elements,     # Number of elements per row
    eps: tl.constexpr,  # Small constant for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
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
    rms = tl.sqrt(mean_x_squared + eps)
    
    # Normalize: x / rms
    normalized = x / rms
    
    # Apply weight scaling if weight is provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
        normalized = normalized * weight
    
    # Store the result
    tl.store(output_ptr + offsets, normalized, mask=mask)

# Test variants with different block sizes and configurations
variants = [
    {'BLOCK_SIZE': 1024, 'EPS': 1e-6},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-6},
    {'BLOCK_SIZE': 4096, 'EPS': 1e-6},
    {'BLOCK_SIZE': 8192, 'EPS': 1e-6},
    {'BLOCK_SIZE': 1024, 'EPS': 1e-5},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-5},
]

def run():
    """Test function to run RMS norm kernel variants."""
    batch_size = 32
    hidden_size = 2048
    
    print(f"--- Running RMS Norm Kernel Variants ---")
    print(f"Batch size: {batch_size}, Hidden size: {hidden_size}")

    for variant in variants:
        block_size = variant['BLOCK_SIZE']
        eps = variant['EPS']
        
        print(f"Testing BLOCK_SIZE={block_size:4}, eps={eps:.0e}... ", end="")

        try:
            # Create test data
            x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float32)
            weight = torch.randn(hidden_size, device='cuda', dtype=torch.float32)
            output = torch.empty_like(x)
            
            # Ensure block size can handle the hidden size
            if hidden_size > block_size:
                print(f"SKIPPED (hidden_size {hidden_size} > BLOCK_SIZE {block_size})")
                continue
            
            # Launch kernel
            grid = (batch_size,)  # One block per row
            
            rms_norm_kernel[grid](
                x_ptr=x,
                output_ptr=output,
                weight_ptr=weight,
                n_elements=hidden_size,
                eps=eps,
                BLOCK_SIZE=block_size
            )
            
            # Verification using PyTorch
            # RMS norm: x / sqrt(mean(x^2) + eps) * weight
            x_squared = x * x
            mean_x_squared = torch.mean(x_squared, dim=-1, keepdim=True)
            rms = torch.sqrt(mean_x_squared + eps)
            expected = (x / rms) * weight
            
            # Check if results match
            is_correct = torch.allclose(output, expected, rtol=1e-4, atol=1e-6)
            status = "PASSED" if is_correct else "FAILED"
            print(status)
            
            if not is_correct:
                max_diff = torch.max(torch.abs(output - expected)).item()
                print(f"      Max difference: {max_diff:.6e}")
            
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n--- All variants tested. ---")

if __name__ == "__main__":
    run()