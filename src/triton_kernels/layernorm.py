#!/usr/bin/env python3

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, n_rows, eps, 
                    BLOCK_SIZE: tl.constexpr, HIDDEN_SIZE: tl.constexpr):
    """
    Optimized single-pass RMSNorm kernel with improved numerical stability.
    
    Args:
        x_ptr: Input tensor pointer
        weight_ptr: Weight tensor pointer  
        out_ptr: Output tensor pointer
        n_rows: Number of rows to process
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Block size (must be >= HIDDEN_SIZE)
        HIDDEN_SIZE: Hidden dimension size
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Calculate row start offset
    row_start = row_idx * HIDDEN_SIZE
    
    # Load entire row at once (single-pass approach)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < HIDDEN_SIZE
    
    # Load input and weight in single operations
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    
    # Convert BOTH x and weight to fp32 for computation (critical for numerical stability)
    x_fp32 = x.to(tl.float32)
    weight_fp32 = weight.to(tl.float32)
    
    # Compute mean of squares
    mean_of_squares = tl.sum(x_fp32 * x_fp32) / HIDDEN_SIZE
    
    # Clamp epsilon to prevent numerical issues with very small values
    eps_clamped = tl.maximum(eps, 1e-8)
    
    # Handle edge case where variance is near zero
    is_near_zero = mean_of_squares < 1e-12
    safe_mean_of_squares = tl.where(is_near_zero, 1.0, mean_of_squares)
    
    # Compute inverse RMS
    rstd = tl.rsqrt(safe_mean_of_squares + eps_clamped)
    
    # Zero out rstd if input was near zero to avoid amplifying noise
    rstd = tl.where(is_near_zero, 0.0, rstd)
    
    # Apply normalization and weight entirely in fp32, then convert once at the end
    result_fp32 = (x_fp32 * rstd) * weight_fp32
    result = result_fp32.to(x_ptr.dtype.element_ty)
    
    # Store result
    tl.store(out_ptr + row_start + col_offsets, result, mask=mask)


@triton.jit
def rms_norm_fused_kernel(x_ptr, weight_ptr, out_ptr, n_rows, eps, 
                         BLOCK_SIZE: tl.constexpr, HIDDEN_SIZE: tl.constexpr):
    """
    Fused RMSNorm kernel - currently identical to basic kernel.
    
    This is kept separate for potential future optimizations like:
    - Fused activation functions
    - Residual connections
    - Multi-layer fusion
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Calculate row start offset
    row_start = row_idx * HIDDEN_SIZE
    
    # Load entire row at once (single-pass approach)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < HIDDEN_SIZE
    
    # Load input and weight in single operations
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    
    # Convert BOTH x and weight to fp32 for computation
    x_fp32 = x.to(tl.float32)
    weight_fp32 = weight.to(tl.float32)
    
    # Compute mean of squares
    mean_of_squares = tl.sum(x_fp32 * x_fp32) / HIDDEN_SIZE
    
    # Clamp epsilon to prevent numerical issues
    eps_clamped = tl.maximum(eps, 1e-8)
    
    # Handle edge case where variance is near zero
    is_near_zero = mean_of_squares < 1e-12
    safe_mean_of_squares = tl.where(is_near_zero, 1.0, mean_of_squares)
    
    # Compute inverse RMS
    rstd = tl.rsqrt(safe_mean_of_squares + eps_clamped)
    
    # Zero out rstd if input was near zero
    rstd = tl.where(is_near_zero, 0.0, rstd)
    
    # Apply normalization and weight entirely in fp32
    result_fp32 = (x_fp32 * rstd) * weight_fp32
    result = result_fp32.to(x_ptr.dtype.element_ty)
    
    # Store result
    tl.store(out_ptr + row_start + col_offsets, result, mask=mask)


# Auto-tuning configurations for different scenarios
AUTOTUNE_CONFIGS = [
    {'BLOCK_SIZE': 1024, 'HIDDEN_SIZE': 512},
    {'BLOCK_SIZE': 2048, 'HIDDEN_SIZE': 1024}, 
    {'BLOCK_SIZE': 4096, 'HIDDEN_SIZE': 2048},
    {'BLOCK_SIZE': 8192, 'HIDDEN_SIZE': 4096},
    {'BLOCK_SIZE': 16384, 'HIDDEN_SIZE': 8192},
]


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    High-level interface for Triton RMSNorm.
    
    Args:
        x: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Epsilon for numerical stability
        
    Returns:
        Output tensor of same shape as input
    """
    # Get input properties
    input_shape = x.shape
    hidden_size = input_shape[-1]
    
    # Flatten input to 2D for kernel processing
    x_flat = x.view(-1, hidden_size)
    n_rows = x_flat.shape[0]
    
    # Create output tensor
    out_flat = torch.empty_like(x_flat)
    
    # Select appropriate block size (must be >= hidden_size)
    block_size = 1024
    for config in AUTOTUNE_CONFIGS:
        if config['HIDDEN_SIZE'] == hidden_size:
            block_size = config['BLOCK_SIZE']
            break
    
    # Ensure block size is at least as large as hidden size
    while block_size < hidden_size:
        block_size *= 2
    
    # Launch kernel
    grid = (n_rows,)
    rms_norm_kernel[grid](
        x_flat, weight, out_flat, n_rows, eps,
        BLOCK_SIZE=block_size, HIDDEN_SIZE=hidden_size
    )
    
    # Reshape back to original shape
    return out_flat.view(input_shape)


def test_rms_norm():
    """Simple test function to verify kernel correctness."""
    batch_size = 2
    seq_len = 128
    hidden_size = 4096
    eps = 1e-6
    dtype = torch.float16
    
    print(f"Testing RMSNorm kernel with shape: {batch_size}x{seq_len}x{hidden_size}")
    
    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
    weight = torch.ones(hidden_size, device='cuda', dtype=dtype)
    
    # PyTorch reference
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    expected = x_norm * weight
    
    # Triton implementation
    output = rms_norm_triton(x, weight, eps)
    
    # Compare results
    diff = torch.abs(output - expected)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Set tolerance based on dtype
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.01
    
    if max_diff < tolerance:
        print("✅ Test PASSED")
    else:
        print("❌ Test FAILED")
    
    return max_diff < tolerance


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_rms_norm()
    else:
        print("CUDA not available - cannot test kernels")