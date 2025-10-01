#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_nontiled_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # unused (kept for symmetric signature)
):
    """
    Naive blocked matmul - no shared memory, no K-tiling.
    Each program computes a BM x BN tile by looping K with scalar/broadcast loads.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    a_row_ptr = a_ptr + offs_m[:, None] * stride_am  # (BM,1)
    b_col_ptr = b_ptr + offs_n[None, :] * stride_bn  # (1,BN)

    for k in range(0, K):
        a_vals = tl.load(a_row_ptr + k * stride_ak, mask=(offs_m[:, None] < M), other=0.0)  # (BM,1)
        b_vals = tl.load(b_col_ptr + k * stride_bk, mask=(offs_n[None, :] < N), other=0.0)  # (1,BN)
        acc += a_vals * b_vals  # broadcast outer-product add

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


variants = [
    # M=16
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
    # M=32
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
    # M=64
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16},
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32},
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
]


def matmul_nontiled(a, b, block_size_m, block_size_n, block_size_k):
    """
    Non-tiled matrix multiplication wrapper.
    a: (M, K) matrix
    b: (K, N) matrix
    returns: (M, N) matrix
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)
    
    matmul_nontiled_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=1,
        num_stages=1,
    )
    
    return c


def main():
    M, N, K = 512, 512, 512
    print(f"--- Running Nontiled Matmul Variants (Matrix Size: {M}x{N}x{K}) ---\n")
    
    torch.manual_seed(42)
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    # PyTorch reference
    expected = a @ b
    
    passed = 0
    failed = 0
    errors = 0
    
    for v in variants:
        bm, bn, bk = v['BLOCK_SIZE_M'], v['BLOCK_SIZE_N'], v['BLOCK_SIZE_K']
        print(f"Testing BM={bm:3}, BN={bn:3}, BK={bk:3}... ", end="", flush=True)
        
        try:
            triton_output = matmul_nontiled(a, b, 
                                                   block_size_m=bm, 
                                                   block_size_n=bn, 
                                                   block_size_k=bk)
            
            if torch.allclose(triton_output, expected, rtol=5e-3, atol=1e-1):
                print("PASSED")
                passed += 1
            else:
                max_diff = (triton_output - expected).abs().max().item()
                print(f"FAILED (max_diff={max_diff:.6f})")
                failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
    
    print(f"\n--- Summary ---")
    print(f"Total variants: {len(variants)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()