#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_tiled(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Tiled matmul - loads blocks of K dimension, uses tl.dot for tensor cores.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for ki in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - ki * BLOCK_SIZE_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


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


def run():
    M, N, K = 512, 512, 512
    print(f"--- Running Tiled Matmul Variants (Matrix Size: {M}x{N}x{K}) ---")

    torch.manual_seed(42)
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    expected = a @ b

    for v in variants:
        bm, bn, bk = v['BLOCK_SIZE_M'], v['BLOCK_SIZE_N'], v['BLOCK_SIZE_K']
        print(f"Testing BM={bm:3}, BN={bn:3}, BK={bk:3}... ", end="", flush=True)
        
        try:
            output = torch.empty((M, N), device='cuda', dtype=torch.float32)
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            
            matmul_kernel_tiled[grid](
                a, b, output, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                num_warps=4, num_stages=2,
            )
            
            if torch.allclose(output, expected, rtol=5e-3, atol=1e-1):
                print("PASSED")
            else:
                max_diff = (output - expected).abs().max().item()
                print(f"FAILED (max_diff={max_diff:.6f})")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n--- All variants tested. ---")


if __name__ == "__main__":
    run()