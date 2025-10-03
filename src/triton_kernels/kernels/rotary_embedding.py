#!/usr/bin/env python3

# Copyright (c) 2023, Tri Dao.
# Source: https://github.com/Dao-AILab/flash-attention

# Copyright (c) 2025, Kernelize AI


import torch
import triton
import triton.language as tl
from typing import Optional, Union


@triton.jit
def rotary_embedding_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    rotary_dim,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Rotary embedding kernel: applies rotary positional embeddings to input tensor.
    
    Processes blocks of the input in parallel across batch, heads, and sequence dimensions.
    Supports both interleaved and non-interleaved rotation formats.
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # Interleaved format
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def compute_rotary_cos_sin(seqlen: int, rotary_dim: int, base: float = 10000.0, device: str = 'cuda'):
    """
    Precompute cos and sin values for rotary embeddings.
    """
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    t = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    
    return cos, sin


# Test variants with different configurations
variants = [
    {'BATCH': 2, 'SEQLEN': 128, 'NHEADS': 4, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 2, 'SEQLEN': 256, 'NHEADS': 8, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 4, 'SEQLEN': 512, 'NHEADS': 8, 'HEADDIM': 128, 'ROTARY_DIM': 128, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 2, 'SEQLEN': 128, 'NHEADS': 4, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': True, 'INPLACE': False},
    {'BATCH': 1, 'SEQLEN': 1024, 'NHEADS': 16, 'HEADDIM': 64, 'ROTARY_DIM': 32, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 2, 'SEQLEN': 128, 'NHEADS': 4, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': False, 'INPLACE': True},
]


def run():
    print("--- Running Rotary Embedding Kernel Variants ---\n")

    for i, variant in enumerate(variants):
        batch = variant['BATCH']
        seqlen = variant['SEQLEN']
        nheads = variant['NHEADS']
        headdim = variant['HEADDIM']
        rotary_dim = variant['ROTARY_DIM']
        interleaved = variant['INTERLEAVED']
        inplace = variant['INPLACE']
        seqlen_offsets = 0
        conjugate = False

        print(f"Variant {i+1}: batch={batch}, seqlen={seqlen}, nheads={nheads}, "
              f"headdim={headdim}, rotary_dim={rotary_dim}, interleaved={interleaved}, "
              f"inplace={inplace}... ", end="")

        try:
            x = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float32)
            x_original = x.clone()
            
            cos, sin = compute_rotary_cos_sin(seqlen, rotary_dim, device='cuda')
            cos, sin = cos.contiguous(), sin.contiguous()
            
            output = x if inplace else torch.empty_like(x)
            if rotary_dim < headdim and not inplace:
                output[..., rotary_dim:].copy_(x[..., rotary_dim:])
        
            BLOCK_K = (32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256)))
            BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 128 else 4)
            
            grid = (triton.cdiv(seqlen, BLOCK_M), nheads, batch)
            
            with torch.cuda.device(x.device.index):
                rotary_embedding_kernel[grid](
                    output, x, cos, sin,
                    None, seqlen_offsets,
                    seqlen, rotary_dim, seqlen,
                    output.stride(0), output.stride(-3), output.stride(-2), output.stride(-1),
                    x.stride(0), x.stride(-3), x.stride(-2), x.stride(-1),
                    BLOCK_K=BLOCK_K,
                    IS_SEQLEN_OFFSETS_TENSOR=False,
                    IS_VARLEN=False,
                    INTERLEAVED=interleaved,
                    CONJUGATE=conjugate,
                    BLOCK_M=BLOCK_M,
                    num_warps=2 if rotary_dim <= 64 else 4,
                )
            
            assert output.shape == x_original.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            if rotary_dim < headdim:
                assert torch.allclose(output[..., rotary_dim:], x_original[..., rotary_dim:])
            
            print("PASSED")

        except Exception as e:
            print(f"ERROR: {e}")

    print("\n--- All variants tested. ---")


if __name__ == "__main__":
    run()