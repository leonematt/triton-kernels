#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def add_vectors(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE2: tl.constexpr):
    pid = tl.program_id(0)
    # Use both block sizes in the computation
    total_block_size = BLOCK_SIZE + BLOCK_SIZE2
    offsets = pid * total_block_size + tl.arange(0, total_block_size)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

# Define variants with both BLOCK_SIZE and BLOCK_SIZE2
variants = [
    {'BLOCK_SIZE': 128, 'BLOCK_SIZE2': 64},
    {'BLOCK_SIZE': 128, 'BLOCK_SIZE2': 128},
    {'BLOCK_SIZE': 256, 'BLOCK_SIZE2': 128},
    {'BLOCK_SIZE': 256, 'BLOCK_SIZE2': 256},
    {'BLOCK_SIZE': 512, 'BLOCK_SIZE2': 256},
    {'BLOCK_SIZE': 512, 'BLOCK_SIZE2': 512}
]

def run():
    n = 4096

    for variant in variants:
        block_size = variant['BLOCK_SIZE']
        block_size2 = variant['BLOCK_SIZE2']
        total_block_size = block_size + block_size2
        
        print(f"Running with BLOCK_SIZE={block_size}, BLOCK_SIZE2={block_size2}")

        x = torch.randn(n, device='cuda', dtype=torch.float32)
        y = torch.randn(n, device='cuda', dtype=torch.float32)
        out = torch.empty_like(x)

        # Calculate grid size based on total block size
        grid = (triton.cdiv(n, total_block_size),)
        add_vectors[grid](x, y, out, n, BLOCK_SIZE=block_size, BLOCK_SIZE2=block_size2)

        expected = x + y
        max_error = torch.max(torch.abs(out - expected)).item()
        print(f"  Total block size: {total_block_size}")
        print(f"  Grid size: {grid}")
        print(f"  Max error: {max_error:.2e}")
        print(f"  Success: {max_error < 1e-5}")
        print()

if __name__ == "__main__":
    run()