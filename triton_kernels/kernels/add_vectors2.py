#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit  
def add_vectors(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE2: tl.constexpr):
    pid = tl.program_id(0)

    # Process BLOCK_SIZE elements
    offsets1 = pid * (BLOCK_SIZE + BLOCK_SIZE2) + tl.arange(0, BLOCK_SIZE)
    mask1 = offsets1 < n
    x1 = tl.load(x_ptr + offsets1, mask=mask1)
    y1 = tl.load(y_ptr + offsets1, mask=mask1)
    tl.store(out_ptr + offsets1, x1 + y1, mask=mask1)

    # Process BLOCK_SIZE2 elements
    offsets2 = pid * (BLOCK_SIZE + BLOCK_SIZE2) + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE2)
    mask2 = offsets2 < n
    x2 = tl.load(x_ptr + offsets2, mask=mask2)
    y2 = tl.load(y_ptr + offsets2, mask=mask2)
    tl.store(out_ptr + offsets2, x2 + y2, mask=mask2)

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