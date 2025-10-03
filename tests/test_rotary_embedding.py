import pytest
import torch
import nexus
import os

# Test variants matching your rotary_embedding.py variants
VARIANTS = [
    {'BATCH': 2, 'SEQLEN': 128, 'NHEADS': 4, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 2, 'SEQLEN': 256, 'NHEADS': 8, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 4, 'SEQLEN': 512, 'NHEADS': 8, 'HEADDIM': 128, 'ROTARY_DIM': 128, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 2, 'SEQLEN': 128, 'NHEADS': 4, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': True, 'INPLACE': False},
    {'BATCH': 1, 'SEQLEN': 1024, 'NHEADS': 16, 'HEADDIM': 64, 'ROTARY_DIM': 32, 'INTERLEAVED': False, 'INPLACE': False},
    {'BATCH': 2, 'SEQLEN': 128, 'NHEADS': 4, 'HEADDIM': 64, 'ROTARY_DIM': 64, 'INTERLEAVED': False, 'INPLACE': True},
]


@pytest.fixture(scope="module")
def runtime_and_device():
    """Setup runtime and device once for all tests"""
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


def compute_rotary_cos_sin(seqlen, rotary_dim, base=10000.0):
    """Compute cos and sin for rotary embeddings"""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(seqlen, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def naive_rotary_embedding(x, cos, sin, interleaved=False):
    """Reference PyTorch implementation for verification"""
    batch, seqlen, nheads, headdim = x.shape
    rotary_dim = cos.shape[1] * 2
    
    x_rot = x[..., :rotary_dim].clone()
    x_pass = x[..., rotary_dim:].clone()
    
    cos = cos[:seqlen, :].unsqueeze(0).unsqueeze(2)
    sin = sin[:seqlen, :].unsqueeze(0).unsqueeze(2)
    
    if not interleaved:
        x1 = x_rot[..., :rotary_dim//2]
        x2 = x_rot[..., rotary_dim//2:rotary_dim]
        
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        
        output_rot = torch.cat([o1, o2], dim=-1)
    else:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        
        output_rot = torch.empty_like(x_rot)
        output_rot[..., 0::2] = o1
        output_rot[..., 1::2] = o2
    
    if rotary_dim < headdim:
        return torch.cat([output_rot, x_pass], dim=-1)
    return output_rot


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"b{v['BATCH']}_s{v['SEQLEN']}_h{v['NHEADS']}_d{v['HEADDIM']}_r{v['ROTARY_DIM']}_i{v['INTERLEAVED']}_ip{v['INPLACE']}")
def test_rotary_embedding(runtime_and_device, variant):
    """Test rotary embedding kernel with different configurations"""
    rt, dev = runtime_and_device
    
    batch = variant['BATCH']
    seqlen = variant['SEQLEN']
    nheads = variant['NHEADS']
    headdim = variant['HEADDIM']
    rotary_dim = variant['ROTARY_DIM']
    interleaved = variant['INTERLEAVED']
    inplace = variant['INPLACE']
    
    # Create test data
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
    x_original = x.clone()
    
    # Compute cos/sin
    cos, sin = compute_rotary_cos_sin(seqlen, rotary_dim)
    
    # Prepare output
    if inplace:
        output = x.clone()
    else:
        output = torch.empty_like(x)
        if rotary_dim < headdim:
            output[..., rotary_dim:] = x[..., rotary_dim:]
    
    # Ensure contiguous
    x = x.contiguous()
    output = output.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # Create device buffers
    nb_out = dev.create_buffer(output)
    nb_x = dev.create_buffer(x)
    nb_cos = dev.create_buffer(cos)
    nb_sin = dev.create_buffer(sin)
    
    # Create dummy buffer
    dummy = torch.zeros(1, dtype=torch.float32)
    nb_dummy = dev.create_buffer(dummy)
    
    # Load kernel
    kernel_name = f'rotary_embedding_kernel_BATCH{batch}_SEQLEN{seqlen}_NHEADS{nheads}_HEADDIM{headdim}_ROTARY_DIM{rotary_dim}_INTERLEAVED{interleaved}_INPLACE{inplace}'
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    
    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")
    
    try:
        lib = dev.load_library_file(lib_path)
        kern = lib.get_kernel(kernel_name)
    except Exception as e:
        pytest.fail(f"Failed to load kernel {kernel_name}: {e}")
    
    # Determine block sizes
    BLOCK_K = 32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 128 else 4)
    
    # Calculate grid dimensions
    grid_m = (seqlen + BLOCK_M - 1) // BLOCK_M
    grid = [grid_m, nheads, batch]
    block = [64, 1, 1]
    
    print(f"\nTesting rotary embedding:")
    print(f"  Config: batch={batch}, seqlen={seqlen}, nheads={nheads}, headdim={headdim}")
    print(f"  Rotary: rotary_dim={rotary_dim}, interleaved={interleaved}, inplace={inplace}")
    print(f"  Grid: {grid}, Block: {block}")
    
    # Create and configure command
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    # Set arguments - PTX has 15 params (0-14) but Triton optimizes away 2 strides
    # Try passing all 17 to match what Triton does
    cmd.set_arg(0, nb_out)                    # OUT pointer
    cmd.set_arg(1, nb_x)                      # X pointer
    cmd.set_arg(2, nb_cos)                    # COS pointer
    cmd.set_arg(3, nb_sin)                    # SIN pointer
    cmd.set_arg(4, 0)                         # seqlen_offsets (u32)
    cmd.set_arg(5, seqlen)                    # seqlen (u32)
    cmd.set_arg(6, rotary_dim)                # rotary_dim (u32)
    cmd.set_arg(7, seqlen)                    # seqlen_ro (u32)
    cmd.set_arg(8, output.stride(0))          # stride_out_batch
    cmd.set_arg(9, output.stride(1))          # stride_out_seqlen
    cmd.set_arg(10, output.stride(2))         # stride_out_nheads
    cmd.set_arg(11, output.stride(3))         # stride_out_headdim
    cmd.set_arg(12, x.stride(0))              # stride_x_batch
    cmd.set_arg(13, x.stride(1))              # stride_x_seqlen
    cmd.set_arg(14, x.stride(2))              # stride_x_nheads (optimized away in PTX)
    cmd.set_arg(15, x.stride(3))              # stride_x_headdim (optimized away in PTX)
    cmd.set_arg(16, nb_dummy)                 # dummy pointer
    
    cmd.finalize(grid, block)
    
    # Run kernel
    sched.run()
    
    # Copy result back
    nb_out.copy(output)
    
    # Compute expected result
    expected = naive_rotary_embedding(x_original, cos, sin, interleaved=interleaved)
    
    # Verify results
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0, 0, 0, :5].tolist()}")
    print(f"  Expected sample: {expected[0, 0, 0, :5].tolist()}")
    
    # Check rotary dimensions
    if not torch.allclose(output[..., :rotary_dim], expected[..., :rotary_dim], rtol=1e-4, atol=1e-4):
        max_diff = (output[..., :rotary_dim] - expected[..., :rotary_dim]).abs().max().item()
        pytest.fail(f"Rotary dimensions mismatch. Max diff: {max_diff}")
    
    # Check non-rotary dimensions (if applicable)
    if rotary_dim < headdim:
        assert torch.allclose(output[..., rotary_dim:], expected[..., rotary_dim:], rtol=1e-5, atol=1e-5), \
            "Non-rotary dimensions should be unchanged"
    
    print("  ✓ Test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])