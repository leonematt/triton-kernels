import os
import pytest
import torch
import nexus


# Must match Triton rotary VARIANTS
VARIANTS = [
    # non-interleaved
    {
        "BLOCK_K": 32,
        "BLOCK_M": 8,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": False,
        "CONJUGATE": False,
    },
    {
        "BLOCK_K": 64,
        "BLOCK_M": 8,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": False,
        "CONJUGATE": False,
    },
    {
        "BLOCK_K": 128,
        "BLOCK_M": 4,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": False,
        "CONJUGATE": False,
    },
    {
        "BLOCK_K": 256,
        "BLOCK_M": 4,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": False,
        "CONJUGATE": False,
    },
    # interleaved
    {
        "BLOCK_K": 32,
        "BLOCK_M": 4,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": True,
        "CONJUGATE": False,
    },
    {
        "BLOCK_K": 64,
        "BLOCK_M": 4,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": True,
        "CONJUGATE": False,
    },
    {
        "BLOCK_K": 128,
        "BLOCK_M": 4,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": True,
        "CONJUGATE": False,
    },
    {
        "BLOCK_K": 256,
        "BLOCK_M": 4,
        "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False,
        "INTERLEAVED": True,
        "CONJUGATE": False,
    },
]


def make_kernel_name_from_variant(v: dict) -> str:
    return (
        "rotary_kernel"
        f"_BLOCK_K_{v['BLOCK_K']}"
        f"_BLOCK_M_{v['BLOCK_M']}"
        f"_CONJUGATE_{v['CONJUGATE']}"
        f"_INTERLEAVED_{v['INTERLEAVED']}"
        f"_IS_SEQLEN_OFFSETS_TENSOR_{v['IS_SEQLEN_OFFSETS_TENSOR']}"
        f"_IS_VARLEN_{v['IS_VARLEN']}"
    )


def build_rope_cos_sin(seqlen_ro: int, rotary_dim_half: int, dtype=torch.float32):
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim_half, dtype=dtype) / rotary_dim_half))
    t = torch.arange(seqlen_ro, dtype=torch.float32).unsqueeze(1)
    freqs = t * inv_freq.unsqueeze(0)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def apply_rope_reference(x, cos, sin, rotary_dim, interleaved: bool):
    # x: [B, S, H, D]; cos/sin: [S, rotary_dim/2]
    B, S, H, D = x.shape
    rotary_dim_half = rotary_dim // 2
    x = x.clone()

    if not interleaved:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        x1, x2 = x_rot[..., :rotary_dim_half], x_rot[..., rotary_dim_half:]
        cos_b = cos.view(1, S, 1, rotary_dim_half)
        sin_b = sin.view(1, S, 1, rotary_dim_half)

        y1 = x1 * cos_b - x2 * sin_b
        y2 = x2 * cos_b + x1 * sin_b
        x_rotated = torch.cat([y1, y2], dim=-1)
        return torch.cat([x_rotated, x_pass], dim=-1)

    # interleaved [r0,i0,r1,i1,...]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    even_idx = torch.arange(0, rotary_dim, 2, device=x.device)
    odd_idx = even_idx + 1

    x_even = x_rot.index_select(-1, even_idx)
    x_odd = x_rot.index_select(-1, odd_idx)

    cos_b = cos.view(1, S, 1, rotary_dim_half)
    sin_b = sin.view(1, S, 1, rotary_dim_half)

    y_even = x_even * cos_b - x_odd * sin_b
    y_odd = x_odd * cos_b + x_even * sin_b

    x_rotated = torch.empty_like(x_rot)
    x_rotated[..., even_idx] = y_even
    x_rotated[..., odd_idx] = y_odd

    return torch.cat([x_rotated, x_pass], dim=-1)


@pytest.fixture(scope="module")
def runtime_and_device():
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


@pytest.mark.parametrize(
    "variant",
    VARIANTS,
    ids=lambda v: f"BK_{v['BLOCK_K']}_BM_{v['BLOCK_M']}_INT_{v['INTERLEAVED']}",
)
def test_rotary_kernel_variants(runtime_and_device, variant):
    rt, dev = runtime_and_device

    BLOCK_K = variant["BLOCK_K"]
    BLOCK_M = variant["BLOCK_M"]
    INTERLEAVED = variant["INTERLEAVED"]
    IS_SEQLEN_OFFSETS_TENSOR = variant["IS_SEQLEN_OFFSETS_TENSOR"]
    IS_VARLEN = variant["IS_VARLEN"]
    CONJUGATE = variant["CONJUGATE"]

    assert not IS_SEQLEN_OFFSETS_TENSOR
    assert not IS_VARLEN
    assert not CONJUGATE

    batch_size = 2
    num_heads = 4
    seqlen = 128
    head_dim = 64

    rotary_dim = min(BLOCK_K, head_dim)
    if rotary_dim % 2 == 1:
        rotary_dim -= 1
    rotary_dim_half = rotary_dim // 2
    seqlen_ro = seqlen

    # Layout [B, S, H, D]
    x = torch.randn((batch_size, seqlen, num_heads, head_dim), dtype=torch.float32)
    out = torch.zeros_like(x)

    cos, sin = build_rope_cos_sin(seqlen_ro, rotary_dim_half, dtype=torch.float32)

    x = x.contiguous()
    out = out.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Pre-copy the pass-through portion (rotary kernel doesn't handle this)
    if rotary_dim < head_dim:
        out[..., rotary_dim:].copy_(x[..., rotary_dim:])

    nb_out = dev.create_buffer(out)
    nb_x = dev.create_buffer(x)
    nb_cos = dev.create_buffer(cos)
    nb_sin = dev.create_buffer(sin)

    # Metadata buffer
    meta = torch.zeros(1, dtype=torch.int32)
    nb_meta = dev.create_buffer(meta)

    kernel_name = make_kernel_name_from_variant(variant)
    lib_path = f"ptx_kernels/{kernel_name}.ptx"

    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")

    lib = dev.load_library(lib_path)
    kern = lib.get_kernel(kernel_name)

    # Strides for [B, S, H, D]
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim = out.stride()
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim = x.stride()

    sched = dev.create_schedule()
    cmd = sched.create_command(kern)

    # Parameter order based on PTX signature:
    # 0-3: pointers (OUT, X, COS, SIN)
    # 4-5: integers (cu_seqlens, seqlen_offsets)
    # 6-8: integers (seqlen, rotary_dim, seqlen_ro)
    # 9-16: integers (8 strides)
    # 17: pointer (metadata)
    
    arg_idx = 0
    cmd.set_arg(arg_idx, nb_out); arg_idx += 1
    cmd.set_arg(arg_idx, nb_x); arg_idx += 1
    cmd.set_arg(arg_idx, nb_cos); arg_idx += 1
    cmd.set_arg(arg_idx, nb_sin); arg_idx += 1
    
    # These are integers (0) not pointers since IS_VARLEN=False
    cmd.set_arg(arg_idx, 0); arg_idx += 1  # cu_seqlens
    cmd.set_arg(arg_idx, 0); arg_idx += 1  # seqlen_offsets
    
    cmd.set_arg(arg_idx, seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, rotary_dim); arg_idx += 1
    cmd.set_arg(arg_idx, seqlen_ro); arg_idx += 1
    
    cmd.set_arg(arg_idx, stride_out_batch); arg_idx += 1
    cmd.set_arg(arg_idx, stride_out_seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, stride_out_nheads); arg_idx += 1
    cmd.set_arg(arg_idx, stride_out_headdim); arg_idx += 1
    
    cmd.set_arg(arg_idx, stride_x_batch); arg_idx += 1
    cmd.set_arg(arg_idx, stride_x_seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, stride_x_nheads); arg_idx += 1
    cmd.set_arg(arg_idx, stride_x_headdim); arg_idx += 1
    
    cmd.set_arg(arg_idx, nb_meta); arg_idx += 1

    # Grid: (pid_m over seq, pid_head, pid_batch)
    num_blocks_m = (seqlen + BLOCK_M - 1) // BLOCK_M
    grid_x = num_blocks_m
    grid_y = num_heads
    grid_z = batch_size

    print(
        f"\nTesting rotary_kernel variant={variant}, seqlen={seqlen}, "
        f"rotary_dim={rotary_dim}, grid=({grid_x},{grid_y},{grid_z})"
    )

    # PTX requires exactly 128 threads (.reqntid 128)
    cmd.finalize([grid_x, grid_y, grid_z], [128, 1, 1], 16384)

    sched.run()
    nb_out.copy(out)

    expected = apply_rope_reference(x, cos, sin, rotary_dim, INTERLEAVED)

    max_diff = torch.max(torch.abs(out - expected)).item()
    mean_diff = torch.mean(torch.abs(out - expected)).item()

    print(f"Max diff: {max_diff}, mean diff: {mean_diff}")
    print(f"Sample out[0,0,0,0]: {out[0, 0, 0, 0]}")
    print(f"Sample exp[0,0,0,0]: {expected[0, 0, 0, 0]}")

    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-4), (
        f"Mismatch for variant={variant}. "
        f"Max diff={max_diff}, mean diff={mean_diff}"
    )


@pytest.mark.parametrize("seqlen", [32, 64, 128, 256])
def test_rotary_kernel_different_seqlens(runtime_and_device, seqlen):
    """Quick sweep over different sequence lengths using a fixed variant."""
    rt, dev = runtime_and_device

    variant = VARIANTS[0]
    BLOCK_K = variant["BLOCK_K"]
    BLOCK_M = variant["BLOCK_M"]
    INTERLEAVED = variant["INTERLEAVED"]

    batch_size = 2
    num_heads = 4
    head_dim = 64

    rotary_dim = min(BLOCK_K, head_dim)
    if rotary_dim % 2 == 1:
        rotary_dim -= 1
    rotary_dim_half = rotary_dim // 2
    seqlen_ro = seqlen

    x = torch.randn((batch_size, seqlen, num_heads, head_dim), dtype=torch.float32)
    out = torch.zeros_like(x)
    cos, sin = build_rope_cos_sin(seqlen_ro, rotary_dim_half, dtype=torch.float32)

    x = x.contiguous()
    out = out.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Pre-copy pass-through portion
    if rotary_dim < head_dim:
        out[..., rotary_dim:].copy_(x[..., rotary_dim:])

    nb_out = dev.create_buffer(out)
    nb_x = dev.create_buffer(x)
    nb_cos = dev.create_buffer(cos)
    nb_sin = dev.create_buffer(sin)

    meta = torch.zeros(1, dtype=torch.int32)
    nb_meta = dev.create_buffer(meta)

    kernel_name = make_kernel_name_from_variant(variant)
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")

    lib = dev.load_library(lib_path)
    kern = lib.get_kernel(kernel_name)

    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim = out.stride()
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim = x.stride()

    sched = dev.create_schedule()
    cmd = sched.create_command(kern)

    arg_idx = 0
    cmd.set_arg(arg_idx, nb_out); arg_idx += 1
    cmd.set_arg(arg_idx, nb_x); arg_idx += 1
    cmd.set_arg(arg_idx, nb_cos); arg_idx += 1
    cmd.set_arg(arg_idx, nb_sin); arg_idx += 1
    cmd.set_arg(arg_idx, 0); arg_idx += 1
    cmd.set_arg(arg_idx, 0); arg_idx += 1
    cmd.set_arg(arg_idx, seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, rotary_dim); arg_idx += 1
    cmd.set_arg(arg_idx, seqlen_ro); arg_idx += 1
    cmd.set_arg(arg_idx, stride_out_batch); arg_idx += 1
    cmd.set_arg(arg_idx, stride_out_seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, stride_out_nheads); arg_idx += 1
    cmd.set_arg(arg_idx, stride_out_headdim); arg_idx += 1
    cmd.set_arg(arg_idx, stride_x_batch); arg_idx += 1
    cmd.set_arg(arg_idx, stride_x_seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, stride_x_nheads); arg_idx += 1
    cmd.set_arg(arg_idx, stride_x_headdim); arg_idx += 1
    cmd.set_arg(arg_idx, 0); arg_idx += 1

    num_blocks_m = (seqlen + BLOCK_M - 1) // BLOCK_M
    grid_x = num_blocks_m
    grid_y = num_heads
    grid_z = batch_size

    print(f"\nTesting rotary_kernel seqlen={seqlen} with variant={variant}")
    
    cmd.finalize([grid_x, grid_y, grid_z], [128, 1, 1], 16384)

    sched.run()
    nb_out.copy(out)

    expected = apply_rope_reference(x, cos, sin, rotary_dim, INTERLEAVED)
    max_diff = torch.max(torch.abs(out - expected)).item()
    print(f"seqlen={seqlen}, max diff={max_diff}")

    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-4), \
        f"Mismatch for seqlen={seqlen}, max diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])