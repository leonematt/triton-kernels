import os
import pytest
import torch
import nexus
import math
import struct
import torch.cuda as cuda
import triton

import numpy as np 
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

VARIANTS = [
    {'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_DMODEL': 16, 'CAUSAL': True},
    {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DMODEL': 64, 'CAUSAL': True},
    {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DMODEL': 128, 'CAUSAL': True},
    {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 64, 'CAUSAL': True},
    {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 128, 'CAUSAL': True},
    {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DMODEL': 64, 'CAUSAL': False},
    {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DMODEL': 128, 'CAUSAL': False},
    {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 64, 'CAUSAL': False},
    {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 128, 'CAUSAL': False},
    {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_DMODEL': 64, 'CAUSAL': True},
    {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_DMODEL': 128, 'CAUSAL': True},
    {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_DMODEL': 64, 'CAUSAL': False},
    {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_DMODEL': 128, 'CAUSAL': False},
]

def make_kernel_name_from_variant(v: dict) -> str:
    return (
        "flash_attention_kernel"
        f"_BLOCK_DMODEL_{v['BLOCK_DMODEL']}"
        f"_BLOCK_M_{v['BLOCK_M']}"
        f"_BLOCK_N_{v['BLOCK_N']}"
        f"_CAUSAL_{v['CAUSAL']}"
    )

def attention_reference(q, k, v, causal=False, sm_scale=None):
    """Reference implementation of attention."""
    batch, heads, seqlen_q, head_dim = q.shape
    _, _, seqlen_k, _ = k.shape
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Force float32 for an accurate comparison
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * float(sm_scale)
    
    if causal:
        # Use boolean mask for newer PyTorch versions
        causal_mask = torch.tril(torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, seqlen_q, seqlen_k)
        scores = scores.masked_fill(causal_mask == False, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    output = torch.matmul(attn_weights, v)
    
    return output


@pytest.fixture(scope="module")
def runtime_and_device():
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


@pytest.mark.parametrize(
    "variant",
    VARIANTS,
    ids=lambda v: f"BM_{v['BLOCK_M']}_BN_{v['BLOCK_N']}_D_{v['BLOCK_DMODEL']}_CAUSAL_{v['CAUSAL']}",
)
def test_flash_attention_variants(runtime_and_device, variant):

    rt, dev = runtime_and_device

    BLOCK_M = variant["BLOCK_M"]
    BLOCK_N = variant["BLOCK_N"]
    BLOCK_DMODEL = variant["BLOCK_DMODEL"]
    CAUSAL = variant["CAUSAL"]

    batch_size = 2
    num_heads = 4
    seqlen = 128
    head_dim = BLOCK_DMODEL

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Layout [B, H, S, D]
    q = torch.randn((batch_size, num_heads, seqlen, head_dim), dtype=torch.float32)
    k = torch.randn((batch_size, num_heads, seqlen, head_dim), dtype=torch.float32)
    v = torch.randn((batch_size, num_heads, seqlen, head_dim), dtype=torch.float32)
    out = torch.zeros_like(q)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()

    nb_q = dev.create_buffer(q)
    nb_k = dev.create_buffer(k)
    nb_v = dev.create_buffer(v)
    nb_out = dev.create_buffer(out)

    kernel_name = make_kernel_name_from_variant(variant)
    print(f"\n--- Testing Kernel: {kernel_name} ---")
    lib_path = f"ptx_kernels/{kernel_name}.ptx"

    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")

    lib = dev.load_library(lib_path)
    kern = lib.get_kernel(kernel_name)

    # Strides for [B, H, S, D]
    stride_qz, stride_qh, stride_qm, stride_qd = q.stride()
    stride_kz, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vz, stride_vh, stride_vn, stride_vd = v.stride()
    stride_oz, stride_oh, stride_om, stride_od = out.stride()

    sched = dev.create_schedule()
    cmd = sched.create_command(kern)

    arg_idx = 0
    cmd.set_arg(arg_idx, nb_q); arg_idx += 1
    cmd.set_arg(arg_idx, nb_k); arg_idx += 1
    cmd.set_arg(arg_idx, nb_v); arg_idx += 1
    cmd.set_arg(arg_idx, nb_out); arg_idx += 1
    cmd.set_arg(arg_idx, float(sm_scale)); arg_idx += 1
    cmd.set_arg(arg_idx, stride_qz); arg_idx += 1
    cmd.set_arg(arg_idx, stride_qh); arg_idx += 1
    cmd.set_arg(arg_idx, stride_qm); arg_idx += 1
    cmd.set_arg(arg_idx, stride_qd); arg_idx += 1
    cmd.set_arg(arg_idx, stride_kz); arg_idx += 1
    cmd.set_arg(arg_idx, stride_kh); arg_idx += 1
    cmd.set_arg(arg_idx, stride_kn); arg_idx += 1
    cmd.set_arg(arg_idx, stride_kd); arg_idx += 1
    cmd.set_arg(arg_idx, stride_vz); arg_idx += 1
    cmd.set_arg(arg_idx, stride_vh); arg_idx += 1
    cmd.set_arg(arg_idx, stride_vn); arg_idx += 1
    cmd.set_arg(arg_idx, stride_vd); arg_idx += 1
    cmd.set_arg(arg_idx, stride_oz); arg_idx += 1
    cmd.set_arg(arg_idx, stride_oh); arg_idx += 1
    cmd.set_arg(arg_idx, stride_om); arg_idx += 1
    cmd.set_arg(arg_idx, stride_od); arg_idx += 1
    cmd.set_arg(arg_idx, batch_size); arg_idx += 1
    cmd.set_arg(arg_idx, num_heads); arg_idx += 1
    cmd.set_arg(arg_idx, seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, seqlen); arg_idx += 1
    cmd.set_arg(arg_idx, 0); arg_idx += 1

    num_blocks_m = (seqlen + BLOCK_M - 1) // BLOCK_M
    total_blocks = num_blocks_m * num_heads * batch_size

    shared_mem = (BLOCK_M + BLOCK_N) * BLOCK_DMODEL * 16

    cmd.finalize([total_blocks, 1, 1], [128, 1, 1], shared_mem)
    sched.run()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            err = torch.cuda.get_last_error() if hasattr(torch.cuda, 'get_last_error') else None
            if err:
                print(f"CUDA Error after kernel launch: {err}")
    except Exception as e:
        print(f"Error during CUDA sync: {e}")
    torch.cuda.synchronize()

    nb_out.copy(out)

    expected = attention_reference(q, k, v, causal=CAUSAL, sm_scale=sm_scale)

    max_diff = torch.max(torch.abs(out - expected)).item()
    mean_diff = torch.mean(torch.abs(out - expected)).item()

    print(f"Max diff: {max_diff}, mean diff: {mean_diff}")
    
    if torch.all(out == 0):
        print("!!! KERNEL FAILED: Output tensor is all zeros.")
    else:
        print(f"Sample out[0,0,0,:5]: {out[0, 0, 0, :5]}")
        print(f"Sample exp[0,0,0,:5]: {expected[0, 0, 0, :5]}")

    assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2), (
        f"Mismatch for variant BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, "
        f"BLOCK_DMODEL={BLOCK_DMODEL}, CAUSAL={CAUSAL}. "
        f"Max diff={max_diff}, mean diff={mean_diff}"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])