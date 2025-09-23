#!/usr/bin/env python3

import torch
import os
import sys
from pathlib import Path

# Add kernels directory to path
kernels_dir = Path(__file__).parent.parent / "ptx_triton_kernels"
sys.path.insert(0, str(kernels_dir))

def pytorch_rotary_embedding(query: torch.Tensor,
                             key: torch.Tensor,
                             positions: torch.Tensor,
                             head_size: int,
                             is_neox: bool,
                             base: float = 10000.0):
  seq_len = positions.shape[0]
  rotary_dim = head_size
  device = query.device
  
  inv_freq = 1.0 / (base**(
    torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim
  ))
  t = positions.float()
  freqs = torch.outer(t, inv_freq)
  cos = torch.cos(freqs)
  sin = torch.sin(freqs)
  
  def apply_rope(tensor, cos, sin):
    reshaped_tensor = tensor.reshape(seq_len, -1, head_size)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    if is_neox:
      p1, p2 = torch.chunk(reshaped_tensor, 2, dim=-1)
      rotated = torch.cat([p1 * cos - p2 * sin, p2 * cos + p1 * sin], dim=-1)
    else:
      reshaped_pairs = reshaped_tensor.reshape(*reshaped_tensor.shape[:-1], -1, 2)
      p1 = reshaped_pairs[..., 0]
      p2 = reshaped_pairs[..., 1]
      
      rotated_pairs = torch.empty_like(reshaped_pairs)
      rotated_pairs[..., 0] = p1 * cos - p2 * sin
      rotated_pairs[..., 1] = p2 * cos + p1 * sin
      rotated = rotated_pairs.flatten(start_dim=-2)
      
    return rotated.reshape_as(tensor)

  q_out = apply_rope(query, cos, sin)
  k_out = apply_rope(key, cos, sin)
  return q_out, k_out

def test_triton_rope():
  seq_len = 128
  num_heads = 32
  num_kv_heads = 8
  head_size = 128
  rotary_dim = head_size
  hidden_size = num_heads * head_size
  max_pos = 2048
  base = 10000.0
  dtype = torch.float16
  
  positions = torch.arange(seq_len, dtype=torch.long, device="cuda")
  query = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
  key = torch.randn(seq_len, num_kv_heads * head_size, dtype=dtype, device="cuda")
  
  # Create cos/sin cache
  inv_freq = 1.0 / (base ** (
    torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim
  ))
  t = torch.arange(max_pos, dtype=torch.float32, device="cuda")
  freqs = torch.outer(t, inv_freq)
  cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).to(dtype)
  
  for is_neox_style in [True, False]:
    for do_key_rope in [True, False]:
      print("=" * 60)
      print(f"RoPE Comparison: IS_NEOX = {is_neox_style}, DO_KEY_ROPE = {do_key_rope}")
      print("=" * 60)
      print(f"Shape: Query({seq_len}, {hidden_size}), Key({seq_len}, {num_kv_heads * head_size})")
      print(f"Num heads: {num_heads}, Head size: {head_size}")
      print(f"Dtype: {dtype}\n")
      
      # 1. PyTorch reference
      print(f"1. PyTorch Reference:")
      q_out_ref, k_out_ref = pytorch_rotary_embedding(
        query.clone(), key.clone(), positions, rotary_dim, is_neox_style, base
      )
      print(f"   Query output mean: {q_out_ref.mean().item():.6f}")
      print(f"   Key output mean: {k_out_ref.mean().item():.6f}")
      
      # 2. Triton JIT kernel
      print(f"\n2. Triton JIT Kernel:")
      
      # Reshape for kernel (seq_len, num_heads, head_size)
      q_test = query.clone().view(seq_len, num_heads, head_size)
      k_test = key.clone().view(seq_len, num_kv_heads, head_size)
      
      # Calculate strides
      query_stride_token = q_test.stride(0)
      key_stride_token = k_test.stride(0)
      head_stride = q_test.stride(1)
      
      try:
        # Launch Triton kernel
        grid = (seq_len,)
        rotary_embedding_kernel[grid](
          positions, q_test, k_test, cos_sin_cache,
          query_stride_token, key_stride_token, head_stride,
          num_heads, num_kv_heads, head_size, rotary_dim,
          IS_NEOX=is_neox_style,
          DO_KEY_ROPE=do_key_rope,
          BLOCK_SIZE_ROT=256
        )
        
        # Reshape back for comparison
        output_q_triton = q_test.view(seq_len, hidden_size)
        output_k_triton = k_test.view(seq_len, -1) if do_key_rope else k_out_ref
        
        print(f"   Query output mean: {output_q_triton.mean().item():.6f}")
        print(f"   Key output mean: {output_k_triton.mean().item():.6f}")
        
        # Compare with appropriate tolerances
        diff_q = torch.abs(output_q_triton - q_out_ref)
        diff_k = torch.abs(output_k_triton - k_out_ref) if do_key_rope else torch.tensor(0.0)
        max_diff_q = diff_q.max().item()
        max_diff_k = diff_k.max().item() if do_key_rope else 0.0
        
        print(f"\n   Comparison:")
        print(f"   Query max difference: {max_diff_q:.2e}")
        if do_key_rope:
          print(f"   Key max difference: {max_diff_k:.2e}")
        else:
          print(f"   Key difference: N/A (DO_KEY_ROPE=False)")
        
        tolerance = 0.05
        print(f"   Tolerance: {tolerance}")
        
        max_diff = max(max_diff_q, max_diff_k) if do_key_rope else max_diff_q
        if max_diff < tolerance:
          print(f"   ✅ PASSED (within tolerance)")
        else:
          print(f"   ❌ FAILED (exceeds tolerance)")
        
      except Exception as e:
        print(f"   ❌ FAILED: {e}")
      
      print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
  try:
    from rotary_embedding import rotary_embedding_kernel
  except ImportError as e:
    print(f"Could not import rotary_embedding_kernel: {e}")
    print("Make sure rotary_embedding.py is in the kernels directory.")
    sys.exit(1)
  
  if not torch.cuda.is_available():
    print("This test requires a CUDA-enabled GPU.")
    sys.exit(1)

  test_triton_rope()