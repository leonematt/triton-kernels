import torch
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(
    Positions_ptr, Query_ptr, Key_ptr, CosSinCache_ptr,
    query_stride_token, key_stride_token, head_stride,
    num_heads, num_kv_heads, head_size, rot_dim,
    IS_NEOX: tl.constexpr, DO_KEY_ROPE: tl.constexpr, BLOCK_SIZE_ROT: tl.constexpr,
):
    """Triton kernel for Rotary Position Embedding."""
    # (Kernel implementation is identical to before)
    token_idx = tl.program_id(0)
    pos = tl.load(Positions_ptr + token_idx)
    rot_offset_dim = rot_dim // 2
    cache_base_ptr = CosSinCache_ptr + pos * rot_dim
    cos_base_ptr = cache_base_ptr
    sin_base_ptr = cache_base_ptr + rot_offset_dim
    num_q_work_items = num_heads * rot_offset_dim
    for i_base in range(0, num_q_work_items, BLOCK_SIZE_ROT):
        i_offsets = i_base + tl.arange(0, BLOCK_SIZE_ROT)
        i_mask = i_offsets < num_q_work_items
        head_idx = i_offsets // rot_offset_dim
        rot_offset = i_offsets % rot_offset_dim
        token_q_ptr = Query_ptr + token_idx * query_stride_token
        token_head_q_ptr = token_q_ptr + head_idx * head_stride
        if IS_NEOX:
            x_indices, y_indices, cos_sin_load_indices = rot_offset, rot_offset + rot_offset_dim, rot_offset
        else:
            x_indices, y_indices, cos_sin_load_indices = 2 * rot_offset, 2 * rot_offset + 1, rot_offset
        cos = tl.load(cos_base_ptr + cos_sin_load_indices, mask=i_mask)
        sin = tl.load(sin_base_ptr + cos_sin_load_indices, mask=i_mask)
        x = tl.load(token_head_q_ptr + x_indices, mask=i_mask)
        y = tl.load(token_head_q_ptr + y_indices, mask=i_mask)
        x_new, y_new = x * cos - y * sin, y * cos + x * sin
        tl.store(token_head_q_ptr + x_indices, x_new, mask=i_mask)
        tl.store(token_head_q_ptr + y_indices, y_new, mask=i_mask)
    if DO_KEY_ROPE:
        num_k_work_items = num_kv_heads * rot_offset_dim
        for i_base in range(0, num_k_work_items, BLOCK_SIZE_ROT):
            i_offsets = i_base + tl.arange(0, BLOCK_SIZE_ROT)
            i_mask = i_offsets < num_k_work_items
            head_idx = i_offsets // rot_offset_dim
            rot_offset = i_offsets % rot_offset_dim
            token_k_ptr = Key_ptr + token_idx * key_stride_token
            token_head_k_ptr = token_k_ptr + head_idx * head_stride
            if IS_NEOX:
                x_indices, y_indices, cos_sin_load_indices = rot_offset, rot_offset + rot_offset_dim, rot_offset
            else:
                x_indices, y_indices, cos_sin_load_indices = 2 * rot_offset, 2 * rot_offset + 1, rot_offset
            cos = tl.load(cos_base_ptr + cos_sin_load_indices, mask=i_mask)
            sin = tl.load(sin_base_ptr + cos_sin_load_indices, mask=i_mask)
            x = tl.load(token_head_k_ptr + x_indices, mask=i_mask)
            y = tl.load(token_head_k_ptr + y_indices, mask=i_mask)
            x_new, y_new = x * cos - y * sin, y * cos + x * sin
            tl.store(token_head_k_ptr + x_indices, x_new, mask=i_mask)
            tl.store(token_head_k_ptr + y_indices, y_new, mask=i_mask)
            
            
variants = [
  {'IS_NEOX': True, 'DO_KEY_ROPE': True, 'BLOCK_SIZE_ROT': 256},
  {'IS_NEOX': True, 'DO_KEY_ROPE': False, 'BLOCK_SIZE_ROT': 256},
  {'IS_NEOX': False, 'DO_KEY_ROPE': True, 'BLOCK_SIZE_ROT': 256},
  {'IS_NEOX': False, 'DO_KEY_ROPE': False, 'BLOCK_SIZE_ROT': 256},
]

def run():
    """
    Run the rotary embedding kernel with different variants to populate the cache.
    """
    import torch
    import triton
    
    # Test parameters
    seq_len = 128
    num_heads = 32
    num_kv_heads = 8  # For GQA (Grouped Query Attention)
    head_size = 128
    rot_dim = head_size  # Full rotational dimension
    max_pos = 2048
    base = 10000.0
    dtype = torch.float16
    device = 'cuda'
    
    print(f"Running rotary embedding kernel with {len(variants)} variants")
    
    # Create test tensors
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    query = torch.randn(seq_len, num_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
    
    # Create cos/sin cache
    inv_freq = 1.0 / (base ** (
        torch.arange(0, rot_dim, 2, dtype=torch.float32, device=device) / rot_dim
    ))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).to(dtype)
    
    # Calculate strides
    query_stride_token = query.stride(0)
    key_stride_token = key.stride(0)
    head_stride = query.stride(1)
    
    for i, variant in enumerate(variants):
        is_neox = variant['IS_NEOX']
        do_key_rope = variant['DO_KEY_ROPE']
        block_size_rot = variant['BLOCK_SIZE_ROT']
        
        print(f"  [{i+1}/{len(variants)}] IS_NEOX={is_neox}, DO_KEY_ROPE={do_key_rope}, BLOCK_SIZE_ROT={block_size_rot}")
        
        # Clone tensors for this test
        query_test = query.clone()
        key_test = key.clone()
        
        try:
            # Calculate grid size
            grid = (seq_len,)
            
            # Launch kernel
            rotary_embedding_kernel[grid](
                positions, query_test, key_test, cos_sin_cache,
                query_stride_token, key_stride_token, head_stride,
                num_heads, num_kv_heads, head_size, rot_dim,
                IS_NEOX=is_neox, 
                DO_KEY_ROPE=do_key_rope, 
                BLOCK_SIZE_ROT=block_size_rot
            )
            
            # Verify outputs are reasonable
            query_mean = query_test.mean().item()
            key_mean = key_test.mean().item()
            
            print(f"    Query mean: {query_mean:.6f}, Key mean: {key_mean:.6f}")
            print(f"    Success: Kernel executed without errors")
            
        except Exception as e:
            print(f"    Failed: {e}")
        
        print()

if __name__ == "__main__":
    run()