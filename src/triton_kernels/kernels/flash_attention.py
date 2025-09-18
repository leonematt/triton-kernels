import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX_Q, N_CTX_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Flash Attention kernel - simplified version."""
    # Get program ID and decompose into batch/head/query_block
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX_Q, BLOCK_M)
    num_pid_in_group = num_pid_m
    group_id = pid // num_pid_in_group
    pid_m = pid % num_pid_in_group
    
    off_z = group_id // H
    off_h = group_id % H
    
    # Offset arrays
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Base offsets for this batch and head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    # Load Q block
    q_ptrs = q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_CTX_Q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    qk_scale = sm_scale * 1.44269504089
    
    # Determine blocks to process
    if CAUSAL:
        n_blocks = tl.cdiv(tl.minimum((pid_m + 1) * BLOCK_M, N_CTX_K), BLOCK_N)
    else:
        n_blocks = tl.cdiv(N_CTX_K, BLOCK_N)
    
    # Loop over K, V blocks
    for block_n in range(0, n_blocks):
        start_n = block_n * BLOCK_N
        
        # Load K
        k_ptrs = k_ptr + k_offset + offs_d[:, None] * stride_kd + (start_n + offs_n[None, :]) * stride_kn
        k_mask = (start_n + offs_n[None, :]) < N_CTX_K
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= qk_scale
        
        # Causal mask
        if CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # Load V
        v_ptrs = v_ptr + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (start_n + offs_n[:, None]) < N_CTX_K
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(p.to(q.dtype), v)
        
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    
    # Normalize
    acc = acc / l_i[:, None]
    
    # Handle causal NaN rows
    if CAUSAL:
        causal_start_idx = N_CTX_Q - N_CTX_K
        if causal_start_idx > pid_m * BLOCK_M and causal_start_idx < (pid_m + 1) * BLOCK_M:
            out_mask_boundary = tl.full((BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            nan_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            acc = tl.where(nan_mask, acc, 0.0)
    
    # Store output
    o_ptrs = o_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < N_CTX_Q
    tl.store(o_ptrs, acc.to(q.dtype), mask=o_mask)


VARIANTS = [
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