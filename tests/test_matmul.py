import pytest
import torch
import nexus
import os

def test_matmul_simple():
    """Test single matmul configuration for debugging"""
    # Simple configuration
# Test with smallest possible case first
    M, N, K = 64, 64, 64
    block_m, block_n, block_k = 64, 64, 16

    # This should need only 1 block: grid_size = 1

    # Also try with num_warps matching what kernel expects
    threads_per_block = 32  # num_warps=1 → 1 warp = 32 threads
    
    print(f"\n=== SIMPLE MATMUL TEST ===")
    print(f"Matrix size: {M}x{N}x{K}")
    print(f"Block size: {block_m}x{block_n}x{block_k}")
    
    # Setup runtime
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    print(f"Device: {dev}")
    
    # Create test data
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)
    c = torch.zeros((M, N), dtype=torch.float32)
    print(f"Created tensors: a={a.shape}, b={b.shape}, c={c.shape}")
    print(f"Strides: a={a.stride()}, b={b.stride()}, c={c.stride()}")
    
    # Create buffers
    nb_a = dev.create_buffer(a)
    nb_b = dev.create_buffer(b)
    nb_c = dev.create_buffer(c)
    print(f"Created device buffers")
    
    # Load kernel - use the Triton-generated name
    kernel_name = f'matmul_nontiled_kernel_BLOCK_SIZE_M{block_m}_BLOCK_SIZE_N{block_n}_BLOCK_SIZE_K{block_k}'
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    print(f"Loading: {lib_path}")
    
    if not os.path.exists(lib_path):
        print(f"ERROR: File not found!")
        print(f"Available PTX files:")
        if os.path.exists("ptx_kernels"):
            for f in os.listdir("ptx_kernels"):
                if f.endswith(".ptx"):
                    print(f"  {f}")
        return
    
    lib = dev.load_library_file(lib_path)
    kern = lib.get_kernel(kernel_name)
    print(f"Loaded kernel")
    
    # Setup command
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    # Set arguments (10 parameters total based on kernel signature)
    # Pointers
    cmd.set_arg(0, nb_a)      # a_ptr
    cmd.set_arg(1, nb_b)      # b_ptr
    cmd.set_arg(2, nb_c)      # c_ptr
    # Dimensions
    cmd.set_arg(3, M)         # M
    cmd.set_arg(4, N)         # N
    cmd.set_arg(5, K)         # K
    # Strides
    cmd.set_arg(6, a.stride(0))  # stride_am
    cmd.set_arg(7, a.stride(1))  # stride_ak
    cmd.set_arg(8, b.stride(0))  # stride_bk
    cmd.set_arg(9, b.stride(1))  # stride_bn
    # Note: c strides are handled inside the kernel based on the PTX
    
    print(f"Set 10 arguments")
    
    # Grid setup - matches Triton's grid calculation
    grid_m = (M + block_m - 1) // block_m
    grid_n = (N + block_n - 1) // block_n
    grid_size = grid_m * grid_n
    threads_per_block = 128  # num_warps=1 means 32 threads, but let's use 128 to be safe
    
    print(f"Grid: {grid_size} blocks ({grid_m}x{grid_n}), {threads_per_block} threads/block")
    
    cmd.finalize([grid_size, 1, 1], [threads_per_block, 1, 1])
    print(f"Command finalized")
    
    # Run
    sched.run()
    print(f"Kernel executed")
    
    # Get result
    nb_c.copy(c)
    print(f"Copied result back")
    
    # Check
    expected = torch.matmul(a, b)
    print(f"\nResult sample (first row):")
    print(f"  Got:      {c[0, :5]}")
    print(f"  Expected: {expected[0, :5]}")
    max_diff = torch.max(torch.abs(c - expected)).item()
    mean_diff = torch.mean(torch.abs(c - expected)).item()
    print(f"\nDifferences:")
    print(f"  Max:  {max_diff}")
    print(f"  Mean: {mean_diff}")
    print(f"  All zeros? {torch.all(c == 0).item()}")
    
    is_correct = torch.allclose(c, expected, rtol=5e-3, atol=1e-1)
    print(f"\nTest passed: {is_correct}")
    
    if not is_correct:
        print(f"\nDEBUG INFO:")
        print(f"  Result min/max: {c.min().item():.6f} / {c.max().item():.6f}")
        print(f"  Expected min/max: {expected.min().item():.6f} / {expected.max().item():.6f}")
        print(f"  Result norm: {torch.norm(c).item():.6f}")
        print(f"  Expected norm: {torch.norm(expected).item():.6f}")
    
    assert is_correct, f"Results don't match! Max diff: {max_diff}"

if __name__ == "__main__":
    test_matmul_simple()