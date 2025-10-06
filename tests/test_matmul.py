import pytest
import torch
import nexus
import os

# Test variants - matching your matmul VARIANTS
VARIANTS = [
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
    {'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
]


@pytest.fixture(scope="module")
def runtime_and_device():
    """Setup runtime and device once for all tests"""
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"M{v['BLOCK_SIZE_M']}_N{v['BLOCK_SIZE_N']}_K{v['BLOCK_SIZE_K']}")
def test_matmul(runtime_and_device, variant):
    """Test matmul kernel with different block sizes"""
    rt, dev = runtime_and_device
    
    BLOCK_K = variant['BLOCK_SIZE_K']
    BLOCK_M = variant['BLOCK_SIZE_M']
    BLOCK_N = variant['BLOCK_SIZE_N']
    
    # Matrix dimensions
    M, N, K = 512, 512, 512
    
    # Create test data on CPU
    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)
    c = torch.zeros((M, N), dtype=torch.float32)

    a_tile_size = BLOCK_M * BLOCK_K * 4  # 4 bytes per float
    b_tile_size = BLOCK_K * BLOCK_N * 4
    shared_mem = 2 * (a_tile_size + b_tile_size)

    # Create device buffers
    nb_a = dev.create_buffer(a)
    nb_b = dev.create_buffer(b)
    nb_c = dev.create_buffer(c)
    
    # Kernel name format: matmul_BLOCK_SIZE_K_16_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16
    kernel_name = f'matmul_BLOCK_SIZE_K_{BLOCK_K}_BLOCK_SIZE_M_{BLOCK_M}_BLOCK_SIZE_N_{BLOCK_N}'
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    
    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")
    
    try:
        lib = dev.load_library(lib_path)
        kern = lib.get_kernel(kernel_name)
    except Exception as e:
        pytest.fail(f"Failed to load kernel {kernel_name}: {e}")
    
    # Create and configure command
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    # Strides (row-major)
    stride_ak = 1
    stride_am = K
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1
    
    # Set arguments matching the kernel signature:
    # a_ptr, b_ptr, c_ptr, K, M, N, stride_ak, stride_am, stride_bk, stride_bn, stride_cm, stride_cn
    cmd.set_arg(0, nb_a)          # a_ptr
    cmd.set_arg(1, nb_b)          # b_ptr
    cmd.set_arg(2, nb_c)          # c_ptr
    cmd.set_arg(3, K)             # K
    cmd.set_arg(4, M)             # M
    cmd.set_arg(5, N)             # N
    cmd.set_arg(6, stride_ak)     # stride_ak
    cmd.set_arg(7, stride_am)     # stride_am
    cmd.set_arg(8, stride_bk)     # stride_bk
    cmd.set_arg(9, stride_bn)     # stride_bn
    cmd.set_arg(10, stride_cm)    # stride_cm
    cmd.set_arg(11, stride_cn)    # stride_cn
    cmd.set_arg(12, 0)            # metadata pointer
    
    # Calculate grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_size = grid_m * grid_n
    
    print(f"\nTesting matmul M={BLOCK_M}, N={BLOCK_N}, K={BLOCK_K}")
    print(f"Grid: [{grid_size}, 1, 1], Block: [128, 1, 1]")
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"Kernel name: {kernel_name}")
    
    cmd.finalize([grid_size, 1, 1], [128, 1, 1], shared_mem)
    
    # Run kernel
    sched.run()
    
    # Copy result back to CPU
    nb_c.copy(c)
    
    # Verify results using PyTorch
    expected = torch.matmul(a, b)
    
    print(f"Result[0,0]: {c[0, 0]}")
    print(f"Expected[0,0]: {expected[0, 0]}")
    
    assert torch.allclose(c, expected, rtol=1e-1, atol=1e-1), \
        f"Mismatch for M={BLOCK_M}, N={BLOCK_N}, K={BLOCK_K}.\n" \
        f"Max diff: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])