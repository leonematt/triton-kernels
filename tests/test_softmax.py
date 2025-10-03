#!/usr/bin/env python3

import pytest
import torch
import nexus
import os

# Test variants
VARIANTS = [
    {'BLOCK_SIZE': 128},
    {'BLOCK_SIZE': 256},
    {'BLOCK_SIZE': 512},
    {'BLOCK_SIZE': 1024},
]

# Test shapes: (n_rows, n_cols)
TEST_SHAPES = [
    (128, 256),
    (256, 512),
    (512, 1024),
    (64, 128),
]


@pytest.fixture(scope="module")
def runtime_and_device():
    """Setup runtime and device once for all tests"""
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_SIZE{v['BLOCK_SIZE']}")
@pytest.mark.parametrize("shape", TEST_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_softmax_kernel(runtime_and_device, variant, shape):
    """Test softmax kernel with different block sizes and shapes"""
    rt, dev = runtime_and_device
    block_size = variant['BLOCK_SIZE']
    n_rows, n_cols = shape
    
    # Create test data
    input_tensor = torch.randn(n_rows, n_cols, dtype=torch.float32)
    output = torch.zeros(n_rows, n_cols, dtype=torch.float32)
    
    # Create device buffers
    nb_input = dev.create_buffer(input_tensor)
    nb_output = dev.create_buffer(output)
    
    # Load kernel
    kernel_name = f'softmax_kernel_BLOCK_SIZE{block_size}'
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    
    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")
    
    try:
        lib = dev.load_library_file(lib_path)
        kern = lib.get_kernel(kernel_name)
    except Exception as e:
        pytest.fail(f"Failed to load kernel {kernel_name}: {e}")
    
    # Create and configure command
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    # Set kernel arguments
    cmd.set_arg(0, nb_input)              # input_ptr
    cmd.set_arg(1, nb_output)             # output_ptr
    cmd.set_arg(2, n_rows)                # n_rows
    cmd.set_arg(3, n_cols)                # n_cols
    cmd.set_arg(4, input_tensor.stride(0)) # input_row_stride
    cmd.set_arg(5, output.stride(0))      # output_row_stride
    
    # Grid configuration: one block per row
    grid_size = n_rows
    # Threads per block can be adjusted based on block size
    threads_per_block = min(block_size, 1024)  # Max 1024 threads per block
    
    print(f"\nTesting softmax with BLOCK_SIZE={block_size}, shape={shape}")
    print(f"Grid: [{grid_size}, 1, 1], Block: [{threads_per_block}, 1, 1]")
    
    cmd.finalize([grid_size, 1, 1], [threads_per_block, 1, 1])
    
    # Run kernel
    sched.run()
    
    # Copy result back
    nb_output.copy(output)
    
    # Compute expected result with PyTorch
    expected = torch.softmax(input_tensor, dim=-1)
    
    print(f"Result (first row, first 5):   {output[0, :5].tolist()}")
    print(f"Expected (first row, first 5): {expected[0, :5].tolist()}")
    
    # Check correctness
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-5), \
        f"Mismatch for softmax with BLOCK_SIZE={block_size}, shape={shape}.\n" \
        f"Max diff: {torch.max(torch.abs(output - expected)).item()}\n" \
        f"First row output: {output[0, :10]}\n" \
        f"First row expected: {expected[0, :10]}"
    
    # Additional check: verify row sums equal 1
    row_sums = output.sum(dim=-1)
    ones = torch.ones(n_rows, dtype=torch.float32)
    assert torch.allclose(row_sums, ones, rtol=1e-4, atol=1e-5), \
        f"Row sums not equal to 1.0 for BLOCK_SIZE={block_size}, shape={shape}.\n" \
        f"Max deviation: {torch.max(torch.abs(row_sums - 1.0)).item()}"


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_SIZE{v['BLOCK_SIZE']}")
def test_softmax_numerical_stability(runtime_and_device, variant):
    """Test numerical stability with extreme values"""
    rt, dev = runtime_and_device
    block_size = variant['BLOCK_SIZE']
    n_rows, n_cols = 128, 256
    
    # Test with large values that could cause overflow
    input_tensor = torch.randn(n_rows, n_cols, dtype=torch.float32) * 100
    output = torch.zeros(n_rows, n_cols, dtype=torch.float32)
    
    # Create device buffers
    nb_input = dev.create_buffer(input_tensor)
    nb_output = dev.create_buffer(output)
    
    # Load kernel
    kernel_name = f'softmax_kernel_BLOCK_SIZE{block_size}'
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    
    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")
    
    lib = dev.load_library_file(lib_path)
    kern = lib.get_kernel(kernel_name)
    
    # Create and configure command
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    cmd.set_arg(0, nb_input)
    cmd.set_arg(1, nb_output)
    cmd.set_arg(2, n_rows)
    cmd.set_arg(3, n_cols)
    cmd.set_arg(4, input_tensor.stride(0))
    cmd.set_arg(5, output.stride(0))
    
    grid_size = n_rows
    threads_per_block = min(block_size, 1024)
    
    cmd.finalize([grid_size, 1, 1], [threads_per_block, 1, 1])
    sched.run()
    nb_output.copy(output)
    
    # Check for NaN or Inf
    assert not torch.isnan(output).any(), \
        f"NaN detected in output for BLOCK_SIZE={block_size} with large values"
    assert not torch.isinf(output).any(), \
        f"Inf detected in output for BLOCK_SIZE={block_size} with large values"
    
    # Verify against PyTorch
    expected = torch.softmax(input_tensor, dim=-1)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-5), \
        f"Numerical stability test failed for BLOCK_SIZE={block_size}"
    
    print(f"Numerical stability test PASSED for BLOCK_SIZE={block_size}")


@pytest.mark.parametrize("variant", VARIANTS[:2], ids=lambda v: f"BLOCK_SIZE{v['BLOCK_SIZE']}")
def test_softmax_edge_cases(runtime_and_device, variant):
    """Test edge cases like single row or non-power-of-2 dimensions"""
    rt, dev = runtime_and_device
    block_size = variant['BLOCK_SIZE']
    
    # Test single row
    n_rows, n_cols = 1, 256
    input_tensor = torch.randn(n_rows, n_cols, dtype=torch.float32)
    output = torch.zeros(n_rows, n_cols, dtype=torch.float32)
    
    nb_input = dev.create_buffer(input_tensor)
    nb_output = dev.create_buffer(output)
    
    kernel_name = f'softmax_kernel_BLOCK_SIZE{block_size}'
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    
    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")
    
    lib = dev.load_library_file(lib_path)
    kern = lib.get_kernel(kernel_name)
    
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    cmd.set_arg(0, nb_input)
    cmd.set_arg(1, nb_output)
    cmd.set_arg(2, n_rows)
    cmd.set_arg(3, n_cols)
    cmd.set_arg(4, input_tensor.stride(0))
    cmd.set_arg(5, output.stride(0))
    
    cmd.finalize([n_rows, 1, 1], [min(block_size, 1024), 1, 1])
    sched.run()
    nb_output.copy(output)
    
    expected = torch.softmax(input_tensor, dim=-1)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-5), \
        f"Single row test failed for BLOCK_SIZE={block_size}"
    
    print(f"Edge case (single row) test PASSED for BLOCK_SIZE={block_size}")
    
    # Test non-power-of-2 dimensions
    n_rows, n_cols = 100, 300
    input_tensor = torch.randn(n_rows, n_cols, dtype=torch.float32)
    output = torch.zeros(n_rows, n_cols, dtype=torch.float32)
    
    nb_input = dev.create_buffer(input_tensor)
    nb_output = dev.create_buffer(output)
    
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    cmd.set_arg(0, nb_input)
    cmd.set_arg(1, nb_output)
    cmd.set_arg(2, n_rows)
    cmd.set_arg(3, n_cols)
    cmd.set_arg(4, input_tensor.stride(0))
    cmd.set_arg(5, output.stride(0))
    
    cmd.finalize([n_rows, 1, 1], [min(block_size, 1024), 1, 1])
    sched.run()
    nb_output.copy(output)
    
    expected = torch.softmax(input_tensor, dim=-1)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-5), \
        f"Non-power-of-2 test failed for BLOCK_SIZE={block_size}"
    
    print(f"Edge case (non-power-of-2) test PASSED for BLOCK_SIZE={block_size}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])