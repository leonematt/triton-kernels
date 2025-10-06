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


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_SIZE_{v['BLOCK_SIZE']}")
@pytest.mark.parametrize("shape", TEST_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_softmax(runtime_and_device, variant, shape):
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
    
    # Kernel name format: softmax_BLOCK_SIZE_128 (no _kernel suffix)
    kernel_name = f'softmax_BLOCK_SIZE_{block_size}'
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
    
    # Set kernel arguments
    cmd.set_arg(0, nb_input)              # input_ptr
    cmd.set_arg(1, nb_output)             # output_ptr
    cmd.set_arg(2, n_rows)                # n_rows
    cmd.set_arg(3, n_cols)                # n_cols
    cmd.set_arg(4, input_tensor.stride(0)) # input_row_stride
    cmd.set_arg(5, output.stride(0))      # output_row_stride
    cmd.set_arg(6, 0)                     # metadata pointer
    
    # Grid configuration: one block per row
    grid_size = n_rows
    
    print(f"\nTesting softmax BLOCK_SIZE={block_size}, shape={shape}")
    print(f"Grid: [{grid_size}, 1, 1], Block: [128, 1, 1]")
    
    cmd.finalize([grid_size, 1, 1], [128, 1, 1])
    
    # Run kernel
    sched.run()
    
    # Copy result back
    nb_output.copy(output)
    
    # Compute expected result with PyTorch
    expected = torch.softmax(input_tensor, dim=-1)
    
    print(f"Result (first row, first 5):   {output[0, :5].tolist()}")
    print(f"Expected (first row, first 5): {expected[0, :5].tolist()}")
    
    max_diff = torch.max(torch.abs(output - expected)).item()
    print(f"Max diff: {max_diff}")
    
    # Check correctness
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-5), \
        f"Mismatch for softmax BLOCK_SIZE={block_size}, shape={shape}. Max diff: {max_diff}"
    
    # Verify row sums equal 1
    row_sums = output.sum(dim=-1)
    ones = torch.ones(n_rows, dtype=torch.float32)
    assert torch.allclose(row_sums, ones, rtol=1e-4, atol=1e-5), \
        f"Row sums not equal to 1.0. Max deviation: {torch.max(torch.abs(row_sums - 1.0)).item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])