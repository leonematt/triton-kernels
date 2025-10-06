import pytest
import torch
import nexus
import os

# Test variants - matching your rms_norm VARIANTS
VARIANTS = [
    {'BLOCK_SIZE': 1024, 'EPS': 1e-6},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-6},
    {'BLOCK_SIZE': 4096, 'EPS': 1e-6},
    {'BLOCK_SIZE': 8192, 'EPS': 1e-6},
    {'BLOCK_SIZE': 1024, 'EPS': 1e-5},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-5},
]


@pytest.fixture(scope="module")
def runtime_and_device():
    """Setup runtime and device once for all tests"""
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}_EPS_{v['EPS']}")
def test_rms_norm(runtime_and_device, variant):
    """Test RMS norm kernel with different block sizes and epsilon values"""
    rt, dev = runtime_and_device
    
    BLOCK_SIZE = variant['BLOCK_SIZE']
    EPS = variant['EPS']
    
    # Test dimensions
    batch_size = 32
    hidden_size = 2048
    
    # Skip if block size is smaller than hidden size
    if BLOCK_SIZE < hidden_size:
        pytest.skip(f"BLOCK_SIZE {BLOCK_SIZE} < hidden_size {hidden_size}")
    
    # Create test data on CPU
    x = torch.randn((batch_size, hidden_size), dtype=torch.float32)
    weight = torch.ones(hidden_size, dtype=torch.float32)
    output = torch.zeros((batch_size, hidden_size), dtype=torch.float32)
    
    # Create device buffers
    nb_x = dev.create_buffer(x)
    nb_weight = dev.create_buffer(weight)
    nb_output = dev.create_buffer(output)
    
    # Kernel name format: rms_norm_BLOCK_SIZE_1024_EPS_1e-06
    # Use str() to get Python's default string representation
    eps_str = str(EPS).replace('-', '_')
    kernel_name = f'rms_norm_BLOCK_SIZE_{BLOCK_SIZE}_EPS_{eps_str}'
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
    
    # Set arguments: x_ptr, output_ptr, weight_ptr, n_elements
    cmd.set_arg(0, nb_x)
    cmd.set_arg(1, nb_output)
    cmd.set_arg(2, nb_weight)
    cmd.set_arg(3, hidden_size)
    cmd.set_arg(4, 0)  # metadata pointer
    
    # Grid: one block per row
    grid_size = batch_size
    
    print(f"\nTesting RMS norm BLOCK_SIZE={BLOCK_SIZE}, EPS={EPS}")
    print(f"Grid: [{grid_size}, 1, 1], Block: [128, 1, 1]")
    print(f"Batch: {batch_size}, Hidden: {hidden_size}")
    print(f"Kernel name: {kernel_name}")
    
    cmd.finalize([grid_size, 1, 1], [128, 1, 1])
    
    # Run kernel
    sched.run()
    
    # Copy result back to CPU
    nb_output.copy(output)
    
    # Verify results using PyTorch
    x_squared = x * x
    mean_x_squared = torch.mean(x_squared, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_x_squared + EPS)
    expected = (x / rms) * weight
    
    print(f"Result[0,0]: {output[0, 0]}")
    print(f"Expected[0,0]: {expected[0, 0]}")
    
    max_diff = torch.max(torch.abs(output - expected)).item()
    print(f"Max difference: {max_diff}")
    
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-4), \
        f"Mismatch for BLOCK_SIZE={BLOCK_SIZE}, EPS={EPS}.\n" \
        f"Max diff: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])