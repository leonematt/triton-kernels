import pytest
import torch
import nexus
import os

# Test variants - matching the silu_and_mul VARIANTS
VARIANTS = [
    {'BLOCK_SIZE': 1024},
    {'BLOCK_SIZE': 2048},
    {'BLOCK_SIZE': 4096},
    {'BLOCK_SIZE': 8192},
    {'BLOCK_SIZE': 16384},
]


@pytest.fixture(scope="module")
def runtime_and_device():
    """Setup runtime and device once for all tests"""
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}")
def test_silu_and_mul(runtime_and_device, variant):
    """Test SiluAndMul kernel with different block sizes"""
    rt, dev = runtime_and_device
    
    BLOCK_SIZE = variant['BLOCK_SIZE']
    
    # Test dimensions
    batch_size = 32
    hidden_size = 2048
    full_size = hidden_size * 2  # Input is 2x hidden size
    
    # Skip if block size is smaller than hidden size
    if BLOCK_SIZE < hidden_size:
        pytest.skip(f"BLOCK_SIZE {BLOCK_SIZE} < hidden_size {hidden_size}")
    
    # Create test data on CPU
    x = torch.randn((batch_size, full_size), dtype=torch.float32)
    output = torch.zeros((batch_size, hidden_size), dtype=torch.float32)
    
    # Create device buffers
    nb_x = dev.create_buffer(x)
    nb_output = dev.create_buffer(output)
    
    # Kernel name format: silu_and_mul_BLOCK_SIZE_1024
    kernel_name = f'silu_and_mul_BLOCK_SIZE_{BLOCK_SIZE}'
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
    
    # Set arguments: x_ptr, output_ptr, n_elements
    cmd.set_arg(0, nb_x)
    cmd.set_arg(1, nb_output)
    cmd.set_arg(2, full_size)  # n_elements is the full size
    cmd.set_arg(3, 0)  # metadata pointer
    
    # Grid: one block per row
    grid_size = batch_size
    
    print(f"\nTesting SiluAndMul BLOCK_SIZE={BLOCK_SIZE}")
    print(f"Grid: [{grid_size}, 1, 1], Block: [128, 1, 1]")
    print(f"Batch: {batch_size}, Hidden: {hidden_size}, Full: {full_size}")
    print(f"Kernel name: {kernel_name}")
    
    cmd.finalize([grid_size, 1, 1], [128, 1, 1])
    
    # Run kernel
    sched.run()
    
    # Copy result back to CPU
    nb_output.copy(output)
    
    # Verify results using PyTorch
    gate = x[:, :hidden_size]
    up = x[:, hidden_size:]
    expected = torch.nn.functional.silu(gate) * up
    
    print(f"Result[0,0]: {output[0, 0]}")
    print(f"Expected[0,0]: {expected[0, 0]}")
    print(f"Result[0,:5]: {output[0, :5].tolist()}")
    print(f"Expected[0,:5]: {expected[0, :5].tolist()}")
    
    max_diff = torch.max(torch.abs(output - expected)).item()
    mean_diff = torch.mean(torch.abs(output - expected)).item()
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-4), \
        f"Mismatch for BLOCK_SIZE={BLOCK_SIZE}.\n" \
        f"Max diff: {max_diff:.6e}\n" \
        f"Mean diff: {mean_diff:.6e}"
    
    print("✅ PASSED")


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}_FP16")
def test_silu_and_mul_fp16(runtime_and_device, variant):
    """Test SiluAndMul kernel with FP16 data"""
    rt, dev = runtime_and_device
    
    BLOCK_SIZE = variant['BLOCK_SIZE']
    
    # Test dimensions
    batch_size = 32
    hidden_size = 2048
    full_size = hidden_size * 2
    
    if BLOCK_SIZE < hidden_size:
        pytest.skip(f"BLOCK_SIZE {BLOCK_SIZE} < hidden_size {hidden_size}")
    
    # Create test data in FP16
    x_fp16 = torch.randn((batch_size, full_size), dtype=torch.float16)
    x = x_fp16.float()  # Convert to FP32 for kernel
    output = torch.zeros((batch_size, hidden_size), dtype=torch.float32)
    
    # Create device buffers
    nb_x = dev.create_buffer(x)
    nb_output = dev.create_buffer(output)
    
    kernel_name = f'silu_and_mul_kernel_BLOCK_SIZE_{BLOCK_SIZE}'
    lib_path = f"ptx_kernels/{kernel_name}.ptx"
    
    if not os.path.exists(lib_path):
        pytest.skip(f"Kernel file not found: {lib_path}")
    
    try:
        lib = dev.load_library(lib_path)
        kern = lib.get_kernel(kernel_name)
    except Exception as e:
        pytest.fail(f"Failed to load kernel {kernel_name}: {e}")
    
    sched = dev.create_schedule()
    cmd = sched.create_command(kern)
    
    cmd.set_arg(0, nb_x)
    cmd.set_arg(1, nb_output)
    cmd.set_arg(2, full_size)
    cmd.set_arg(3, 0)
    
    grid_size = batch_size
    
    print(f"\nTesting SiluAndMul FP16 BLOCK_SIZE={BLOCK_SIZE}")
    print(f"Batch: {batch_size}, Hidden: {hidden_size}")
    
    cmd.finalize([grid_size, 1, 1], [128, 1, 1])
    sched.run()
    nb_output.copy(output)
    
    # Convert back to FP16
    output_fp16 = output.half()
    
    # Verify with FP16 reference
    gate = x_fp16[:, :hidden_size]
    up = x_fp16[:, hidden_size:]
    expected_fp16 = (torch.nn.functional.silu(gate) * up)
    
    max_diff = torch.max(torch.abs(output_fp16.float() - expected_fp16.float())).item()
    print(f"Max difference (FP16): {max_diff:.6e}")
    
    # Relaxed tolerance for FP16
    assert torch.allclose(output_fp16, expected_fp16, rtol=1e-2, atol=1e-3), \
        f"Mismatch for BLOCK_SIZE={BLOCK_SIZE} (FP16). Max diff: {max_diff:.6e}"
    
    print("✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])