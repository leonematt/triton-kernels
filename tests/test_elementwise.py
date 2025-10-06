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

# Operations to test
OPERATIONS = ['add', 'subtract', 'multiply', 'divide']


@pytest.fixture(scope="module")
def runtime_and_device():
    """Setup runtime and device once for all tests"""
    rt = nexus.get_runtime("cuda")
    dev = rt.get_devices()[0]
    return rt, dev


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_SIZE_{v['BLOCK_SIZE']}")
@pytest.mark.parametrize("operation", OPERATIONS)
def test_elementwise_operation(runtime_and_device, variant, operation):
    """Test elementwise operations with different block sizes"""
    rt, dev = runtime_and_device
    block_size = variant['BLOCK_SIZE']
    n_elements = 1024
    
    # Create test data on CPU
    buf0 = torch.full((n_elements,), 3.0, dtype=torch.float32)
    buf1 = torch.ones(n_elements, dtype=torch.float32)
    res = torch.zeros(n_elements, dtype=torch.float32)
    
    # Create device buffers
    nb0 = dev.create_buffer(buf0)
    nb1 = dev.create_buffer(buf1)
    nb_res = dev.create_buffer(res)
    
    # The actual kernel name inside PTX matches the Triton function name
    kernel_name = f'elementwise_{operation}_BLOCK_SIZE_{block_size}'
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
    
    cmd.set_arg(0, nb0)          # a_ptr
    cmd.set_arg(1, nb1)          # b_ptr
    cmd.set_arg(2, nb_res)       # output_ptr
    cmd.set_arg(3, n_elements)   # n_elements
    cmd.set_arg(4, 0)
    # Calculate grid size based on block_size
    grid_size = (n_elements + block_size - 1) // block_size
    
    print(f"\nTesting {operation} with kernel BLOCK_SIZE={block_size}")
    print(f"Grid: [{grid_size}, 1, 1], Block: [128, 1, 1]")
    print(f"Kernel name: {kernel_name}")
    print(f"Library path: {lib_path}")
    
    cmd.finalize([grid_size, 1, 1], [128, 1, 1])
    
    # Run kernel and synchronize
    sched.run()
    
    # Copy result back to CPU
    nb_res.copy(res)
    
    # Verify results based on operation
    expected = _compute_expected(buf0, buf1, operation)
    
    print(f"Result (first 5): {res[:5].tolist()}")
    print(f"Expected (first 5): {expected[:5].tolist()}")
    
    assert torch.allclose(res, expected, rtol=1e-5, atol=1e-5), \
        f"Mismatch for {operation} with BLOCK_SIZE={block_size}.\n" \
        f"Got: {res[:10]}\nExpected: {expected[:10]}"


def _compute_expected(a, b, operation):
    """Compute expected results for verification"""
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])