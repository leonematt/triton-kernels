#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def test_ptx_mapper():
    """Test the PTX kernel mapper with various inputs"""
    
    # Path to the mapper script
    mapper_script = "triton_kernels/utils/kernel_mapper.py"
    ptx_dir = "kernels"
    
    # Test cases: (kernel_name, values, expected_result_contains)
    test_cases = [
        ("rotary_embedding_kernel", ["256", "False", "True"], "rotary_embedding_kernel_BLOCK_SIZE_ROT256_DO_KEY_ROPEFalse_IS_NEOXTrue"),
        ("rotary_embedding_kernel", ["256", "True", "False"], "rotary_embedding_kernel_BLOCK_SIZE_ROT256_DO_KEY_ROPETrue_IS_NEOXFalse"),
        ("add_vectors", ["256"], "add_vectors_BLOCK_SIZE256"),
        ("add_vectors", ["512"], "add_vectors_BLOCK_SIZE512"),
        ("add_vectors", ["1024"], "add_vectors_BLOCK_SIZE1024"),
    ]
    
    print("Testing PTX Kernel Mapper")
    print("=" * 50)
    
    for i, (kernel_name, values, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {kernel_name} with values {values}")
        
        # Build command
        cmd = ["python3", mapper_script, ptx_dir, kernel_name] + values
        
        try:
            # Run the mapper
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                
                # Check if we got the expected result
                if expected in output:
                    print(f"  ✅ SUCCESS: {output}")
                else:
                    print(f"  ❌ UNEXPECTED: Got '{output}', expected something containing '{expected}'")
            else:
                print(f"  ❌ FAILED: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ TIMEOUT: Command took too long")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    # Test listing kernels
    print(f"\nTest: List all kernels")
    try:
        cmd = ["python3", mapper_script, ptx_dir, "--list"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"  ✅ SUCCESS: Found kernels:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"    {line}")
        else:
            print(f"  ❌ FAILED: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

def test_specific_case():
    """Test the specific case you mentioned"""
    
    mapper_script = "triton_kernels/utils/kernel_mapper.py"
    ptx_dir = "kernels"
    
    print("\nTesting specific case: rotary_embedding_kernel 256 False True")
    print("-" * 60)
    
    cmd = ["python3", mapper_script, ptx_dir, "rotary_embedding_kernel", "256", "False", "True"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout.strip()}")
        
        if result.stderr:
            print(f"Errors: {result.stderr.strip()}")
            
        # Check if we got the expected result
        expected = "rotary_embedding_kernel_BLOCK_SIZE_ROT256_DO_KEY_ROPEFalse_IS_NEOXTrue"
        if expected in result.stdout:
            print(f"✅ SUCCESS: Got expected result!")
        else:
            print(f"❌ UNEXPECTED: Expected '{expected}'")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    # Check if mapper script exists
    mapper_path = Path("triton_kernels/utils/kernel_mapper.py")
    if not mapper_path.exists():
        print(f"Error: Mapper script not found at {mapper_path}")
        print("Make sure you're running from the correct directory")
        sys.exit(1)
    
    # Check if kernels directory exists
    kernels_path = Path("kernels")
    if not kernels_path.exists():
        print(f"Error: Kernels directory not found at {kernels_path}")
        print("Make sure the PTX files are in the 'kernels' directory")
        sys.exit(1)
    
    # Run tests
    test_specific_case()
    test_ptx_mapper()
    
    print(f"\nDone!")