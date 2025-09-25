#!/usr/bin/env python3

import torch
import os
import sys
from pathlib import Path

# Add kernels directory to path
kernels_dir = Path(__file__).parent.parent / "ptx_triton_kernels"
sys.path.insert(0, str(kernels_dir))

def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """PyTorch reference implementation of RMSNorm"""
    # Compute variance (mean of squares)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    # Normalize and apply weight
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight

def test_triton_rms_norm():
    """Test Triton RMSNorm kernels against PyTorch reference"""
    
    # Test configurations
    test_configs = [
        {"batch_size": 2, "seq_len": 128, "hidden_size": 512, "block_size": 1024},
        {"batch_size": 4, "seq_len": 256, "hidden_size": 1024, "block_size": 2048},
        {"batch_size": 1, "seq_len": 512, "hidden_size": 2048, "block_size": 4096},
        {"batch_size": 2, "seq_len": 128, "hidden_size": 4096, "block_size": 8192},
        {"batch_size": 8, "seq_len": 64, "hidden_size": 4096, "block_size": 8192},  # Different batch/seq combo
    ]
    
    dtypes = [torch.float16, torch.bfloat16]
    eps_values = [1e-6, 1e-5, 1e-8]
    
    for dtype in dtypes:
        for eps in eps_values:
            print("=" * 80)
            print(f"RMSNorm Kernel Test: dtype={dtype}, eps={eps}")
            print("=" * 80)
            
            for config in test_configs:
                batch_size = config["batch_size"]
                seq_len = config["seq_len"]
                hidden_size = config["hidden_size"]
                block_size = config["block_size"]
                
                print(f"\nConfiguration: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")
                print(f"Block size: {block_size}, dtype: {dtype}")
                
                # Skip if block size is smaller than hidden size (kernel limitation)
                if block_size < hidden_size:
                    print(f"   ⏭️  SKIPPED: BLOCK_SIZE ({block_size}) < HIDDEN_SIZE ({hidden_size})")
                    continue
                
                # Create test tensors
                try:
                    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
                    weight = torch.ones(hidden_size, device='cuda', dtype=dtype)
                    
                    # Add some variation to weights for more realistic testing
                    weight = weight + 0.1 * torch.randn_like(weight)
                    
                except RuntimeError as e:
                    print(f"   ❌ FAILED: Could not create tensors - {e}")
                    continue
                
                print(f"   Input shape: {x.shape}")
                print(f"   Weight shape: {weight.shape}")
                print(f"   Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
                
                # 1. PyTorch reference
                print(f"\n   1. PyTorch Reference:")
                try:
                    expected = pytorch_rms_norm(x, weight, eps)
                    print(f"      Output mean: {expected.mean().item():.6f}")
                    print(f"      Output std: {expected.std().item():.6f}")
                except Exception as e:
                    print(f"      ❌ FAILED: {e}")
                    continue
                
                # 2. Test basic Triton kernel
                print(f"\n   2. Triton RMSNorm Kernel:")
                try:
                    # Flatten input for kernel processing  
                    x_flat = x.view(-1, hidden_size)
                    n_rows = x_flat.shape[0]
                    out_flat = torch.empty_like(x_flat)
                    
                    # Launch kernel
                    grid = (n_rows,)
                    rms_norm_kernel[grid](
                        x_flat, weight, out_flat, n_rows, eps, 
                        BLOCK_SIZE=block_size, HIDDEN_SIZE=hidden_size
                    )
                    
                    # Reshape back
                    output_triton = out_flat.view(batch_size, seq_len, hidden_size)
                    
                    print(f"      Output mean: {output_triton.mean().item():.6f}")
                    print(f"      Output std: {output_triton.std().item():.6f}")
                    
                    # Compare with reference
                    diff = torch.abs(output_triton - expected)
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    
                    print(f"      Max difference: {max_diff:.2e}")
                    print(f"      Mean difference: {mean_diff:.2e}")
                    
                    # Relative error
                    rel_error = diff / (torch.abs(expected) + 1e-8)
                    max_rel_error = rel_error.max().item()
                    print(f"      Max relative error: {max_rel_error:.2%}")
                    
                    # Set tolerance based on dtype and hidden size
                    if dtype == torch.bfloat16:
                        tolerance = 0.02 if hidden_size >= 4096 else 0.01
                    else:  # float16
                        tolerance = 0.01 if hidden_size >= 4096 else 0.005
                    
                    print(f"      Tolerance: {tolerance}")
                    
                    if max_diff < tolerance:
                        print(f"      ✅ PASSED (within tolerance)")
                    else:
                        print(f"      ❌ FAILED (exceeds tolerance)")
                        
                except Exception as e:
                    print(f"      ❌ FAILED: {e}")
                
                # 3. Test fused kernel (if different from basic)
                print(f"\n   3. Triton Fused RMSNorm Kernel:")
                try:
                    # Flatten input for kernel processing  
                    x_flat_fused = x.view(-1, hidden_size)
                    out_flat_fused = torch.empty_like(x_flat_fused)
                    
                    # Launch fused kernel
                    grid = (n_rows,)
                    rms_norm_fused_kernel[grid](
                        x_flat_fused, weight, out_flat_fused, n_rows, eps,
                        BLOCK_SIZE=block_size, HIDDEN_SIZE=hidden_size
                    )
                    
                    # Reshape back
                    output_fused = out_flat_fused.view(batch_size, seq_len, hidden_size)
                    
                    print(f"      Output mean: {output_fused.mean().item():.6f}")
                    print(f"      Output std: {output_fused.std().item():.6f}")
                    
                    # Compare with reference
                    diff_fused = torch.abs(output_fused - expected)
                    max_diff_fused = diff_fused.max().item()
                    mean_diff_fused = diff_fused.mean().item()
                    
                    print(f"      Max difference: {max_diff_fused:.2e}")
                    print(f"      Mean difference: {mean_diff_fused:.2e}")
                    
                    # Compare kernels (should be identical in current implementation)
                    if 'output_triton' in locals():
                        kernel_diff = torch.abs(output_triton - output_fused).max().item()
                        print(f"      Kernel consistency: {kernel_diff:.2e}")
                        if kernel_diff < 1e-8:
                            print(f"      ✅ Kernels produce identical results")
                        else:
                            print(f"      ⚠️  Kernels differ (expected if implementations differ)")
                    
                    if max_diff_fused < tolerance:
                        print(f"      ✅ PASSED (within tolerance)")
                    else:
                        print(f"      ❌ FAILED (exceeds tolerance)")
                        
                except Exception as e:
                    print(f"      ❌ FAILED: {e}")
                
                # 4. Edge case testing for specific configuration
                if hidden_size == 4096 and dtype == torch.float16:
                    print(f"\n   4. Edge Case Testing:")
                    try:
                        # Test with zeros
                        x_zeros = torch.zeros_like(x)
                        expected_zeros = pytorch_rms_norm(x_zeros, weight, eps)
                        
                        x_flat_zeros = x_zeros.view(-1, hidden_size)
                        out_flat_zeros = torch.empty_like(x_flat_zeros)
                        
                        rms_norm_kernel[grid](
                            x_flat_zeros, weight, out_flat_zeros, n_rows, eps,
                            BLOCK_SIZE=block_size, HIDDEN_SIZE=hidden_size
                        )
                        
                        output_zeros = out_flat_zeros.view_as(x_zeros)
                        zeros_diff = torch.abs(output_zeros - expected_zeros).max().item()
                        
                        print(f"      Zero input test - Max diff: {zeros_diff:.2e}")
                        print(f"      ✅ Zero input handled correctly" if zeros_diff < tolerance else "      ❌ Zero input failed")
                        
                        # Test with very small values
                        x_small = torch.full_like(x, 1e-4)
                        expected_small = pytorch_rms_norm(x_small, weight, eps)
                        
                        x_flat_small = x_small.view(-1, hidden_size)
                        out_flat_small = torch.empty_like(x_flat_small)
                        
                        rms_norm_kernel[grid](
                            x_flat_small, weight, out_flat_small, n_rows, eps,
                            BLOCK_SIZE=block_size, HIDDEN_SIZE=hidden_size
                        )
                        
                        output_small = out_flat_small.view_as(x_small)
                        small_diff = torch.abs(output_small - expected_small).max().item()
                        
                        print(f"      Small input test - Max diff: {small_diff:.2e}")
                        print(f"      ✅ Small input handled correctly" if small_diff < tolerance else "      ❌ Small input failed")
                        
                    except Exception as e:
                        print(f"      ❌ Edge case testing failed: {e}")
                
                print("-" * 60)
            
            print("\n" + "=" * 80 + "\n")

def benchmark_kernels():
    """Simple benchmark comparing PyTorch vs Triton implementations"""
    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    batch_size = 8
    seq_len = 512
    hidden_size = 4096
    block_size = 8192
    eps = 1e-6
    dtype = torch.float16
    
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
    weight = torch.ones(hidden_size, device='cuda', dtype=dtype)
    
    # Warmup
    for _ in range(10):
        _ = pytorch_rms_norm(x, weight, eps)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    import time
    start = time.time()
    for _ in range(100):
        _ = pytorch_rms_norm(x, weight, eps)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    # Benchmark Triton
    x_flat = x.view(-1, hidden_size)
    n_rows = x_flat.shape[0]
    out_flat = torch.empty_like(x_flat)
    grid = (n_rows,)
    
    # Warmup
    for _ in range(10):
        rms_norm_kernel[grid](
            x_flat, weight, out_flat, n_rows, eps,
            BLOCK_SIZE=block_size, HIDDEN_SIZE=hidden_size
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        rms_norm_kernel[grid](
            x_flat, weight, out_flat, n_rows, eps,
            BLOCK_SIZE=block_size, HIDDEN_SIZE=hidden_size
        )
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    print(f"Configuration: {batch_size}x{seq_len}x{hidden_size}, dtype={dtype}")
    print(f"PyTorch time: {pytorch_time*1000:.2f} ms")
    print(f"Triton time: {triton_time*1000:.2f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")

if __name__ == "__main__":
    try:
        from triton_kernels.layernorm import rms_norm_kernel, rms_norm_fused_kernel
    except ImportError as e:
        print(f"Could not import RMSNorm kernels: {e}")
        print("Make sure layernorm.py is in the kernels directory.")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("This test requires a CUDA-enabled GPU.")
        sys.exit(1)

    print("Starting RMSNorm Kernel Tests...\n")
    test_triton_rms_norm()
    
    print("\nRunning performance benchmark...\n")
    try:
        benchmark_kernels()
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    print("\nTesting complete!")