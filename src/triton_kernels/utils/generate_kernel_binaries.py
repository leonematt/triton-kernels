#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
from pathlib import Path
import time
import re
import sys
import importlib.util
import shutil
import os

def clear_triton_cache():
  cache_dir = Path.home() / ".triton" / "cache"
  
  if cache_dir.exists():
    try:
      shutil.rmtree(cache_dir)
      print(f"  Cleared Triton cache: {cache_dir}")
    except Exception as e:
      print(f"  Warning: Could not clear cache: {e}")
  else:
    print(f"  No cache directory found at: {cache_dir}")

def get_kernel_cache_info(kernel, constants):
  try:
    import base64
    import hashlib
    
    cache_root = Path.home() / ".triton" / "cache"
    
    compiled_kernel = kernel[constants]
    
    cache_key = None
    metadata_hash = None
    
    if hasattr(compiled_kernel, 'metadata') and hasattr(compiled_kernel.metadata, 'hash'):
      metadata_hash = compiled_kernel.metadata.hash
      
      try:
        hash_bytes = bytes.fromhex(metadata_hash)
        cache_key = base64.b32encode(hash_bytes).decode('ascii').rstrip('=')
      except Exception as e:
        pass
    
    if not cache_key:
      if hasattr(compiled_kernel, 'cache_key'):
        cache_key = compiled_kernel.cache_key
      elif hasattr(compiled_kernel, 'asm') and hasattr(compiled_kernel.asm, 'hash'):
        cache_key = compiled_kernel.asm.hash
      elif hasattr(compiled_kernel, 'fn') and hasattr(compiled_kernel.fn, 'cache_key'):
        cache_key = compiled_kernel.fn.cache_key
    
    if cache_key and cache_root.exists():
      cache_dir_path = cache_root / cache_key
      if cache_dir_path.exists():
        ptx_files = list(cache_dir_path.glob("*.ptx"))
        if ptx_files:
          return cache_dir_path, ptx_files[0], cache_key
      
      for cache_dir in cache_root.iterdir():
        if cache_dir.is_dir() and (cache_key in cache_dir.name or cache_dir.name.startswith(cache_key[:20])):
          ptx_files = list(cache_dir.glob("*.ptx"))
          if ptx_files:
            return cache_dir, ptx_files[0], cache_dir.name
    
    if cache_root.exists():
      recent_dirs = sorted([d for d in cache_root.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
      
      kernel_name = kernel.fn.__name__
      for cache_dir in recent_dirs[:3]:
        ptx_files = list(cache_dir.glob("*.ptx"))
        if ptx_files:
          try:
            with open(ptx_files[0], 'r') as f:
              content = f.read()
              if kernel_name in content:
                return cache_dir, ptx_files[0], cache_dir.name
          except Exception as e:
            continue
    
    return None, None, None
    
  except Exception as e:
    return None, None, None

def compile_triton_variants(kernel, variants, output_dir="./ptx_output", rename_kernels=True):

  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  kernel_name = kernel.fn.__name__
  results = []

  print(f"Compiling {len(variants)} variants of {kernel_name}")

  # Detect kernel type by inspecting variant keys
  if variants:
    sample_variant = variants[0]
    has_matmul_keys = all(k in sample_variant for k in ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K'])
    has_elementwise_keys = 'BLOCK_SIZE' in sample_variant and not has_matmul_keys
    
    if has_matmul_keys:
      kernel_type = 'matmul'
      print(f"  Detected kernel type: matmul")
    elif has_elementwise_keys:
      kernel_type = 'elementwise'
      print(f"  Detected kernel type: elementwise")
    else:
      print(f"  Warning: Could not detect kernel type")
      kernel_type = 'unknown'
  else:
    kernel_type = 'unknown'

  # Prepare test data based on kernel type
  if kernel_type == 'matmul':
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    output = torch.empty((M, N), device='cuda', dtype=torch.float32)
  elif kernel_type == 'elementwise':
    size = 1024
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    output = torch.empty_like(x)
  else:
    print(f"  Skipping unknown kernel type")
    return []

  for i, constants in enumerate(variants):
    const_str = "_".join(f"{k}{v}" for k, v in constants.items())
    variant_name = f"{kernel_name}_{const_str}"

    print(f"  [{i+1}/{len(variants)}] {variant_name}")

    try:
      # Run kernel based on type
      if kernel_type == 'matmul':
        BLOCK_M = constants['BLOCK_SIZE_M']
        BLOCK_N = constants['BLOCK_SIZE_N']
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        kernel[grid](
          a, b, output, M, N, K,
          a.stride(0), a.stride(1),
          b.stride(0), b.stride(1),
          output.stride(0), output.stride(1),
          **constants
        )
      else:  # elementwise
        grid = (triton.cdiv(size, constants.get('BLOCK_SIZE', 256)),)
        kernel[grid](x, y, output, size, **constants)
      
      time.sleep(0.1)
      
      compiled = kernel[constants]
      cache_dir, ptx_file_path, cache_key = get_kernel_cache_info(kernel, constants)
      
      if not ptx_file_path or not ptx_file_path.exists():
        print(f"    Failed to find PTX file for {variant_name}")
        continue
      
      print(f"    Cache key: {cache_key}")
      print(f"    PTX file: {ptx_file_path}")

      try:
        with open(ptx_file_path, 'r') as f:
          ptx_content = f.read()
      except Exception as e:
        print(f"    Failed to read PTX file: {e}")
        continue

      original_mangled_name = "unknown"
      matches = re.findall(r'\.visible\s+\.entry\s+(\w+)', ptx_content)
      if matches:
        original_mangled_name = matches[0]

      final_mangled_name = original_mangled_name
      if rename_kernels:
        new_mangled_name = variant_name
        modified_ptx = ptx_content.replace(original_mangled_name, new_mangled_name)
        ptx_content = modified_ptx
        final_mangled_name = new_mangled_name
        print(f"    Renamed: {original_mangled_name} -> {new_mangled_name}")

      output_ptx_file = output_path / f"{variant_name}.ptx"
      with open(output_ptx_file, 'w') as f:
        f.write(ptx_content)
      
      print(f"    Created {output_ptx_file.name} ({len(ptx_content)} bytes)")

      if cache_dir and cache_dir.exists():
        try:
          shutil.rmtree(cache_dir)
          print(f"    Removed cache entry: {cache_key}")
        except Exception as e:
          print(f"    Warning: Could not remove cache entry: {e}")

      results.append({
        'variant': variant_name,
        'constants': constants,
        'original_mangled_name': original_mangled_name,
        'final_mangled_name': final_mangled_name,
        'ptx_file': str(output_ptx_file),
        'cache_key': cache_key,
        'source_cache_dir': str(cache_dir),
        'source_ptx_file': str(ptx_file_path),
        'source_file': None
      })

    except Exception as e:
      print(f"    Failed: {e}")

  print(f"Compiled {len(results)} variants")
  return results

def load_kernel_module(file_path):

  file_path = Path(file_path)
  if not file_path.exists():
    raise FileNotFoundError(f"Kernel file not found: {file_path}")

  module_name = f"kernel_module_{file_path.stem}_{hash(str(file_path))}"
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  module = importlib.util.module_from_spec(spec)

  module.torch = torch
  module.triton = triton
  module.tl = tl

  try:
    spec.loader.exec_module(module)
  except Exception as e:
    raise ImportError(f"Failed to load kernel module {file_path}: {e}")

  kernels = {}
  variants = None
  run_func = None

  for name in dir(module):
    obj = getattr(module, name)

    if hasattr(obj, '__call__') and hasattr(obj, 'fn'):
      kernels[name] = obj
    elif name == 'variants' and isinstance(obj, list):
      variants = obj
    elif name == 'run' and callable(obj):
      run_func = obj

  return kernels, variants, run_func

def compile_kernel_file(file_path, output_dir="./ptx_output"):

  print(f"\nProcessing kernel file: {file_path}")

  try:
    kernels, variants, run_func = load_kernel_module(file_path)
  except Exception as e:
    print(f"  Error loading file: {e}")
    return []

  print(f"  Found kernels: {list(kernels.keys())}")
  print(f"  Found variants: {variants is not None}")
  print(f"  Found run function: {run_func is not None}")

  if not kernels:
    print("  No Triton kernels found in file")
    return []

  if variants is None:
    print("  No 'variants' list found in file")
    return []

  if run_func:
    print("  Running kernels to populate cache...")
    try:
      run_func()
      print("  Cache populated successfully")
    except Exception as e:
      print(f"  Run function failed: {e}")
      print("  Proceeding with compilation anyway...")
  else:
    print("  No run function found - cache may not be populated")

  all_results = []

  for kernel_name, kernel_func in kernels.items():
    print(f"\n  {'='*40}")
    results = compile_triton_variants(kernel_func, variants, output_dir)
    
    for result in results:
      result['source_file'] = str(file_path)
    
    all_results.extend(results)

  return all_results

def find_triton_files(directory):
  directory = Path(directory)
  if not directory.exists():
    raise FileNotFoundError(f"Directory not found: {directory}")
  
  if not directory.is_dir():
    raise NotADirectoryError(f"Path is not a directory: {directory}")
  
  python_files = list(directory.rglob("*.py"))
  
  excluded_patterns = [
    "__pycache__",
    ".git",
    ".pytest_cache",
    "test_",
    "_test.py",
    "setup.py",
    "__init__.py"
  ]
  
  filtered_files = []
  for file_path in python_files:
    skip = False
    for pattern in excluded_patterns:
      if pattern in str(file_path):
        skip = True
        break
    
    if not skip:
      filtered_files.append(file_path)
  
  return filtered_files

def compile_directory(directory, output_dir="./ptx_output"):

  print(f"Searching for Triton kernel files in: {directory}")
  
  try:
    python_files = find_triton_files(directory)
  except Exception as e:
    print(f"Error searching directory: {e}")
    return {'files_processed': 0, 'files_with_kernels': 0, 'total_variants': 0, 'results': []}
  
  print(f"Found {len(python_files)} Python files to examine")
  
  if not python_files:
    print("No Python files found in directory")
    return {'files_processed': 0, 'files_with_kernels': 0, 'total_variants': 0, 'results': []}
  
  all_results = []
  files_with_kernels = 0
  
  for file_path in python_files:
    print(f"\n{'='*60}")
    results = compile_kernel_file(file_path, output_dir)
    
    if results:
      files_with_kernels += 1
      all_results.extend(results)
  
  summary = {
    'files_processed': len(python_files),
    'files_with_kernels': files_with_kernels,
    'total_variants': len(all_results),
    'results': all_results
  }
  
  return summary

def main():

  if len(sys.argv) != 2:
    print("Usage: python generate_kernel_binaries.py <directory>")
    print("\nExample:")
    print("  python generate_kernel_binaries.py ./kernels/")
    print("  python generate_kernel_binaries.py /path/to/triton/kernels/")
    print("\nThe script will recursively search for Python files containing:")
    print("  - @triton.jit decorated functions")
    print("  - A 'variants' list with parameter combinations")
    print("  - A 'run' function to populate the cache")
    sys.exit(1)

  directory = sys.argv[1]
  
  input_dir_name = Path(directory).resolve().name
  output_dir = f"./ptx_{input_dir_name}"

  try:
    summary = compile_directory(directory, output_dir)

    print(f"\n{'='*80}")
    print(f"COMPILATION SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {summary['files_processed']}")
    print(f"Files with kernels: {summary['files_with_kernels']}")
    print(f"Total variants compiled: {summary['total_variants']}")
    print(f"Output directory: {output_dir}")
    
    if summary['results']:
      print(f"\nCompiled variants:")
      
      by_file = {}
      for result in summary['results']:
        source_file = result['source_file']
        if source_file not in by_file:
          by_file[source_file] = []
        by_file[source_file].append(result)
      
      for source_file, results in by_file.items():
        print(f"\n  {Path(source_file).relative_to(Path(directory))}:")
        for result in results:
          print(f"    {result['variant']}")
          print(f"      Original: {result['original_mangled_name']}")
          print(f"      Renamed:  {result['final_mangled_name']}")
          print(f"      File:     {Path(result['ptx_file']).name}")
          print(f"      Cache:    {result['cache_key']}")

      print(f"\nPTX files saved to: {output_dir}/")
    else:
      print("\nNo variants were compiled")
      print("Make sure your Python files contain:")
      print("  - @triton.jit decorated functions")
      print("  - A 'variants' list")
      print("  - A 'run' function")

  except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
  main()