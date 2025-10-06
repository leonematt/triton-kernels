#!/usr/bin/env python3

from triton.compiler import compile as triton_compile, ASTSource
import json
import os
import re
from pathlib import Path
import sys
import importlib.util


def compile_variant(kernel, constexprs):
    """Compile a kernel variant with specific constexpr values."""
    
    # Build signature for non-constexpr args
    signature = {}
    for i, name in enumerate(kernel.arg_names):
        if i not in kernel.constexprs:
            if 'ptr' in name:
                signature[name] = '*fp32'
            else:
                signature[name] = 'i32'
    
    # Create source and compile
    src = ASTSource(kernel, signature, constexprs)
    compiled = triton_compile(src)
    
    # Get runtime args
    runtime_args = [n for i, n in enumerate(kernel.arg_names) 
                   if i not in kernel.constexprs]
    
    # Get original PTX
    ptx = compiled.asm['ptx']
    
    # Create unique kernel name based on constexpr values
    name_parts = [kernel.fn.__name__]
    for key, value in sorted(constexprs.items()):
        # Replace minus with underscore for scientific notation
        value_str = str(value).replace('-', '_')
        name_parts.append(f"{key}_{value_str}")
    unique_name = '_'.join(name_parts)
    
    # Rename kernel in PTX to avoid conflicts
    ptx = re.sub(r'\.visible \.entry \w+\(', f'.visible .entry {unique_name}(', ptx)
    ptx = re.sub(r'\.visible \.func \w+\(', f'.visible .func {unique_name}(', ptx)
    
    return {
        'ptx': ptx,
        'original_kernel_name': compiled.name,
        'renamed_kernel_name': unique_name,
        'cache_hash': compiled.hash,
        'metadata': compiled.metadata._asdict(),
        'runtime_args': runtime_args,
        'constexpr': constexprs
    }


def load_kernel_module(file_path):
    """Load a Python file and extract Triton kernels and variants."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {file_path}")

    module_name = f"kernel_module_{file_path.stem}_{hash(str(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load kernel module {file_path}: {e}")

    kernels = {}
    variants = None

    for name in dir(module):
        obj = getattr(module, name)
        if hasattr(obj, '__call__') and hasattr(obj, 'fn'):
            kernels[name] = obj
        elif name == 'VARIANTS' and isinstance(obj, list):
            variants = obj

    return kernels, variants


def compile_kernel_file(file_path, output_dir):
    """Compile all kernels in a file."""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print('='*60)
    
    try:
        kernels, variants = load_kernel_module(file_path)
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return []

    if not kernels:
        print("✗ No Triton kernels found")
        return []

    if variants is None:
        print("✗ No VARIANTS list found")
        return []

    print(f"Found {len(kernels)} kernel(s): {list(kernels.keys())}")
    print(f"Found {len(variants)} variant(s)")

    all_results = []
    
    for kernel_name, kernel_func in kernels.items():
        print(f"\nCompiling {kernel_name}:")
        
        for i, config in enumerate(variants):
            config_str = ', '.join(f"{k}={v}" for k, v in config.items())
            print(f"  [{i+1}/{len(variants)}] {config_str}...", end=" ")
            
            try:
                result = compile_variant(kernel_func, config)
                print(f"✓")
                
                # Save PTX file
                ptx_filename = f"{result['renamed_kernel_name']}.ptx"
                ptx_path = output_dir / ptx_filename
                with open(ptx_path, 'w') as f:
                    f.write(result['ptx'])
                
                result['ptx_file'] = ptx_filename
                result['source_file'] = str(file_path)
                result['kernel_name'] = kernel_name
                
                # Remove the full PTX content from result to save memory
                del result['ptx']
                
                all_results.append(result)
                
            except Exception as e:
                print(f"✗ Failed: {e}")

    return all_results


def find_kernel_files(directory):
    """Find all Python files that might contain Triton kernels."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
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
        if not any(pattern in str(file_path) for pattern in excluded_patterns):
            filtered_files.append(file_path)
    
    return filtered_files


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_kernels_binaries.py <directory>")
        print("\nExample:")
        print("  python generate_kernels_binaries.py ./kernels/")
        print("\nSearches for Python files containing:")
        print("  - @triton.jit decorated functions")
        print("  - A 'VARIANTS' list with parameter combinations")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(f"./ptx_{input_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("TRITON KERNEL COMPILER")
    print("="*60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    # Find all kernel files
    kernel_files = find_kernel_files(input_dir)
    print(f"\nFound {len(kernel_files)} Python file(s) to examine")

    if not kernel_files:
        print("No Python files found")
        sys.exit(1)

    # Compile all kernels
    all_results = []
    files_with_kernels = 0

    for file_path in kernel_files:
        results = compile_kernel_file(file_path, output_dir)
        if results:
            files_with_kernels += 1
            all_results.extend(results)

    # Save metadata
    if all_results:
        # Group by source file and kernel
        by_file = {}
        for result in all_results:
            key = (result['source_file'], result['kernel_name'])
            if key not in by_file:
                by_file[key] = {
                    'kernel_name': result['kernel_name'],
                    'source_file': result['source_file'],
                    'runtime_args': result['runtime_args'],
                    'variants': []
                }
            by_file[key]['variants'].append({
                'constexpr': result['constexpr'],
                'renamed_kernel_name': result['renamed_kernel_name'],
                'cache_hash': result['cache_hash'],
                'ptx_file': result['ptx_file'],
            })

        metadata = {
            'kernels': list(by_file.values())
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("COMPILATION SUMMARY")
    print("="*60)
    print(f"Files processed:      {len(kernel_files)}")
    print(f"Files with kernels:   {files_with_kernels}")
    print(f"Total variants:       {len(all_results)}")
    print(f"Output directory:     {output_dir}")

    if all_results:
        print(f"\n✓ Saved {len(all_results)} PTX files")
        print(f"✓ Saved metadata.json")
        
        print("\nCompiled variants:")
        for (source_file, kernel_name), info in by_file.items():
            rel_path = Path(source_file).relative_to(input_dir)
            print(f"\n  {rel_path} :: {kernel_name}")
            print(f"    Runtime args: {', '.join(info['runtime_args'])}")
            print(f"    Variants: {len(info['variants'])}")
            for variant in info['variants'][:3]:
                print(f"      - {variant['renamed_kernel_name']}")
            if len(info['variants']) > 3:
                print(f"      ... and {len(info['variants']) - 3} more")
    else:
        print("\n✗ No variants were compiled")
        print("Make sure your Python files contain:")
        print("  - @triton.jit decorated functions")
        print("  - A 'VARIANTS' list")


if __name__ == "__main__":
    main()