#!/usr/bin/env python3

import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

class KernelNameMapper:
    """
    Creates and manages mappings between kernel names + constants and mangled variant names.
    This is a supporting script that just handles the name mapping logic.
    """
    
    def __init__(self, kernels_directory: Path = None):
        self.kernels_directory = Path(kernels_directory) if kernels_directory else None
        self.static_mappings = {}  # For manually added mappings
        self.loaded_mappings = {}  # For mappings loaded from kernel files
        
        if self.kernels_directory:
            self._load_kernel_mappings()
    
    def _load_kernel_mappings(self):
        """Load kernel mappings from Python files in the directory."""
        if not self.kernels_directory or not self.kernels_directory.exists():
            return
        
        python_files = list(self.kernels_directory.rglob("*.py"))
        excluded_patterns = ["__pycache__", ".git", "test_", "_test.py", "setup.py", "__init__.py"]
        
        for file_path in python_files:
            if any(pattern in str(file_path) for pattern in excluded_patterns):
                continue
            
            try:
                self._load_kernel_file(file_path)
            except Exception:
                continue  # Skip files that can't be loaded
    
    def _load_kernel_file(self, file_path: Path):
        """Load a single kernel file and extract its variants."""
        module_name = f"kernel_module_{file_path.stem}_{hash(str(file_path))}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add required imports (minimal to avoid errors)
        try:
            import torch
            import triton
            import triton.language as tl
            module.torch = torch
            module.triton = triton
            module.tl = tl
        except ImportError:
            pass  # Skip if dependencies not available
        
        try:
            spec.loader.exec_module(module)
        except Exception:
            return  # Skip files that fail to load
        
        # Extract kernels and variants
        kernels = {}
        variants = []
        
        for name in dir(module):
            obj = getattr(module, name)
            
            if hasattr(obj, '__call__') and hasattr(obj, 'fn'):
                kernels[name] = obj
            elif name == 'variants' and isinstance(obj, list):
                variants = obj
        
        # Store mappings for each kernel
        for kernel_name in kernels.keys():
            if kernel_name not in self.loaded_mappings:
                self.loaded_mappings[kernel_name] = {}
            
            for variant_constants in variants:
                # Generate variant name using same logic as compiler
                const_str = "_".join(f"{k}{v}" for k, v in sorted(variant_constants.items()))
                variant_name = f"{kernel_name}_{const_str}"
                
                # Create lookup key from constants
                constants_key = tuple(sorted(variant_constants.items()))
                self.loaded_mappings[kernel_name][constants_key] = variant_name
    
    def add_mapping(self, kernel_name: str, constants: Dict[str, Any], variant_name: str = None):
        """
        Manually add a kernel mapping.
        
        Args:
            kernel_name: Base kernel name (e.g., 'add_vectors')
            constants: Dictionary of constant values
            variant_name: Optional custom variant name (auto-generated if None)
        """
        if variant_name is None:
            const_str = "_".join(f"{k}{v}" for k, v in sorted(constants.items()))
            variant_name = f"{kernel_name}_{const_str}"
        
        if kernel_name not in self.static_mappings:
            self.static_mappings[kernel_name] = {}
        
        constants_key = tuple(sorted(constants.items()))
        self.static_mappings[kernel_name][constants_key] = variant_name
    
    def get_variant_name(self, kernel_name: str = None, constants: Dict[str, Any] = None, const_args: List[str] = None, all_args: List[str] = None) -> Optional[str]:
        """
        Get the mangled variant name for given kernel and constants.
        
        Args:
            kernel_name: Base kernel name (e.g., 'add_vectors', 'add_vectors2') - optional if all_args provided
            constants: Dictionary of constant values (optional)
            const_args: List of constant arguments (optional)
            all_args: Complete list of all arguments including kernel name (e.g., ["add_vectors", "x_ptr", "y_ptr", "out_ptr", "n", "BLOCK_SIZE=256", "BLOCK_SIZE2=128"])
            
        Returns:
            Mangled variant name (e.g., 'add_vectors_BLOCK_SIZE256_BLOCK_SIZE2128')
        """
        # Handle all_args input format - extract kernel name and constants automatically
        if all_args is not None:
            if len(all_args) < 2:
                raise ValueError("all_args must contain at least kernel name and one argument")
            
            # First argument is always the kernel name
            kernel_name = all_args[0]
            remaining_args = all_args[1:]
            
            # Filter out non-constant arguments and extract constants
            const_args = []
            for arg in remaining_args:
                # Skip pointer arguments (common patterns)
                if any(suffix in arg.lower() for suffix in ['_ptr', '_pointer']):
                    continue
                
                # Skip common non-constant parameter names
                if arg.lower() in ['n', 'size', 'length', 'count', 'offset', 'stride', 'x', 'y', 'z', 'input', 'output', 'data']:
                    continue
                
                # Look for arguments that contain = or : (explicit constants)
                if '=' in arg or ':' in arg:
                    const_args.append(arg)
                    continue
                
                # Look for arguments that are ALL_CAPS (likely constants)
                if arg.replace('_', '').isupper():
                    # This is a constant name without value - skip it since we need the value
                    continue
                
                # Look for KEY VALUE pairs where next arg could be the value
                if (remaining_args.index(arg) < len(remaining_args) - 1 and
                    arg.replace('_', '').isupper()):
                    next_arg = remaining_args[remaining_args.index(arg) + 1]
                    # Check if next arg looks like a value (number or simple string)
                    if (next_arg.isdigit() or 
                        (next_arg.replace('.', '').replace('-', '').isdigit()) or
                        next_arg.lower() in ['true', 'false']):
                        const_args.append(f"{arg}={next_arg}")
            
            # Parse the extracted constant arguments
            if const_args:
                constants = parse_constants(const_args)
            else:
                # If no constants found, assume we need to extract from known patterns
                constants = self._extract_constants_from_args(remaining_args)
        
        # Handle const_args input format
        elif const_args is not None:
            constants = parse_constants(const_args)
        
        if constants is None or len(constants) == 0:
            raise ValueError("No constant expressions found in arguments")
        
        if kernel_name is None:
            raise ValueError("Kernel name is required")
        
        constants_key = tuple(sorted(constants.items()))
        
        # Check static mappings first (manually added)
        if kernel_name in self.static_mappings:
            if constants_key in self.static_mappings[kernel_name]:
                return self.static_mappings[kernel_name][constants_key]
        
        # Check loaded mappings (from kernel files)
        if kernel_name in self.loaded_mappings:
            if constants_key in self.loaded_mappings[kernel_name]:
                return self.loaded_mappings[kernel_name][constants_key]
        
        # If not found, generate the expected name (best guess)
        const_str = "_".join(f"{k}{v}" for k, v in sorted(constants.items()))
        return f"{kernel_name}_{const_str}"
    
    def get_variant_name_from_values(self, kernel_name: str, all_values: List[Any]) -> Optional[str]:
        """
        Get variant name by matching actual runtime values to known constant patterns.
        Automatically selects variants based on the number of arguments passed.
        
        Args:
            kernel_name: Base kernel name (e.g., 'add_vectors', 'add_vectors2') 
            all_values: List of actual runtime values passed to kernel
            
        Returns:
            Mangled variant name that matches the constant pattern
            
        Example:
            # For add_vectors with 1 constexpr (5 total args: x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE)
            values = [x_ptr, y_ptr, out_ptr, 4096, 256]
            name = mapper.get_variant_name_from_values("add_vectors", values)
            # Returns: "add_vectors_BLOCK_SIZE256"
            
            # For add_vectors with 2 constexpr (6 total args: x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE, BLOCK_SIZE2)
            values = [x_ptr, y_ptr, out_ptr, 4096, 256, 128]
            name = mapper.get_variant_name_from_values("add_vectors", values)
            # Returns: "add_vectors_BLOCK_SIZE256_BLOCK_SIZE2128"
        """
        if kernel_name not in self.loaded_mappings and kernel_name not in self.static_mappings:
            raise ValueError(f"No mappings found for kernel '{kernel_name}'. Load kernel file first.")
        
        # Get all known variants for this kernel
        all_variants = []
        
        if kernel_name in self.loaded_mappings:
            for constants_key, variant_name in self.loaded_mappings[kernel_name].items():
                constants = dict(constants_key)
                all_variants.append((constants, variant_name))
        
        if kernel_name in self.static_mappings:
            for constants_key, variant_name in self.static_mappings[kernel_name].items():
                constants = dict(constants_key)
                all_variants.append((constants, variant_name))
        
        if not all_variants:
            raise ValueError(f"No variants found for kernel '{kernel_name}'")
        
        # Filter variants by number of constant expressions based on argument count
        num_args = len(all_values)
        filtered_variants = []
        
        for constants, variant_name in all_variants:
            # Calculate expected argument count for this variant
            # Base args (pointers + regular params) + number of constants
            num_constants = len(constants)
            
            # For add_vectors: x_ptr, y_ptr, out_ptr, n + constants
            # So: 4 base args + number of constants = total expected args
            expected_args = 4 + num_constants  # Assuming 4 base args for add_vectors
            
            # Match variants that expect the same number of arguments
            if expected_args == num_args:
                filtered_variants.append((constants, variant_name))
        
        if not filtered_variants:
            # Show available options
            available_counts = []
            for constants, _ in all_variants:
                expected = 4 + len(constants)
                available_counts.append(f"{len(constants)} constants ({expected} total args)")
            
            raise ValueError(
                f"No variants found for kernel '{kernel_name}' with {num_args} arguments. "
                f"Available: {', '.join(set(available_counts))}"
            )
        
        # Extract numeric values from the input (skip pointers and non-numeric values)
        numeric_values = []
        for value in all_values:
            if isinstance(value, (int, float)):
                numeric_values.append(value)
            elif isinstance(value, str) and value.isdigit():
                numeric_values.append(int(value))
            elif hasattr(value, '__int__'):  # Handle tensor sizes, etc.
                try:
                    numeric_values.append(int(value))
                except:
                    continue
        
        # Try to match numeric values to constant patterns from filtered variants
        best_match = None
        best_score = -1
        
        for constants, variant_name in filtered_variants:
            # Get the constant values from this variant
            const_values = list(constants.values())
            const_values = [v for v in const_values if isinstance(v, (int, float))]
            
            # Try to find these constant values in our numeric values
            match_score = 0
            if len(const_values) <= len(numeric_values):
                # Check if all constant values appear in our numeric values
                # Look at the last N values where N = number of constants
                tail_values = numeric_values[-len(const_values):] if const_values else []
                
                # Try exact match on the tail values (constants are usually at the end)
                if tail_values == const_values:
                    return variant_name
                
                # Fallback: check if all constant values appear anywhere
                for const_val in const_values:
                    if const_val in numeric_values:
                        match_score += 1
                
                # Prefer exact matches (all constants found)
                if match_score == len(const_values) and match_score > best_score:
                    best_match = variant_name
                    best_score = match_score
        
        return best_match
    
    def find_kernel_by_signature(self, kernel_name: str, all_values: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Find the best matching kernel variant by analyzing the value signature.
        Returns detailed information about the match.
        """
        variant_name = self.get_variant_name_from_values(kernel_name, all_values)
        if not variant_name:
            return None
        
        # Get the constants for this variant
        constants = None
        if kernel_name in self.loaded_mappings:
            for constants_key, v_name in self.loaded_mappings[kernel_name].items():
                if v_name == variant_name:
                    constants = dict(constants_key)
                    break
        
        if not constants and kernel_name in self.static_mappings:
            for constants_key, v_name in self.static_mappings[kernel_name].items():
                if v_name == variant_name:
                    constants = dict(constants_key)
                    break
        
        return {
            'variant_name': variant_name,
            'constants': constants,
            'kernel_name': kernel_name
        }
    
    def list_kernels(self) -> List[str]:
        """Get list of all known kernel names."""
        all_kernels = set()
        all_kernels.update(self.static_mappings.keys())
        all_kernels.update(self.loaded_mappings.keys())
        return sorted(all_kernels)
    
    def list_variants(self, kernel_name: str) -> List[Dict[str, Any]]:
        """Get all known variants for a kernel."""
        variants = []
        
        # From static mappings
        if kernel_name in self.static_mappings:
            for constants_key, variant_name in self.static_mappings[kernel_name].items():
                constants = dict(constants_key)
                variants.append({
                    'constants': constants,
                    'variant_name': variant_name,
                    'source': 'static'
                })
        
        # From loaded mappings
        if kernel_name in self.loaded_mappings:
            for constants_key, variant_name in self.loaded_mappings[kernel_name].items():
                constants = dict(constants_key)
                variants.append({
                    'constants': constants,
                    'variant_name': variant_name,
                    'source': 'loaded'
                })
        
        return variants
    
    def save_mappings(self, filepath: Path):
        """Save all mappings to a JSON file."""
        data = {
            'static_mappings': self._serialize_mappings(self.static_mappings),
            'loaded_mappings': self._serialize_mappings(self.loaded_mappings)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_mappings(self, filepath: Path):
        """Load mappings from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'static_mappings' in data:
            self.static_mappings = self._deserialize_mappings(data['static_mappings'])
        
        if 'loaded_mappings' in data:
            self.loaded_mappings = self._deserialize_mappings(data['loaded_mappings'])
    
    def _serialize_mappings(self, mappings):
        """Convert mappings to JSON-serializable format."""
        serialized = {}
        for kernel_name, kernel_mappings in mappings.items():
            serialized[kernel_name] = {}
            for constants_key, variant_name in kernel_mappings.items():
                # Convert tuple key to string
                key_str = json.dumps(list(constants_key), sort_keys=True)
                serialized[kernel_name][key_str] = variant_name
        return serialized
    
    def _deserialize_mappings(self, serialized):
        """Convert from JSON format back to internal format."""
        mappings = {}
        for kernel_name, kernel_mappings in serialized.items():
            mappings[kernel_name] = {}
            for key_str, variant_name in kernel_mappings.items():
                # Convert string key back to tuple
                constants_list = json.loads(key_str)
                constants_key = tuple(constants_list)
                mappings[kernel_name][constants_key] = variant_name
        return mappings

def get_kernel_from_values(kernel_name: str, runtime_values: List[Any], kernels_dir: str = None) -> str:
    """
    Get kernel variant name by analyzing actual runtime values.
    
    Args:
        kernel_name: Base kernel name (e.g., 'add_vectors', 'add_vectors2')
        runtime_values: List of actual values passed to kernel at runtime
        kernels_dir: Directory containing kernel files to load mappings from
        
    Returns:
        Mangled kernel variant name
        
    Examples:
        # For add_vectors(x_ptr, y_ptr, out_ptr, n=4096, BLOCK_SIZE=256, BLOCK_SIZE2=128)
        values = [x_ptr, y_ptr, out_ptr, 4096, 256, 128]
        name = get_kernel_from_values("add_vectors", values, "./kernels")
        # Returns: "add_vectors_BLOCK_SIZE256_BLOCK_SIZE2128"
        
        # For add_vectors2 with different values
        values = [x_ptr, y_ptr, out_ptr, 8192, 512, 256]  
        name = get_kernel_from_values("add_vectors2", values, "./kernels")
        # Returns: "add_vectors2_BLOCK_SIZE512_BLOCK_SIZE2256"
    """
    if not kernels_dir:
        raise ValueError("kernels_dir is required to load kernel mappings for value-based lookup")
    
    mapper = KernelNameMapper(kernels_dir)
    variant_name = mapper.get_variant_name_from_values(kernel_name, runtime_values)
    
    if not variant_name:
        raise ValueError(f"Could not match runtime values to any known variant of '{kernel_name}'")
    
    return variant_name

def analyze_kernel_call(kernel_name: str, runtime_values: List[Any], kernels_dir: str = None) -> Dict[str, Any]:
    """
    Analyze a kernel call and return detailed information about the matching variant.
    
    Args:
        kernel_name: Base kernel name
        runtime_values: Actual runtime values
        kernels_dir: Directory containing kernel files
        
    Returns:
        Dictionary with variant_name, constants, and analysis info
    """
    if not kernels_dir:
        raise ValueError("kernels_dir is required to load kernel mappings")
    
    mapper = KernelNameMapper(kernels_dir)
    return mapper.find_kernel_by_signature(kernel_name, runtime_values)
    """
    Smart function that figures out the kernel name and constants from all arguments.
    
    Args:
        all_args: Complete list of kernel call arguments, including:
                 - kernel name (first argument)
                 - regular parameters (x_ptr, y_ptr, n, etc.)
                 - constant expressions (BLOCK_SIZE=256, BLOCK_SIZE2=128, etc.)
        kernels_dir: Optional directory to load kernel mappings from
        
    Returns:
        Mangled variant name
        
    Examples:
        # Format 1: Mixed arguments with explicit constants
        args = ["add_vectors", "x_ptr", "y_ptr", "out_ptr", "n", "BLOCK_SIZE=256", "BLOCK_SIZE2=128"]
        name = get_variant_name_smart(args)
        # Returns: "add_vectors_BLOCK_SIZE256_BLOCK_SIZE2128"
        
        # Format 2: Arguments with KEY VALUE pairs
        args = ["add_vectors", "x_ptr", "y_ptr", "out_ptr", "n", "BLOCK_SIZE", "256", "BLOCK_SIZE2", "128"]
        name = get_variant_name_smart(args)
        # Returns: "add_vectors_BLOCK_SIZE256_BLOCK_SIZE2128"
        
        # Format 3: Only kernel name and constants
        args = ["matmul", "M=256", "N=512", "K=128"]
        name = get_variant_name_smart(args)
        # Returns: "matmul_K128_M256_N512"
    """
    mapper = KernelNameMapper(kernels_dir)
    return mapper.get_variant_name(all_args=all_args)
    """Parse constant arguments from command line."""
    constants = {}
    
    i = 0
    while i < len(const_args):
        arg = const_args[i]
        
        # Handle KEY=VALUE or KEY:VALUE format
        if '=' in arg:
            key, value = arg.split('=', 1)
        elif ':' in arg:
            key, value = arg.split(':', 1)
        # Handle KEY VALUE format (two separate args)
        elif i + 1 < len(const_args) and not ('=' in const_args[i + 1] or ':' in const_args[i + 1]):
            key = arg
            value = const_args[i + 1]
            i += 1  # Skip next arg since we consumed it
        else:
            raise ValueError(f"Invalid constant format: {arg}. Use KEY=VALUE, KEY:VALUE, or KEY VALUE")
        
        # Try to convert value to appropriate type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        
        constants[key.strip()] = value
        i += 1
    
    return constants

def main():
    parser = argparse.ArgumentParser(
        description="Map kernel names and constants to mangled variant names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get variant name for kernel with constants
  %(prog)s add_vectors BLOCK_SIZE=256 BLOCK_SIZE2=128
  %(prog)s add_vectors2 BLOCK_SIZE:512 NUM_WARPS:4
  %(prog)s matmul M 256 N 512 K 128
  
  # Load mappings from kernel directory
  %(prog)s --load-dir /path/to/kernels add_vectors BLOCK_SIZE=256
  
  # Add custom mapping and save
  %(prog)s --add add_vectors BLOCK_SIZE=256 BLOCK_SIZE2=128 --save mappings.json
  
  # Load saved mappings
  %(prog)s --load mappings.json add_vectors BLOCK_SIZE=256 BLOCK_SIZE2=128
  
  # Just generate name without loading anything
  %(prog)s my_kernel PARAM1=100 PARAM2=200
        """
    )
    
    parser.add_argument("kernel_name", nargs='?',
                       help="Name of the kernel (e.g., 'add_vectors', 'add_vectors2')")
    
    parser.add_argument("constants", nargs='*',
                       help="Constant expressions (e.g., BLOCK_SIZE=256 BLOCK_SIZE2=128)")
    
    parser.add_argument("--load-dir", metavar="DIR",
                       help="Load kernel mappings from directory")
    
    parser.add_argument("--load", metavar="FILE",
                       help="Load mappings from JSON file")
    
    parser.add_argument("--save", metavar="FILE",
                       help="Save mappings to JSON file")
    
    parser.add_argument("--add", action="store_true",
                       help="Add this mapping to the mapper (use with --save)")
    
    parser.add_argument("--list", action="store_true",
                       help="List all known kernels")
    
    parser.add_argument("--list-variants", metavar="KERNEL",
                       help="List all variants for a specific kernel")
    
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    try:
        # Create mapper
        mapper = KernelNameMapper(args.load_dir)
        
        # Load existing mappings if specified
        if args.load:
            mapper.load_mappings(Path(args.load))
        
        # Handle list operations
        if args.list:
            kernels = mapper.list_kernels()
            if args.json:
                print(json.dumps({"kernels": kernels}))
            else:
                print("Known kernels:")
                for kernel in kernels:
                    variants = mapper.list_variants(kernel)
                    print(f"  {kernel} ({len(variants)} variants)")
            return
        
        if args.list_variants:
            variants = mapper.list_variants(args.list_variants)
            if args.json:
                print(json.dumps({"kernel": args.list_variants, "variants": variants}))
            else:
                print(f"Variants for {args.list_variants}:")
                for variant in variants:
                    print(f"  {variant['variant_name']}")
                    print(f"    Constants: {variant['constants']}")
                    print(f"    Source: {variant['source']}")
                    print()
            return
        
        # Main mapping operation
        if not args.kernel_name:
            parser.error("kernel_name is required")
        
        if not args.constants:
            parser.error("At least one constant expression is required")
        
        # Parse constants
        constants = parse_constants(args.constants)
        
        # Add mapping if requested
        if args.add:
            mapper.add_mapping(args.kernel_name, constants)
        
        # Get variant name
        variant_name = mapper.get_variant_name(args.kernel_name, constants)
        
        # Save mappings if requested
        if args.save:
            mapper.save_mappings(Path(args.save))
        
        # Output result
        if args.json:
            output = {
                "kernel_name": args.kernel_name,
                "constants": constants,
                "variant_name": variant_name
            }
            print(json.dumps(output))
        else:
            print(variant_name)
    
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()