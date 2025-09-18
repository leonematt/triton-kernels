#!/usr/bin/env python3

import re
import os
import sys
import json
import ast
from pathlib import Path

class KernelMapper:
    def __init__(self, ptx_dir=None):
        self.mappings = {}
        if ptx_dir:
            self._scan_ptx_files(ptx_dir)
    
    def _scan_ptx_files(self, ptx_dir):
        """Scan PTX files for kernel entries"""
        for root, _, files in os.walk(ptx_dir):
            for file in files:
                if file.endswith('.ptx'):
                    self._parse_ptx_file(Path(root) / file)
    
    def _parse_ptx_file(self, file_path):
        """Extract kernel names from PTX file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find .visible .entry declarations
            matches = re.findall(r'\.visible\s+\.entry\s+(\w+)', content)
            
            for mangled_name in matches:
                base_name, constants = self._parse_name(mangled_name)
                
                if base_name not in self.mappings:
                    self.mappings[base_name] = {}
                
                # Store as tuple to preserve order
                key = tuple(constants)
                self.mappings[base_name][key] = mangled_name
                
        except Exception:
            pass  # Skip problematic files
    
    def _parse_name(self, mangled_name):
        """Parse mangled name into base + constants"""
        # Find pattern: _CONSTANT_NAMEvalue
        pattern = r'_([A-Z][A-Z_]*?)(\d+|True|False)'
        matches = re.findall(pattern, mangled_name)
        
        if not matches:
            return mangled_name, []
        
        # Base name is everything before first constant
        first_pos = mangled_name.find('_' + matches[0][0] + matches[0][1])
        base_name = mangled_name[:first_pos]
        
        # Extract constants in order they appear, handle duplicates
        constants = []  # Keep as list to preserve order
        used_keys = set()
        
        for key, value_str in matches:
            # Convert value
            if value_str == 'True':
                value = True
            elif value_str == 'False':
                value = False
            else:
                value = int(value_str)
            
            # Handle duplicate keys
            original_key = key
            counter = 2
            while key in used_keys:
                key = f"{original_key}{counter}"
                counter += 1
            
            used_keys.add(key)
            constants.append((key, value))
        
        return base_name, constants
    
    def get_kernel(self, base_name, *values):
        """Get mangled name by base name and constant values in order"""
        if base_name not in self.mappings:
            return None
        
        # Try to match values to any variant by position
        for const_key, mangled_name in self.mappings[base_name].items():
            # Extract just the values in order
            const_values = [value for name, value in const_key]
            
            # Match by exact position and count
            if len(values) == len(const_values) and list(values) == const_values:
                return mangled_name
        
        return None
    
    def list_kernels(self):
        """List all available kernels"""
        return list(self.mappings.keys())
    
    def get_all_mappings(self):
        """Return the complete mapping dictionary for external use"""
        return self.mappings
    
    def get_variants(self, kernel_name):
        """Get all variants for a specific kernel"""
        if kernel_name not in self.mappings:
            return []
        
        variants = []
        for const_tuple, mangled_name in self.mappings[kernel_name].items():
            variants.append({
                'constants': dict(const_tuple),
                'values': [value for name, value in const_tuple],
                'mangled_name': mangled_name
            })
        return variants
    
    def save_mappings(self, filepath):
        """Save mappings to a JSON file for later use"""
        serializable_mappings = {}
        
        for kernel_name, variants in self.mappings.items():
            serializable_mappings[kernel_name] = {}
            for const_tuple, mangled_name in variants.items():
                # Convert tuple of (name, value) pairs to a string key
                key = str(const_tuple)
                serializable_mappings[kernel_name][key] = mangled_name
        
        with open(filepath, 'w') as f:
            json.dump(serializable_mappings, f, indent=2)
    
    @classmethod
    def load_mappings(cls, filepath):
        """Load mappings from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create empty mapper and populate it
        mapper = cls()
        mapper.mappings = {}
        
        for kernel_name, variants in data.items():
            mapper.mappings[kernel_name] = {}
            for key_str, mangled_name in variants.items():
                # Convert string back to tuple using ast.literal_eval (safer than eval)
                try:
                    const_tuple = ast.literal_eval(key_str)
                    mapper.mappings[kernel_name][const_tuple] = mangled_name
                except (ValueError, SyntaxError):
                    # Skip malformed entries
                    continue
        
        return mapper
    
    def kernel_exists(self, kernel_name):
        """Check if a kernel exists in the mappings"""
        return kernel_name in self.mappings
    
    def get_kernel_info(self, kernel_name):
        """Get detailed information about a kernel"""
        if kernel_name not in self.mappings:
            return None
        
        variants = self.get_variants(kernel_name)
        return {
            'kernel_name': kernel_name,
            'variant_count': len(variants),
            'variants': variants
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: kernel_mapper.py <ptx_dir> [kernel_name] [values...]")
        print("       kernel_mapper.py <ptx_dir> --list")
        print("       kernel_mapper.py <ptx_dir> --save <output.json>")
        print("       kernel_mapper.py --load <mappings.json> [kernel_name] [values...]")
        sys.exit(1)
    
    # Handle loading from file
    if sys.argv[1] == '--load':
        if len(sys.argv) < 3:
            print("Error: Need JSON file path")
            sys.exit(1)
        
        mapper = KernelMapper.load_mappings(sys.argv[2])
        
        if len(sys.argv) == 3:
            print("Loaded mappings for kernels:")
            for kernel in mapper.list_kernels():
                print(f"  {kernel}")
            return
        
        if len(sys.argv) < 5:
            print("Error: Need kernel name and at least one value")
            sys.exit(1)
        
        kernel_name = sys.argv[3]
        values = []
        
        for arg in sys.argv[4:]:
            try:
                values.append(int(arg))
            except ValueError:
                if arg.lower() == 'true':
                    values.append(True)
                elif arg.lower() == 'false':
                    values.append(False)
        
        result = mapper.get_kernel(kernel_name, *values)
        
        if result:
            print(result)
        else:
            print(f"No matching kernel found for {kernel_name} with values {values}")
            sys.exit(1)
        
        return
    
    # Handle PTX directory scanning
    ptx_dir = sys.argv[1]
    mapper = KernelMapper(ptx_dir)
    
    # Handle save operation
    if len(sys.argv) == 4 and sys.argv[2] == '--save':
        output_file = sys.argv[3]
        mapper.save_mappings(output_file)
        print(f"Saved mappings to {output_file}")
        return
    
    # Handle list operation
    if len(sys.argv) == 3 and sys.argv[2] == '--list':
        print("Available kernels:")
        for kernel in mapper.list_kernels():
            print(f"  {kernel}")
        return
    
    # Handle kernel lookup
    if len(sys.argv) < 4:
        print("Error: Need kernel name and at least one value")
        sys.exit(1)
    
    kernel_name = sys.argv[2]
    values = []
    
    for arg in sys.argv[3:]:
        try:
            values.append(int(arg))
        except ValueError:
            if arg.lower() == 'true':
                values.append(True)
            elif arg.lower() == 'false':
                values.append(False)
    
    result = mapper.get_kernel(kernel_name, *values)
    
    if result:
        print(result)
    else:
        print(f"No matching kernel found for {kernel_name} with values {values}")
        sys.exit(1)

if __name__ == "__main__":
    main()