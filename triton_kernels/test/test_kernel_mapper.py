#!/usr/bin/env python3

"""
Main test program for kernel_name_mapper.py
Tests both single and multi-variable kernel mapping based on argument count.
"""

import sys
import traceback
from pathlib import Path

# Import the kernel mapper functions
try:
    from utils.kernel_name_mapper import get_kernel_from_values, KernelNameMapper, analyze_kernel_call
except ImportError as e:
    print(f"Error importing kernel_name_mapper: {e}")
    print("Make sure kernel_name_mapper.py is in the same directory")
    sys.exit(1)

class KernelMapperTester:
    """Test harness for kernel mapping functionality."""
    
    def __init__(self, kernels_dir="./kernels"):
        self.kernels_dir = kernels_dir
        self.test_results = []
        self.passed = 0
        self.failed = 0
        
        # Test data - simulated pointers and common values
        self.x_ptr = 0x1000
        self.y_ptr = 0x2000
        self.out_ptr = 0x3000
        self.n = 4096
        
        print(f"🧪 Kernel Mapper Test Suite")
        print(f"📁 Kernels directory: {kernels_dir}")
        print("=" * 80)
    
    def run_test(self, test_name, test_func):
        """Run a single test and record results."""
        print(f"\n🔍 {test_name}")
        print("-" * 60)
        
        try:
            test_func()
            self.passed += 1
            print(f"✅ {test_name} - PASSED")
        except Exception as e:
            self.failed += 1
            print(f"❌ {test_name} - FAILED")
            print(f"   Error: {str(e)}")
            if "--verbose" in sys.argv:
                print(f"   Traceback: {traceback.format_exc()}")
    
    def test_single_variable_kernels(self):
        """Test single constant expression kernels (5 arguments)."""
        print("Testing single variable kernels (BLOCK_SIZE only)")
        print("Expected format: [x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE]")
        
        test_cases = [
            (256, "add_vectors_BLOCK_SIZE256"),
            (512, "add_vectors_BLOCK_SIZE512"), 
            (1024, "add_vectors_BLOCK_SIZE1024"),
        ]
        
        for block_size, expected_name in test_cases:
            args = [self.x_ptr, self.y_ptr, self.out_ptr, self.n, block_size]
            
            try:
                result = get_kernel_from_values("add_vectors", args, self.kernels_dir)
                
                if result == expected_name:
                    print(f"  ✅ BLOCK_SIZE={block_size:4d}: {result}")
                else:
                    print(f"  ❌ BLOCK_SIZE={block_size:4d}: got {result}, expected {expected_name}")
                    raise AssertionError(f"Expected {expected_name}, got {result}")
                    
            except Exception as e:
                print(f"  ❌ BLOCK_SIZE={block_size:4d}: Error - {str(e)}")
                raise
    
    def test_double_variable_kernels(self):
        """Test double constant expression kernels (6 arguments)."""
        print("Testing double variable kernels (BLOCK_SIZE + BLOCK_SIZE2)")
        print("Expected format: [x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE, BLOCK_SIZE2]")
        
        test_cases = [
            (128, 64, "add_vectors_BLOCK_SIZE128_BLOCK_SIZE264"),
            (128, 128, "add_vectors_BLOCK_SIZE128_BLOCK_SIZE2128"),
            (256, 128, "add_vectors_BLOCK_SIZE256_BLOCK_SIZE2128"),
            (256, 256, "add_vectors_BLOCK_SIZE256_BLOCK_SIZE2256"),
            (512, 256, "add_vectors_BLOCK_SIZE512_BLOCK_SIZE2256"),
            (512, 512, "add_vectors_BLOCK_SIZE512_BLOCK_SIZE2512"),
        ]
        
        for block_size, block_size2, expected_name in test_cases:
            args = [self.x_ptr, self.y_ptr, self.out_ptr, self.n, block_size, block_size2]
            
            try:
                result = get_kernel_from_values("add_vectors", args, self.kernels_dir)
                
                if result == expected_name:
                    print(f"  ✅ BLOCK_SIZE={block_size:3d}, BLOCK_SIZE2={block_size2:3d}: {result}")
                else:
                    print(f"  ❌ BLOCK_SIZE={block_size:3d}, BLOCK_SIZE2={block_size2:3d}: got {result}, expected {expected_name}")
                    raise AssertionError(f"Expected {expected_name}, got {result}")
                    
            except Exception as e:
                print(f"  ❌ BLOCK_SIZE={block_size:3d}, BLOCK_SIZE2={block_size2:3d}: Error - {str(e)}")
                raise
    
    def test_argument_count_validation(self):
        """Test that argument count determines which variants are considered."""
        print("Testing argument count-based variant selection")
        
        # Test with wrong number of arguments
        test_cases = [
            # Too few arguments
            ([self.x_ptr, self.y_ptr, self.out_ptr], "Too few arguments (3)"),
            # 4 arguments - no variants should match
            ([self.x_ptr, self.y_ptr, self.out_ptr, self.n], "4 arguments - no constants"),
            # 7 arguments - too many for available variants
            ([self.x_ptr, self.y_ptr, self.out_ptr, self.n, 256, 128, 64], "7 arguments - too many constants"),
        ]
        
        for args, description in test_cases:
            print(f"  Testing {description}: {len(args)} args")
            try:
                result = get_kernel_from_values("add_vectors", args, self.kernels_dir)
                print(f"    Unexpected success: {result}")
                # This should not happen for these test cases
                if len(args) not in [5, 6]:  # Only 5 and 6 args should work
                    raise AssertionError(f"Expected failure for {len(args)} args, but got: {result}")
            except ValueError as e:
                print(f"    ✅ Expected error: {str(e)}")
            except Exception as e:
                print(f"    ❌ Unexpected error type: {str(e)}")
                raise
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        print("Testing edge cases")
        
        # Test with non-existent kernel
        print("  Testing non-existent kernel")
        try:
            args = [self.x_ptr, self.y_ptr, self.out_ptr, self.n, 256]
            result = get_kernel_from_values("non_existent_kernel", args, self.kernels_dir)
            raise AssertionError(f"Expected error for non-existent kernel, got: {result}")
        except ValueError as e:
            print(f"    ✅ Expected error: {str(e)}")
        
        # Test with non-matching constant values
        print("  Testing non-matching constant values")
        try:
            args = [self.x_ptr, self.y_ptr, self.out_ptr, self.n, 999]  # No variant with BLOCK_SIZE=999
            result = get_kernel_from_values("add_vectors", args, self.kernels_dir)
            # This might succeed if the mapper finds a partial match
            print(f"    Result (may be partial match): {result}")
        except Exception as e:
            print(f"    ✅ Error (expected for non-matching values): {str(e)}")
    
    def test_detailed_analysis(self):
        """Test the detailed analysis function."""
        print("Testing detailed kernel analysis")
        
        test_cases = [
            ([self.x_ptr, self.y_ptr, self.out_ptr, self.n, 256], "Single variable"),
            ([self.x_ptr, self.y_ptr, self.out_ptr, self.n, 256, 128], "Double variable"),
        ]
        
        for args, description in test_cases:
            try:
                result = analyze_kernel_call("add_vectors", args, self.kernels_dir)
                print(f"  ✅ {description}:")
                print(f"    Variant: {result['variant_name']}")
                print(f"    Constants: {result['constants']}")
                print(f"    Kernel: {result['kernel_name']}")
            except Exception as e:
                print(f"  ❌ {description}: Error - {str(e)}")
                raise
    
    def test_performance_simulation(self):
        """Simulate realistic performance testing scenario."""
        print("Testing realistic runtime scenario")
        
        # Simulate different kernel launches
        kernel_launches = [
            # Small data - use small block sizes
            (1024, 128, "Small data set"),
            (1024, 128, 64, "Small data set (dual constants)"),
            
            # Medium data - use medium block sizes  
            (8192, 256, "Medium data set"),
            (8192, 256, 128, "Medium data set (dual constants)"),
            
            # Large data - use large block sizes
            (32768, 512, "Large data set"),  
            (32768, 512, 256, "Large data set (dual constants)"),
        ]
        
        for launch_config in kernel_launches:
            if len(launch_config) == 3:  # Single constant
                data_size, block_size, description = launch_config
                args = [self.x_ptr, self.y_ptr, self.out_ptr, data_size, block_size]
            else:  # Double constant
                data_size, block_size, block_size2, description = launch_config
                args = [self.x_ptr, self.y_ptr, self.out_ptr, data_size, block_size, block_size2]
            
            try:
                result = get_kernel_from_values("add_vectors", args, self.kernels_dir)
                constants_part = result.split('_', 2)[2] if '_' in result else "unknown"
                print(f"  ✅ {description}: data_size={data_size}, {constants_part}")
            except Exception as e:
                print(f"  ❌ {description}: Error - {str(e)}")
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🚀 Starting comprehensive kernel mapper tests...")
        
        # Check if kernels directory exists
        if not Path(self.kernels_dir).exists():
            print(f"❌ Kernels directory not found: {self.kernels_dir}")
            print("Please create the directory and add your kernel files, or specify correct path.")
            return False
        
        # Run all test suites
        test_suites = [
            ("Single Variable Kernels", self.test_single_variable_kernels),
            ("Double Variable Kernels", self.test_double_variable_kernels),
            ("Argument Count Validation", self.test_argument_count_validation),
            ("Edge Cases", self.test_edge_cases),
            ("Detailed Analysis", self.test_detailed_analysis),
            ("Performance Simulation", self.test_performance_simulation),
        ]
        
        for test_name, test_func in test_suites:
            self.run_test(test_name, test_func)
        
        # Print final summary
        self.print_summary()
        return self.failed == 0
    
    def print_summary(self):
        """Print test execution summary."""
        total_tests = self.passed + self.failed
        pass_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 80)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 All tests passed! Kernel mapper is working correctly.")
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Check the errors above.")
        
        print("=" * 80)

def main():
    """Main entry point for the test program."""
    
    # Parse command line arguments
    kernels_dir = "./kernels"  # Default
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("Usage: python test_kernel_mapper.py [kernels_directory] [--verbose]")
            print("\nOptions:")
            print("  kernels_directory    Path to directory containing kernel files (default: ./kernels)")
            print("  --verbose           Show detailed error tracebacks")
            print("\nExample:")
            print("  python test_kernel_mapper.py ./my_kernels --verbose")
            return
        elif not sys.argv[1].startswith("--"):
            kernels_dir = sys.argv[1]
    
    # Create and run test suite
    tester = KernelMapperTester(kernels_dir)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()