#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

# Test files to run (in order)
TEST_FILES = [
    "test_elementwise.py",
    "test_matmul.py",
    "test_layernorm.py",
    "test_softmax.py",
    # "test_rotary_embedding.py",  # Uncomment when ready
]

def run_test_file(test_file):
    """Run a single test file and return success status"""
    print(f"\n{'='*80}")
    print(f"Running: {test_file}")
    print('='*80)
    
    test_path = Path(__file__).parent / test_file
    
    result = subprocess.run(
        ["pytest", str(test_path), "-v", "-s"]
        # No cwd parameter - runs from wherever script is invoked
    )
    
    return result.returncode == 0

def main():
    """Run all tests and report results"""
    print("="*80)
    print("TRITON KERNELS TEST SUITE")
    print("="*80)
    
    results = {}
    
    for test_file in TEST_FILES:
        test_path = Path(__file__).parent / test_file
        if not test_path.exists():
            print(f"\n⚠ Skipping {test_file} (not found)")
            results[test_file] = "SKIP"
            continue
        
        success = run_test_file(test_file)
        results[test_file] = "PASS" if success else "FAIL"
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_file, status in results.items():
        emoji = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "⊘")
        print(f"{emoji} {test_file}: {status}")
    
    # Overall result
    total = len(results)
    passed = sum(1 for s in results.values() if s == "PASS")
    failed = sum(1 for s in results.values() if s == "FAIL")
    skipped = sum(1 for s in results.values() if s == "SKIP")
    
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    
    # Exit with error code if any tests failed
    if failed > 0:
        print("\n❌ Some tests failed")
        sys.exit(1)
    else:
        print("\n✅ All tests passed")
        sys.exit(0)

if __name__ == "__main__":
    main()