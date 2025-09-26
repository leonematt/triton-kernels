#!/usr/bin/env python3
"""
Driver program to run all kernel tests with detailed output
"""

import subprocess
import sys
import re
from pathlib import Path
import argparse


def find_test_files(test_dir='tests'):
    """Find all test_*.py files in the given directory"""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Error: Test directory '{test_dir}' not found")
        return []
    
    test_files = sorted(test_path.glob('test_*.py'))
    return test_files


def main():
    parser = argparse.ArgumentParser(description='Run all kernel tests')
    parser.add_argument('test_dir', nargs='?', default='tests', 
                        help='Directory containing test files (default: tests/)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show verbose test output')
    parser.add_argument('-f', '--file', type=str,
                        help='Run specific test file only (e.g., test_elementwise.py)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("KERNEL TEST SUITE")
    print("=" * 80)
    
    # Determine what to run
    if args.file:
        test_path = Path(args.test_dir) / args.file
        if not test_path.exists():
            print(f"Error: Test file '{test_path}' not found")
            return 1
        test_target = str(test_path)
        print(f"Running: {args.file}")
    else:
        test_files = find_test_files(args.test_dir)
        if not test_files:
            print(f"No test_*.py files found in '{args.test_dir}'")
            return 1
        
        print(f"Found {len(test_files)} test file(s) in '{args.test_dir}':")
        for tf in test_files:
            print(f"  - {tf.name}")
        test_target = args.test_dir
    
    print("=" * 80)
    print()
    
    # Build pytest command
    pytest_args = [
        sys.executable, "-m", "pytest",
        test_target,
        "-v",
        "--tb=short" if not args.verbose else "--tb=long",
        "--no-header"
    ]
    
    if args.verbose:
        pytest_args.append("-s")
    
    # Run pytest and capture output
    result = subprocess.run(
        pytest_args,
        capture_output=True,
        text=True
    )
    
    # Print the output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Parse results for summary
    lines = result.stdout.split('\n')
    passed = 0
    failed = 0
    skipped = 0
    
    for line in lines:
        if 'PASSED' in line:
            passed += 1
        elif 'FAILED' in line:
            failed += 1
        elif 'SKIPPED' in line:
            skipped += 1
    
    # Print summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total:   {passed + failed + skipped}")
    print("=" * 80)
    
    if result.returncode == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ SOME TESTS FAILED (exit code: {result.returncode})")
    
    print("=" * 80)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())