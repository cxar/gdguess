#!/usr/bin/env python3
"""
Script to run all Transformer encoder tests.
"""

import os
import sys
import traceback

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_transformer import run_transformer_sanity_test
from tests.test_transformer_detailed import run_detailed_transformer_test

def main():
    """Run all tests."""
    print("=" * 80)
    print("TRANSFORMER ENCODER TEST SUITE")
    print("=" * 80)
    
    # Run basic sanity test
    try:
        print("\n\nRunning basic sanity test...")
        basic_success = run_transformer_sanity_test()
    except Exception as e:
        print(f"Error in basic test: {e}")
        traceback.print_exc()
        basic_success = False
    
    # Run detailed test
    try:
        print("\n\nRunning detailed test...")
        detailed_success = run_detailed_transformer_test()
    except Exception as e:
        print(f"Error in detailed test: {e}")
        traceback.print_exc()
        detailed_success = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Basic sanity test: {'PASSED' if basic_success else 'FAILED'}")
    print(f"Detailed test: {'PASSED' if detailed_success else 'FAILED'}")
    print("=" * 80)
    
    # Return exit code
    return 0 if (basic_success and detailed_success) else 1

if __name__ == "__main__":
    sys.exit(main()) 