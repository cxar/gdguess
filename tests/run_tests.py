#!/usr/bin/env python3
"""
Main test runner for the Grateful Dead show dating project.
"""

import os
import sys
import argparse
import unittest
import traceback

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

def run_transformer_tests():
    """Run all transformer tests."""
    print("=" * 80)
    print("TRANSFORMER ENCODER TEST SUITE")
    print("=" * 80)
    
    # Import test modules
    from tests.test_transformer_simple import test_transformer_encoder
    from tests.test_transformer import run_transformer_sanity_test
    from tests.test_transformer_detailed import run_detailed_transformer_test
    
    # Track success/failure
    all_success = True
    
    # Run component test
    try:
        print("\n\nRunning component test...")
        test_transformer_encoder()
        print("✅ Component test passed!")
        component_success = True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        traceback.print_exc()
        component_success = False
        all_success = False
    
    # Run sanity test
    try:
        print("\n\nRunning sanity test...")
        sanity_success = run_transformer_sanity_test()
        if sanity_success:
            print("✅ Sanity test passed!")
        else:
            print("❌ Sanity test failed!")
            all_success = False
    except Exception as e:
        print(f"❌ Sanity test failed: {e}")
        traceback.print_exc()
        sanity_success = False
        all_success = False
    
    # Run detailed test
    try:
        print("\n\nRunning detailed test...")
        detailed_success = run_detailed_transformer_test()
        if detailed_success:
            print("✅ Detailed test passed!")
        else:
            print("❌ Detailed test failed!")
            all_success = False
    except Exception as e:
        print(f"❌ Detailed test failed: {e}")
        traceback.print_exc()
        detailed_success = False
        all_success = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRANSFORMER TEST SUMMARY")
    print("=" * 80)
    print(f"Component test: {'PASSED' if component_success else 'FAILED'}")
    print(f"Basic sanity test: {'PASSED' if sanity_success else 'FAILED'}")
    print(f"Detailed test: {'PASSED' if detailed_success else 'FAILED'}")
    print("=" * 80)
    
    return all_success


def run_model_tests():
    """Run all model tests."""
    print("=" * 80)
    print("MODEL TEST SUITE")
    print("=" * 80)

    try:
        # Import test modules
        from src.models import DeadShowDatingModel
        import torch
        
        # Create a small model for testing
        print("Creating test model...")
        model = DeadShowDatingModel(sample_rate=16000)
        
        # Test forward pass with dummy data
        print("Testing forward pass...")
        batch_size = 2
        dummy_input = {
            'harmonic': torch.randn(batch_size, 1, 128, 50),
            'percussive': torch.randn(batch_size, 1, 128, 50),
            'chroma': torch.randn(batch_size, 12, 50),
            'spectral_contrast': torch.randn(batch_size, 6, 50),
            'date': torch.tensor([100, 200])
        }
        
        outputs = model(dummy_input)
        
        # Verify expected outputs
        expected_keys = ['days', 'log_variance', 'era_logits', 'audio_features', 
                         'seasonal_features', 'date', 'year_logits']
        for key in expected_keys:
            assert key in outputs, f"Missing expected output key: {key}"
        
        print("✅ Model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        traceback.print_exc()
        return False


def discover_and_run_unittest_tests():
    """Discover and run all unittest-based tests."""
    print("=" * 80)
    print("UNITTEST SUITE")
    print("=" * 80)
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Discover tests in the tests directory
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report summary
    print("\n" + "=" * 80)
    print(f"Ran {result.testsRun} tests")
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 80)
    
    return result.wasSuccessful()


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for the Grateful Dead show dating project")
    parser.add_argument("--transformer", action="store_true", help="Run transformer tests")
    parser.add_argument("--model", action="store_true", help="Run model tests")
    parser.add_argument("--unittest", action="store_true", help="Run unittest-based tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no specific tests are specified, run all tests
    if not (args.transformer or args.model or args.unittest):
        args.all = True
    
    success = True
    
    if args.all or args.transformer:
        transformer_success = run_transformer_tests()
        success = success and transformer_success
        print("\n")
    
    if args.all or args.model:
        model_success = run_model_tests()
        success = success and model_success
        print("\n")
    
    if args.all or args.unittest:
        unittest_success = discover_and_run_unittest_tests()
        success = success and unittest_success
    
    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())