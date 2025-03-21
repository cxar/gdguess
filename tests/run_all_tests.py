#!/usr/bin/env python3
"""
Run all tests to validate the Grateful Dead Show Dating project.
Includes tests with real data from the audsnippets-all collection.
"""
import os
import sys
import unittest
import subprocess
import argparse
import time
from pathlib import Path

def main():
    """Run all tests with options for different test types."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests for the GDGuess project")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--setup", action="store_true", help="Run only setup validation tests")
    parser.add_argument("--model", action="store_true", help="Run only model structure tests")
    parser.add_argument("--training", action="store_true", help="Run only training tests")
    parser.add_argument("--transformer", action="store_true", help="Run only transformer tests")
    parser.add_argument("--real-data", action="store_true", help="Run tests with real data (no synthetic)")
    parser.add_argument("--synthetic", action="store_true", help="Run tests with synthetic data (no real data)")
    parser.add_argument("--quick", action="store_true", help="Run only essential tests (faster)")
    parser.add_argument("--optimized", action="store_true", help="Run tests for optimized training pipeline")
    
    args = parser.parse_args()
    
    # If no specific test is specified, run all tests
    run_all = not (args.setup or args.model or args.training or args.transformer or 
                  args.real_data or args.synthetic or args.optimized)
    if args.all:
        run_all = True
    
    print("\n" + "="*80)
    print("GRATEFUL DEAD SHOW DATING PROJECT - TEST SUITE")
    print("="*80)
    
    # Directory containing this script
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test steps
    test_steps = []
    
    # Setup validation tests
    if args.setup or run_all:
        test_steps.append({
            "name": "Setup Validation",
            "script": os.path.join(os.path.dirname(tests_dir), "test_setup.py"),
            "description": "Validating project setup"
        })
    
    # Synthetic data tests
    if not args.real_data and (args.synthetic or run_all):
        # Only create synthetic test data if we're running synthetic tests
        if not args.quick:
            test_steps.append({
                "name": "Create Test Data",
                "script": os.path.join(tests_dir, "create_test_data.py"),
                "description": "Creating synthetic test data"
            })
        
        test_steps.append({
            "name": "Model Structure Tests",
            "script": os.path.join(tests_dir, "test_model_structure.py"),
            "description": "Testing model structure with synthetic data"
        })
        
        if not args.quick:
            test_steps.append({
                "name": "Training Tests",
                "script": os.path.join(tests_dir, "test_training.py"),
                "description": "Testing training functionality with synthetic data"
            })
            
    # Optimized training tests (only run if explicitly requested or part of all tests but not in quick mode)
    if (args.optimized or (run_all and not args.quick)):
        test_steps.append({
            "name": "Optimized Training Tests",
            "script": os.path.join(tests_dir, "test_training.py"),
            "args": ["TestTrainingFunctionality.test_optimized_gpu_training_functions"],
            "description": "Testing GPU-accelerated optimized training pipeline"
        })
    
    # Transformer tests
    if args.transformer or run_all:
        test_steps.append({
            "name": "Transformer Component Test",
            "script": os.path.join(tests_dir, "run_transformer_tests.py"),
            "description": "Testing transformer encoder component"
        })
    
    # Real data tests
    if args.real_data or run_all:
        test_steps.append({
            "name": "Real Data Model Test",
            "script": os.path.join(tests_dir, "real_data_test.py"),
            "description": "Testing model with real audio data"
        })
        
        test_steps.append({
            "name": "Transformer Real Data Test",
            "script": os.path.join(tests_dir, "test_transformer_with_real_data.py"),
            "description": "Testing transformer with real audio data"
        })
        
        test_steps.append({
            "name": "Preprocessed Data Test",
            "script": os.path.join(tests_dir, "test_preprocessed_data.py"),
            "description": "Testing model with preprocessed data from audsnippets-all/preprocessed"
        })
    
    # Run each test step
    all_passed = True
    results = {}
    total_start_time = time.time()
    
    for step in test_steps:
        print(f"\n=== {step['name']} ===")
        print(f"{step['description']}...")
        
        start_time = time.time()
        try:
            # Run the test script
            cmd = [sys.executable, step['script']]
            
            # Add any test-specific arguments if provided
            if 'args' in step:
                cmd.extend(step['args'])
                
            result = subprocess.run(
                cmd,
                capture_output=True, 
                text=True
            )
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            
            # Store result
            success = result.returncode == 0
            results[step['name']] = {"success": success, "time": elapsed}
            
            # Check if the test passed
            if success:
                print(f"✅ {step['name']} PASSED ({elapsed:.2f}s)")
            else:
                print(f"❌ {step['name']} FAILED ({elapsed:.2f}s)")
                print("\nError output:")
                print(result.stderr)
                all_passed = False
                
                # Also print stdout for debugging
                print("\nStandard output:")
                print(result.stdout)
                
        except Exception as e:
            # Calculate elapsed time
            elapsed = time.time() - start_time
            results[step['name']] = {"success": False, "time": elapsed}
            
            print(f"❌ {step['name']} ERROR: {e} ({elapsed:.2f}s)")
            all_passed = False
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for step_name, result in results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{step_name}: {status} ({result['time']:.2f}s)")
    
    print("-"*80)
    print(f"Total time: {total_elapsed:.2f}s")
    
    if all_passed:
        print("✅ ALL TESTS PASSED! The project is correctly configured.")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
    
    print("="*80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())