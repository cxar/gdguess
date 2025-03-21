#!/usr/bin/env python3
"""
Test script to verify that the GDGuess project setup works correctly
after removing alternate implementations.
"""

import os
import sys
import subprocess
import importlib.util

# Check that gdguess.py is executable
def test_gdguess_executable():
    gdguess_path = os.path.join(os.path.dirname(__file__), "gdguess.py")
    if not os.path.exists(gdguess_path):
        print("ERROR: gdguess.py not found!")
        return False
    
    if not os.access(gdguess_path, os.X_OK):
        print("WARNING: gdguess.py is not executable. Setting executable bit.")
        os.chmod(gdguess_path, 0o755)
    
    print("✓ gdguess.py is executable")
    return True

# Check that we can import key modules
def test_imports():
    success = True
    
    # Add src directory to the path
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    sys.path.insert(0, src_dir)
    
    try:
        from src.config import Config
        print("✓ Successfully imported Config")
    except ImportError as e:
        print(f"ERROR: Failed to import Config: {e}")
        success = False
    
    try:
        from src.models.dead_model import DeadShowDatingModel
        print("✓ Successfully imported DeadShowDatingModel")
    except ImportError as e:
        print(f"ERROR: Failed to import DeadShowDatingModel: {e}")
        success = False
    
    try:
        importlib.util.find_spec("src.training.trainer")
        print("✓ Found training.trainer module")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"ERROR: Failed to locate training.trainer module: {e}")
        success = False
    
    return success

# Check that unified_training.py exists
def test_unified_training():
    unified_path = os.path.join(os.path.dirname(__file__), "unified_training.py")
    if not os.path.exists(unified_path):
        print("ERROR: unified_training.py not found!")
        return False
    
    if not os.access(unified_path, os.X_OK):
        print("WARNING: unified_training.py is not executable. Setting executable bit.")
        os.chmod(unified_path, 0o755)
    
    print("✓ unified_training.py exists")
    return True

# Run gdguess.py sysinfo to test basic functionality
def test_gdguess_sysinfo():
    gdguess_path = os.path.join(os.path.dirname(__file__), "gdguess.py")
    try:
        result = subprocess.run([gdguess_path, "sysinfo"], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ gdguess.py sysinfo ran successfully")
            return True
        else:
            print(f"ERROR: gdguess.py sysinfo failed with code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("ERROR: gdguess.py sysinfo timed out")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run gdguess.py sysinfo: {e}")
        return False

# Check that removed files are really gone
def test_removed_files():
    removed_files = [
        "gdguess_core.py",
        "minimal_core.py",
        "minimal_test.py",
        "train_simple.py",
        "implementation_guide.py",
    ]
    
    success = True
    for file_path in removed_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"ERROR: {file_path} still exists!")
            success = False
        else:
            print(f"✓ {file_path} successfully removed")
    
    # Check that core directory is removed
    core_dir = os.path.join(os.path.dirname(__file__), "src", "core")
    if os.path.exists(core_dir):
        print("ERROR: src/core/ directory still exists!")
        success = False
    else:
        print("✓ src/core/ directory successfully removed")
    
    return success

def main():
    print("\n=== Testing GDGuess Project Setup ===\n")
    
    tests = [
        ("Checking gdguess.py executable", test_gdguess_executable),
        ("Testing imports", test_imports),
        ("Checking unified_training.py", test_unified_training),
        ("Testing gdguess.py sysinfo", test_gdguess_sysinfo),
        ("Verifying removed files", test_removed_files),
    ]
    
    all_passed = True
    for name, test_func in tests:
        print(f"\n{name}...")
        result = test_func()
        if not result:
            all_passed = False
    
    print("\n=== Test Summary ===")
    if all_passed:
        print("\n✅ All tests passed! The project is correctly configured with a single implementation.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())