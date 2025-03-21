#!/usr/bin/env python3
"""
Ultra-simple entry point for single-process training.
This avoids all multiprocessing pickling issues and is the most robust option
for training the Grateful Dead show dating model.
"""

import sys
import os
from src.core.train.simple_single_process import main

if __name__ == "__main__":
    print("Starting robust single-process training...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nTry using a smaller batch size or fewer workers.")