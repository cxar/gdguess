#!/usr/bin/env python3
"""
Create synthetic test data for the tests.
"""
import os
import torch
import numpy as np
from pathlib import Path

def create_synthetic_data_file(output_path, sample_date="1977-05-08"):
    """
    Create a synthetic data file with the specified sample date.
    
    Args:
        output_path: Path to save the synthetic data file
        sample_date: Date string in YYYY-MM-DD format
    """
    # Parse date
    year, month, day = map(int, sample_date.split("-"))
    
    # Calculate days since base date (Jan 1, 1965)
    import datetime
    base_date = datetime.date(1965, 1, 1)
    sample_date_obj = datetime.date(year, month, day)
    days = (sample_date_obj - base_date).days
    
    # Create synthetic data
    data = {
        # Standard audio features
        'harmonic': torch.randn(1, 128, 128),  # [batch, freq, time]
        'percussive': torch.randn(1, 128, 128),  # [batch, freq, time]
        'chroma': torch.randn(1, 12, 128),  # [batch, chroma_bins, time]
        'spectral_contrast': torch.randn(1, 7, 128),  # [batch, contrast_bands, time]
        
        # Label (days since base date)
        'label': torch.tensor([float(days)]),
        
        # Extra metadata
        'date': sample_date,
        'file_path': f"/path/to/audio/gd_{sample_date}.flac",
        'sample_rate': 44100,
        'duration': 60.0  # seconds
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save data
    torch.save(data, output_path)
    print(f"Created synthetic data file: {output_path}")

def main():
    # Directory to save test data
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create a few sample dates
    sample_dates = [
        "1969-02-11",  # Early era
        "1977-05-08",  # Cornell '77 (classic era)
        "1989-07-07",  # Late era
    ]
    
    # Create synthetic data files
    for date in sample_dates:
        year, month, day = date.split("-")
        filename = f"gd_{date}.pt"
        output_path = os.path.join(test_data_dir, filename)
        create_synthetic_data_file(output_path, date)
    
    print(f"Created {len(sample_dates)} synthetic data files in {test_data_dir}")

if __name__ == "__main__":
    main()