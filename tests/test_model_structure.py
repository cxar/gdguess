#!/usr/bin/env python3
"""
Test the model structure to ensure it works with the data format.
"""
import os
import sys
import unittest
import torch
import glob
import random

# Add project root directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.dead_model import DeadShowDatingModel


class TestModelStructure(unittest.TestCase):
    def setUp(self):
        # Initialize model for testing
        self.model = DeadShowDatingModel()
        
        # Find data directory that contains .pt files
        self.data_dir = self._find_data_dir()
        
    def _find_data_dir(self):
        """Find a directory with .pt files to test against."""
        # Try a few common locations
        possible_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "data"),
            os.path.join(os.path.dirname(__file__), "..", "output"),
            os.path.join(os.path.dirname(__file__), "..", "tests", "test_data"),
        ]
        
        for directory in possible_dirs:
            if os.path.exists(directory):
                pt_files = glob.glob(os.path.join(directory, "**/*.pt"), recursive=True)
                if pt_files:
                    return directory
        
        # If no directory with .pt files is found, return None
        return None
    
    def _get_sample_batch(self):
        """Create a sample batch for testing."""
        # Try to use test_data directory first
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        if os.path.exists(test_data_dir):
            pt_files = glob.glob(os.path.join(test_data_dir, "**/*.pt"), recursive=True)
            if pt_files:
                sample_file = random.choice(pt_files)
                print(f"Using test file: {sample_file}")
                try:
                    data = torch.load(sample_file, map_location='cpu')
                    # Add batch dimension if needed
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor) and value.dim() >= 2:
                            if key in ['harmonic', 'percussive']:
                                # Should be [batch, channels, freq, time]
                                if value.dim() == 2:  # [freq, time]
                                    data[key] = value.unsqueeze(0).unsqueeze(0)
                                elif value.dim() == 3:  # [batch/channels, freq, time]
                                    data[key] = value.unsqueeze(1) if value.shape[0] == 1 else value.unsqueeze(0)
                            elif key in ['chroma', 'spectral_contrast']:
                                # Should be [batch, channels, bins, time]
                                if value.dim() == 2:  # [bins, time]
                                    data[key] = value.unsqueeze(0).unsqueeze(0)
                                elif value.dim() == 3:  # [batch/channels, bins, time]
                                    data[key] = value.unsqueeze(1) if value.shape[0] == 1 else value.unsqueeze(0)
                    return data
                except Exception as e:
                    print(f"Error loading {sample_file}: {e}")
        
        # If test data not found or loading failed, create synthetic data
        print("Using synthetic data for testing")
        return {
            'harmonic': torch.randn(1, 1, 128, 128),
            'percussive': torch.randn(1, 1, 128, 128),
            'chroma': torch.randn(1, 1, 12, 128),
            'spectral_contrast': torch.randn(1, 1, 7, 128),
            'label': torch.tensor([5000.0])
        }
    
    def test_model_initialization(self):
        """Test that the model can be initialized."""
        self.assertIsInstance(self.model, DeadShowDatingModel)
    
    def test_model_forward_pass(self):
        """Test that the model can complete a forward pass."""
        # Get a sample batch
        batch = self._get_sample_batch()
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Check outputs
        self.assertIsInstance(outputs, dict)
        self.assertIn('days', outputs)
        
        # Check that days is a tensor
        self.assertIsInstance(outputs['days'], torch.Tensor)
        
        # Check uncertainty estimation
        self.assertIn('log_variance', outputs)
        self.assertIsInstance(outputs['log_variance'], torch.Tensor)
    
    def test_model_output_shape(self):
        """Test that the model outputs have the correct shape."""
        # Use synthetic data for this test
        batch = {
            'harmonic': torch.randn(3, 1, 128, 128),
            'percussive': torch.randn(3, 1, 128, 128),
            'chroma': torch.randn(3, 1, 12, 128),
            'spectral_contrast': torch.randn(3, 1, 7, 128),
            'label': torch.tensor([5000.0, 6000.0, 7000.0])
        }
        
        # Get batch size
        batch_size = 3  # Using fixed batch size
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Check output shapes
        self.assertEqual(outputs['days'].shape[0], batch_size)
        self.assertEqual(outputs['log_variance'].shape[0], batch_size)
    
    def test_model_with_partial_features(self):
        """Test that the model can handle partial feature inputs."""
        # Create a synthetic batch with only harmonic and chroma features
        partial_batch = {
            'harmonic': torch.randn(1, 1, 128, 128),
            'chroma': torch.randn(1, 1, 12, 128),
            'label': torch.tensor([5000.0])
        }
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(partial_batch)
        
        # Check outputs
        self.assertIsInstance(outputs, dict)
        self.assertIn('days', outputs)
        self.assertIn('log_variance', outputs)
    
    def test_model_with_device_placement(self):
        """Test that the model can be placed on different devices."""
        # Test CPU placement
        cpu_model = DeadShowDatingModel(device='cpu')
        self.assertEqual(next(cpu_model.parameters()).device.type, 'cpu')
        
        # Only test CUDA if available
        if torch.cuda.is_available():
            cuda_model = DeadShowDatingModel(device='cuda')
            self.assertEqual(next(cuda_model.parameters()).device.type, 'cuda')
        
        # Only test MPS if available
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            mps_model = DeadShowDatingModel(device='mps')
            self.assertEqual(next(mps_model.parameters()).device.type, 'mps')


if __name__ == '__main__':
    unittest.main()