#!/usr/bin/env python3
"""
Test the training code to ensure it works with the data format.
"""
import os
import sys
import unittest
import torch
import glob
import random
import importlib.util
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.dead_model import DeadShowDatingModel
from src.models import CombinedDeadLoss


class TestTrainingFunctionality(unittest.TestCase):
    
    def setUp(self):
        # Check if unified_training.py exists
        self.unified_training_path = os.path.join(project_root, "unified_training.py")
        self.training_exists = os.path.exists(self.unified_training_path)
        
        # Find data directory that contains .pt files
        self.data_dir = self._find_data_dir()
        
        # Create model and loss function
        self.model = DeadShowDatingModel(device='cpu')
        self.criterion = CombinedDeadLoss()
    
    def _find_data_dir(self):
        """Find a directory with .pt files to test against."""
        # Try a few common locations
        possible_dirs = [
            os.path.join(project_root, "data"),
            os.path.join(project_root, "output"),
            os.path.join(project_root, "tests", "test_data"),
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
        # Try to use the test_data directory with synthetic data
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        if os.path.exists(test_data_dir):
            pt_files = glob.glob(os.path.join(test_data_dir, "**/*.pt"), recursive=True)
            if pt_files:
                try:
                    # Load up to 3 random files
                    batch = {}
                    sample_files = random.sample(pt_files, min(3, len(pt_files)))
                    
                    for file_path in sample_files:
                        print(f"Loading test data: {file_path}")
                        data = torch.load(file_path, map_location='cpu')
                        
                        # Stack tensors
                        for key, value in data.items():
                            if isinstance(value, torch.Tensor):
                                if key not in batch:
                                    batch[key] = [value]
                                else:
                                    batch[key].append(value)
                    
                    # Convert lists to batched tensors
                    for key in list(batch.keys()):
                        if isinstance(batch[key], list):
                            try:
                                batch[key] = torch.stack(batch[key])
                            except:
                                # Remove keys that can't be stacked
                                print(f"Warning: Could not stack {key}, removing from batch")
                                del batch[key]
                    
                    # Add label if missing
                    if 'label' not in batch:
                        batch['label'] = torch.tensor([5000.0] * len(sample_files))
                    
                    return batch
                
                except Exception as e:
                    print(f"Error loading batch: {e}")
                    # Fall back to synthetic data
        
        # Create synthetic data
        print("Using synthetic data for testing")
        return {
            'harmonic': torch.randn(3, 1, 128, 128),
            'percussive': torch.randn(3, 1, 128, 128),
            'chroma': torch.randn(3, 1, 12, 128),
            'spectral_contrast': torch.randn(3, 1, 7, 128),
            'label': torch.tensor([5000.0, 6000.0, 7000.0])
        }
    
    def test_loss_computation(self):
        """Test that the loss function can compute loss."""
        # Get a sample batch
        batch = self._get_sample_batch()
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Compute loss
        loss_dict = self.criterion(
            outputs,
            {'days': batch.get('label')},
            global_step=0,
            total_steps=1000
        )
        
        # Check loss
        self.assertIn('loss', loss_dict)
        self.assertIsInstance(loss_dict['loss'], torch.Tensor)
        self.assertFalse(torch.isnan(loss_dict['loss']))
        self.assertFalse(torch.isinf(loss_dict['loss']))
    
    def test_optimizer_step(self):
        """Test that the optimizer can take a step."""
        # Set up optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Get a sample batch
        batch = self._get_sample_batch()
        
        # Record initial parameters
        initial_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.clone().detach()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Compute loss
        loss_dict = self.criterion(
            outputs,
            {'days': batch.get('label')},
            global_step=0,
            total_steps=1000
        )
        loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters have changed
        params_changed = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if not torch.allclose(initial_params[name], param):
                    params_changed = True
                    break
        
        self.assertTrue(params_changed, "Parameters did not change after optimizer step")
    
    def test_data_format_compatibility(self):
        """Test compatibility with synthetic test data format."""
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        if not os.path.exists(test_data_dir):
            self.skipTest("No test_data directory found")
        
        # Get paths to test data files
        pt_files = glob.glob(os.path.join(test_data_dir, "**/*.pt"), recursive=True)
        if not pt_files:
            self.skipTest("No .pt files found in test_data directory")
        
        # Test loading and processing each file
        for file_path in pt_files:
            try:
                # Load data
                data = torch.load(file_path, map_location='cpu')
                
                # Check that it's a dictionary
                self.assertIsInstance(data, dict)
                
                # Add batch dimension to tensors if needed
                processed_data = {}
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        if key in ['harmonic', 'percussive'] and value.dim() >= 2:
                            if value.dim() == 2:  # [freq, time]
                                processed_data[key] = value.unsqueeze(0).unsqueeze(0)
                            elif value.dim() == 3:  # Assume [batch/channel, freq, time]
                                processed_data[key] = value.unsqueeze(1) if value.shape[0] == 1 else value.unsqueeze(0)
                            else:
                                processed_data[key] = value
                        elif key in ['chroma', 'spectral_contrast'] and value.dim() >= 2:
                            if value.dim() == 2:  # [bins, time]
                                processed_data[key] = value.unsqueeze(0).unsqueeze(0)
                            elif value.dim() == 3:  # Assume [batch/channel, bins, time]
                                processed_data[key] = value.unsqueeze(1) if value.shape[0] == 1 else value.unsqueeze(0)
                            else:
                                processed_data[key] = value
                        else:
                            processed_data[key] = value
                    else:
                        processed_data[key] = value
                
                # If we don't have at least one feature, skip this file
                if not any(k in processed_data for k in ['harmonic', 'percussive', 'chroma', 'spectral_contrast']):
                    print(f"Skipping {file_path} because it doesn't have any usable features")
                    continue
                
                # Try a forward pass
                with torch.no_grad():
                    outputs = self.model(processed_data)
                
                # Check that we get valid outputs
                self.assertIn('days', outputs)
                self.assertIsInstance(outputs['days'], torch.Tensor)
                
            except Exception as e:
                print(f"Error details for {file_path}: {e}")
                self.fail(f"Error processing {file_path}: {e}")
    
    def test_import_unified_training(self):
        """Test that we can import functions from unified_training.py"""
        if not self.training_exists:
            self.skipTest("unified_training.py not found")
        
        # Import unified_training module dynamically
        unified_training_spec = importlib.util.spec_from_file_location(
            "unified_training", self.unified_training_path
        )
        unified_training = importlib.util.module_from_spec(unified_training_spec)
        unified_training_spec.loader.exec_module(unified_training)
        
        # Check for key components
        self.assertTrue(hasattr(unified_training, "create_dataloaders"))
        self.assertTrue(hasattr(unified_training, "OptimizedTrainer"))
    
    def test_optimized_trainer_initialization(self):
        """Test initializing the OptimizedTrainer class from unified_training.py"""
        if not self.training_exists:
            self.skipTest("unified_training.py not found")
        
        # Import dynamically
        unified_training_spec = importlib.util.spec_from_file_location(
            "unified_training", self.unified_training_path
        )
        unified_training = importlib.util.module_from_spec(unified_training_spec)
        unified_training_spec.loader.exec_module(unified_training)
        
        # Import Config
        from src.config import Config
        
        # Create a basic config
        config = Config()
        config.data_dir = self.data_dir or "/"
        config.total_training_steps = 100
        config.batch_size = 4
        config.initial_lr = 0.001
        config.output_dir = "./output"
        config.use_early_stopping = False
        config.patience = 5
        config.use_mixed_precision = False
        config.aggressive_memory = False
        config.checkpoint_interval = 100
        config.validation_interval = 50
        config.weight_decay = 1e-6
        config.use_augmentation = False
        config.grad_clip_value = 1.0
        config.target_sr = 22050
        config.prefetch_factor = 2
        
        # Create dummy dataloaders
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=10):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self._get_sample_batch()
            
            def _get_sample_batch(self):
                return {
                    'harmonic': torch.randn(1, 128, 128),
                    'percussive': torch.randn(1, 128, 128),
                    'chroma': torch.randn(1, 12, 128),
                    'spectral_contrast': torch.randn(1, 7, 128),
                    'label': torch.tensor([5000.0])
                }
        
        train_loader = torch.utils.data.DataLoader(DummyDataset(10), batch_size=4)
        val_loader = torch.utils.data.DataLoader(DummyDataset(5), batch_size=4)
        
        # Initialize trainer
        try:
            trainer = unified_training.OptimizedTrainer(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=torch.device('cpu'),
                tensorboard_dir=None,
                auto_tune=False
            )
            
            # Check that trainer initialized correctly
            self.assertIsInstance(trainer.model, DeadShowDatingModel)
            self.assertIsInstance(trainer.criterion, CombinedDeadLoss)
            self.assertIsInstance(trainer.optimizer, torch.optim.Optimizer)
            
        except Exception as e:
            self.fail(f"Failed to initialize OptimizedTrainer: {e}")
            
    def test_optimized_gpu_training_functions(self):
        """Test the optimized GPU training functions."""
        # Check if the optimized training module exists
        optimized_path = os.path.join(project_root, "src", "training", "run_unified_training.py")
        if not os.path.exists(optimized_path):
            self.skipTest("run_unified_training.py not found")
            
        # Import optimized training module dynamically
        spec = importlib.util.spec_from_file_location(
            "run_unified_training", optimized_path
        )
        optimized = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimized)
        
        # Check for key functions
        self.assertTrue(hasattr(optimized, "setup_logger"))
        self.assertTrue(hasattr(optimized, "get_optimal_device"))
        self.assertTrue(hasattr(optimized, "get_optimized_optimizer"))
        self.assertTrue(hasattr(optimized, "create_scheduler"))
        self.assertTrue(hasattr(optimized, "validate"))
        self.assertTrue(hasattr(optimized, "train"))
        self.assertTrue(hasattr(optimized, "evaluate"))
        self.assertTrue(hasattr(optimized, "main"))
        
        # Test device selection
        device = optimized.get_optimal_device()
        self.assertIsInstance(device, torch.device)
        
        # Test optimizer creation
        from src.config import Config
        config = Config()
        config.initial_lr = 0.001
        config.weight_decay = 1e-6
        
        model = DeadShowDatingModel(sample_rate=16000)
        optimizer = optimized.get_optimized_optimizer(model, config, torch.device('cpu'))
        
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertEqual(optimizer.param_groups[0]['lr'], config.initial_lr)


if __name__ == '__main__':
    unittest.main()