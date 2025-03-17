#!/usr/bin/env python3
"""
Detailed test for the Transformer encoder implementation in the DeadShowDatingModel.
This script hooks into the model to examine the Transformer encoder's outputs.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dead_model import DeadShowDatingModel
from src.utils.helpers import reset_parameters

class HookedTransformerTester:
    """Class to test Transformer encoders by hooking into their forward passes."""
    
    def __init__(self, model: DeadShowDatingModel, device: torch.device):
        """Initialize the tester with a model."""
        self.model = model
        self.device = device
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        # Clear existing hooks and activations
        self.remove_hooks()
        self.activations = {}
        
        # Register hook for positional encoding
        def hook_pos_encoder(module, input, output):
            self.activations['pos_encoder_output'] = output.detach().clone()
            # Also save the input for comparison
            self.activations['pos_encoder_input'] = input[0].detach().clone()
        
        h = self.model.pos_encoder.register_forward_hook(hook_pos_encoder)
        self.hooks.append(h)
        
        # Register hook for transformer output
        def hook_transformer(module, input, output):
            self.activations['transformer_output'] = output.detach().clone()
            self.activations['transformer_input'] = input[0].detach().clone()
        
        h = self.model.transformer_encoder.register_forward_hook(hook_transformer)
        self.hooks.append(h)
        
        # Register hook for temporal pooling
        def hook_temporal_pooling(module, input, output):
            self.activations['temporal_pooling_output'] = output.detach().clone()
            self.activations['temporal_pooling_input'] = input[0].detach().clone()
        
        h = self.model.temporal_pooling.register_forward_hook(hook_temporal_pooling)
        self.hooks.append(h)
        
        # Register hook for self-attention
        def hook_self_attention(module, input, output):
            self.activations['self_attention_output'] = output[0].detach().clone()
            self.activations['self_attention_input'] = input[0].detach().clone()
        
        h = self.model.self_attention.register_forward_hook(hook_self_attention)
        self.hooks.append(h)
        
        # Register hook for frequency-time attention
        def hook_freq_time_attention(module, input, output):
            self.activations['freq_time_attention_output'] = output.detach().clone()
            self.activations['freq_time_attention_input'] = input[0].detach().clone()
        
        h = self.model.freq_time_attention.register_forward_hook(hook_freq_time_attention)
        self.hooks.append(h)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def run_test(self, batch_size: int = 2, audio_length: int = 24000 * 15):
        """
        Run a forward pass and capture activations.
        
        Args:
            batch_size: Number of samples in the batch
            audio_length: Length of audio in samples
            
        Returns:
            Tuple of (success, outputs, activations)
        """
        # Register hooks
        self.register_hooks()
        
        # Create random input
        print(f"\nCreating random audio input: [batch_size={batch_size}, audio_length={audio_length}]")
        audio = torch.randn(batch_size, audio_length, device=self.device, dtype=torch.float32)
        
        # Run forward pass
        print("\nRunning forward pass with hooks...")
        with torch.no_grad():
            try:
                outputs = self.model(audio)
                success = True
            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
                success = False
                outputs = None
        
        # Remove hooks after forward pass
        self.remove_hooks()
        
        return success, outputs, self.activations
    
    def analyze_transformer_outputs(self):
        """Analyze the captured transformer outputs."""
        if not self.activations:
            print("No activations captured. Run test first.")
            return False
        
        print("\nAnalyzing Transformer outputs...")
        
        # Check that positional encoding modifies the input
        if 'pos_encoder_input' in self.activations and 'pos_encoder_output' in self.activations:
            pos_in = self.activations['pos_encoder_input']
            pos_out = self.activations['pos_encoder_output']
            pos_diff = torch.abs(pos_in - pos_out).mean().item()
            
            print(f"Positional encoding difference: {pos_diff:.6f}")
            if pos_diff > 0:
                print("✅ Positional encoding is modifying the input")
            else:
                print("❌ Positional encoding is not having any effect")
        
        # Check that transformer is modifying the input
        if 'transformer_input' in self.activations and 'transformer_output' in self.activations:
            trans_in = self.activations['transformer_input']
            trans_out = self.activations['transformer_output']
            trans_diff = torch.abs(trans_in - trans_out).mean().item()
            
            print(f"Transformer difference: {trans_diff:.6f}")
            if trans_diff > 0:
                print("✅ Transformer encoder is modifying the input")
            else:
                print("❌ Transformer encoder is not having any effect")
        
        # Check for NaN values
        has_nans = False
        for name, tensor in self.activations.items():
            if torch.isnan(tensor).any():
                print(f"❌ NaN values detected in {name}")
                has_nans = True
        
        if not has_nans:
            print("✅ No NaN values detected in any activation")
        
        # Check shapes
        print("\nTensor shapes:")
        for name, tensor in self.activations.items():
            print(f"  {name}: {tensor.shape}")
        
        # Check attention patterns
        if 'self_attention_output' in self.activations:
            attn_out = self.activations['self_attention_output']
            print(f"\nSelf-attention output stats:")
            print(f"  Min: {attn_out.min().item():.6f}")
            print(f"  Max: {attn_out.max().item():.6f}")
            print(f"  Mean: {attn_out.mean().item():.6f}")
            print(f"  Std: {attn_out.std().item():.6f}")
        
        # Calculate correlation matrix to visualize attention patterns
        if 'transformer_output' in self.activations:
            # Sample the first batch item, first position
            trans_out = self.activations['transformer_output'][0]
            # Correlation between positions in the sequence
            corr_matrix = self._calculate_correlation_matrix(trans_out)
            
            # Save the correlation matrix as an image
            self._plot_correlation_matrix(corr_matrix, 'transformer_correlation.png')
            print("\n✅ Saved attention correlation matrix to transformer_correlation.png")
        
        return True
    
    def _calculate_correlation_matrix(self, tensor):
        """Calculate correlation matrix between positions in sequence."""
        # Convert to numpy for easier correlation calculation
        array = tensor.cpu().numpy()
        seq_len, feat_dim = array.shape
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(array)
        return corr_matrix
    
    def _plot_correlation_matrix(self, corr_matrix, filename):
        """Plot and save correlation matrix."""
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('Position-wise Correlation in Transformer Output')
        plt.xlabel('Sequence Position')
        plt.ylabel('Sequence Position')
        plt.savefig(filename)
        plt.close()


def run_detailed_transformer_test():
    """Run a detailed test on the Transformer encoder implementation."""
    print("\n" + "="*80)
    print("DETAILED TRANSFORMER ENCODER TEST")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    print("\nInitializing model...")
    model = DeadShowDatingModel(sample_rate=24000)
    model.to(device)
    
    # Apply enhanced initialization
    reset_parameters(model)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create tester
    tester = HookedTransformerTester(model, device)
    
    # Run test
    success, outputs, activations = tester.run_test()
    
    if success:
        # Check basic outputs
        print("\nChecking model outputs...")
        
        # Check if we have the expected output keys
        expected_keys = ['days', 'era_logits']
        missing_keys = [key for key in expected_keys if key not in outputs]
        
        if missing_keys:
            print(f"❌ Missing expected keys in output: {missing_keys}")
        else:
            print(f"✅ All expected keys present in output")
        
        # Check output shapes
        batch_size = 2  # As set in run_test
        days_shape = outputs['days'].shape
        era_logits_shape = outputs['era_logits'].shape
        
        expected_days_shape = (batch_size,)
        expected_era_logits_shape = (batch_size, 5)
        
        if days_shape == expected_days_shape:
            print(f"✅ 'days' output has correct shape: {days_shape}")
        else:
            print(f"❌ 'days' output has wrong shape: {days_shape}, expected: {expected_days_shape}")
        
        if era_logits_shape == expected_era_logits_shape:
            print(f"✅ 'era_logits' output has correct shape: {era_logits_shape}")
        else:
            print(f"❌ 'era_logits' output has wrong shape: {era_logits_shape}, expected: {expected_era_logits_shape}")
        
        # Analyze transformer
        tester.analyze_transformer_outputs()
        
        print("\n" + "="*80)
        print("DETAILED TRANSFORMER TEST COMPLETED")
        print("="*80)
    
    return success

if __name__ == "__main__":
    run_detailed_transformer_test() 