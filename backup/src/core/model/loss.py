"""
Simplified loss functions for the Grateful Dead show dating model.
Focused on core uncertainty estimation for regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyLoss(nn.Module):
    """
    Implements uncertainty-aware regression loss with learned aleatoric uncertainty.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_days, true_days, log_variance):
        # Compute precision (inverse variance)
        precision = torch.exp(-log_variance)
        
        # Compute squared error weighted by precision
        sq_error = (pred_days - true_days) ** 2
        uncertainty_loss = 0.5 * (precision * sq_error + log_variance)
        
        return uncertainty_loss.mean()


class SimplifiedDeadLoss(nn.Module):
    """
    Simplified loss function focusing on uncertainty-based regression.
    """
    def __init__(self):
        super().__init__()
        self.uncertainty_loss = UncertaintyLoss()
        self.scale_factor = 10000.0  # For numerical stability
    
    def forward(self, predictions, targets, step=None, total_steps=None):
        # Extract predictions and targets
        pred_days = predictions.get('days')
        true_days = targets.get('days')
        log_variance = predictions.get('log_variance')
        
        if pred_days is None:
            raise ValueError("Predictions must contain 'days' key")
        if true_days is None:
            raise ValueError("Targets must contain 'days' key")
        
        # Standardize dimensions
        if pred_days.dim() == 2 and true_days.dim() == 1:
            true_days = true_days.unsqueeze(1)
        elif pred_days.dim() == 1 and true_days.dim() == 2:
            pred_days = pred_days.unsqueeze(1)
        
        # Scale values to avoid numerical issues
        scaled_true_days = true_days / self.scale_factor
        scaled_pred_days = pred_days  # Model predicts in [0, 1] range
        
        # Compute loss with uncertainty if available
        if log_variance is not None:
            loss = self.uncertainty_loss(scaled_pred_days, scaled_true_days, log_variance)
        else:
            # Fallback to Huber loss
            loss = F.huber_loss(scaled_pred_days, scaled_true_days, delta=0.1)
        
        return {'loss': loss, 'regression_loss': loss}