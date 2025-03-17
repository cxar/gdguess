"""
Loss functions for the Grateful Dead show dating model.
Includes uncertainty estimation and dynamic task weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UncertaintyLoss(nn.Module):
    """
    Implements uncertainty-aware regression loss with learned aleatoric uncertainty.
    Based on the paper "What Uncertainties Do We Need in Bayesian Deep Learning 
    for Computer Vision?" by Kendall & Gal, 2017.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_days: torch.Tensor, true_days: torch.Tensor, 
                log_variance: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-aware regression loss.
        
        Args:
            pred_days: Predicted days since base date
            true_days: Ground truth days since base date
            log_variance: Predicted log variance (uncertainty)
            
        Returns:
            Loss value incorporating uncertainty
        """
        # Compute precision (inverse variance)
        precision = torch.exp(-log_variance)
        
        # Compute squared error weighted by precision
        sq_error = (pred_days - true_days) ** 2
        uncertainty_loss = 0.5 * (precision * sq_error + log_variance)
        
        if self.reduction == 'mean':
            return uncertainty_loss.mean()
        elif self.reduction == 'sum':
            return uncertainty_loss.sum()
        else:
            return uncertainty_loss


class PeriodicityLoss(nn.Module):
    """
    Loss component that encourages the model to learn seasonal patterns
    in show dates through periodic regularization.
    """
    def __init__(self, period_days: float = 365.25):
        super().__init__()
        self.period_days = period_days

    def forward(self, pred_days: torch.Tensor, true_days: torch.Tensor) -> torch.Tensor:
        """
        Compute periodicity loss to encourage seasonal pattern learning.
        
        Args:
            pred_days: Predicted days since base date
            true_days: Ground truth days since base date
            
        Returns:
            Periodicity loss value
        """
        # Convert days to phase in [0, 2π]
        pred_phase = (2 * math.pi * pred_days / self.period_days) % (2 * math.pi)
        true_phase = (2 * math.pi * true_days / self.period_days) % (2 * math.pi)
        
        # Compute circular distance
        phase_diff = torch.abs(torch.sin(pred_phase - true_phase))
        
        return phase_diff.mean()


class DynamicTaskLoss(nn.Module):
    """
    Implements dynamic task weighting for multi-task learning based on
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    by Kendall et al., 2018.
    """
    def __init__(self):
        super().__init__()
        # Initialize log task weights (one per task)
        self.log_vars = nn.Parameter(torch.zeros(3))  # [regression, era, year]

    def forward(self, regression_loss: torch.Tensor, era_loss: torch.Tensor, 
                year_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted multi-task loss with learned weights.
        
        Args:
            regression_loss: Date regression loss
            era_loss: Era classification loss
            year_loss: Year classification loss
            
        Returns:
            Combined loss with learned weights
        """
        # Get precision terms
        precision = torch.exp(-self.log_vars)
        
        # Weight individual losses
        weighted_regression = precision[0] * regression_loss + 0.5 * self.log_vars[0]
        weighted_era = precision[1] * era_loss + 0.5 * self.log_vars[1]
        weighted_year = precision[2] * year_loss + 0.5 * self.log_vars[2]
        
        return weighted_regression + weighted_era + weighted_year


class CombinedDeadLoss(nn.Module):
    """
    Combined loss function incorporating uncertainty estimation,
    periodicity, and dynamic task weighting.
    """
    def __init__(self, period_days: float = 365.25):
        super().__init__()
        self.uncertainty_loss = UncertaintyLoss()
        self.periodicity_loss = PeriodicityLoss(period_days)
        self.dynamic_weighting = DynamicTaskLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Compute combined loss with all components.
        
        Args:
            predictions: Dict containing model predictions
                - days: Predicted days since base date
                - log_variance: Predicted uncertainty
                - era_logits: Era classification logits
                - year_logits: Year classification logits
            targets: Dict containing ground truth
                - days: True days since base date
                - era: True era labels
                - year: True year labels
                
        Returns:
            Dict containing total loss and individual components
        """
        # Compute individual losses
        regression_loss = self.uncertainty_loss(
            predictions['days'],
            targets['days'],
            predictions['log_variance']
        )
        
        periodicity_loss = self.periodicity_loss(
            predictions['days'],
            targets['days']
        )
        
        era_loss = self.ce_loss(predictions['era_logits'], targets['era'])
        year_loss = self.ce_loss(predictions['year_logits'], targets['year'])
        
        # Combine regression and periodicity losses
        total_regression = regression_loss + 0.1 * periodicity_loss
        
        # Apply dynamic task weighting
        total_loss = self.dynamic_weighting(
            total_regression,
            era_loss,
            year_loss
        )
        
        return {
            'loss': total_loss,
            'regression_loss': regression_loss,
            'periodicity_loss': periodicity_loss,
            'era_loss': era_loss,
            'year_loss': year_loss
        } 