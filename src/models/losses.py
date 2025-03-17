"""
Loss functions for the Grateful Dead show dating model.
Includes uncertainty estimation and dynamic task weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


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
    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Initialize log task weights (one per task)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        """
        Compute weighted multi-task loss with learned weights.
        
        Args:
            losses: List of individual task losses
            
        Returns:
            Combined loss with learned weights
        """
        # Ensure we have the right number of tasks
        assert len(losses) == len(self.log_vars), f"Expected {len(self.log_vars)} losses, got {len(losses)}"
        
        # Get precision terms
        precision = torch.exp(-self.log_vars)
        
        # Weight individual losses
        weighted_losses = [
            precision[i] * loss + 0.5 * self.log_vars[i] 
            for i, loss in enumerate(losses)
        ]
        
        # Sum up all weighted losses
        return sum(weighted_losses)


class CombinedDeadLoss(nn.Module):
    """
    Combined loss function incorporating uncertainty estimation,
    periodicity, and dynamic task weighting.
    """
    def __init__(self, period_days: float = 365.25):
        super().__init__()
        self.uncertainty_loss = UncertaintyLoss()
        self.periodicity_loss = PeriodicityLoss(period_days)
        self.dynamic_weighting = DynamicTaskLoss(num_tasks=3)  # regression, era, year
        self.ce_loss = nn.CrossEntropyLoss()
        self.periodicity_weight = 0.1

    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor], 
                global_step: Optional[int] = None,
                total_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
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
            global_step: Current training step (optional)
            total_steps: Total training steps (optional)
                
        Returns:
            Dict containing total loss and individual components
        """
        # Handle missing keys with graceful fallbacks
        pred_days = predictions.get('days')
        if pred_days is None:
            raise ValueError("Predictions must contain 'days' key")
            
        true_days = targets.get('days')
        if true_days is None:
            raise ValueError("Targets must contain 'days' key")
        
        # Make sure tensors match in dimension
        if pred_days.dim() == 2 and true_days.dim() == 1:
            true_days = true_days.unsqueeze(1)
        elif pred_days.dim() == 1 and true_days.dim() == 2:
            pred_days = pred_days.unsqueeze(1)
        
        # Compute regression loss with uncertainty
        log_variance = predictions.get('log_variance')
        if log_variance is not None:
            regression_loss = self.uncertainty_loss(pred_days, true_days, log_variance)
        else:
            # Fallback to Huber loss if no uncertainty provided
            regression_loss = F.huber_loss(pred_days, true_days, delta=5.0)
            
        # Compute periodicity loss
        periodicity_loss = self.periodicity_loss(pred_days, true_days)
        
        # Collect individual task losses
        losses = [regression_loss + self.periodicity_weight * periodicity_loss]
        individual_losses = {
            'regression_loss': regression_loss,
            'periodicity_loss': periodicity_loss
        }
        
        # Compute era loss if provided
        era_logits = predictions.get('era_logits')
        true_era = targets.get('era')
        if era_logits is not None and true_era is not None:
            era_loss = self.ce_loss(era_logits, true_era)
            losses.append(era_loss)
            individual_losses['era_loss'] = era_loss
        else:
            individual_losses['era_loss'] = torch.tensor(0.0, device=pred_days.device)
            
        # Compute year loss if provided
        year_logits = predictions.get('year_logits')
        true_year = targets.get('year')
        if year_logits is not None and true_year is not None:
            year_loss = self.ce_loss(year_logits, true_year)
            losses.append(year_loss)
            individual_losses['year_loss'] = year_loss
        else:
            individual_losses['year_loss'] = torch.tensor(0.0, device=pred_days.device)
            
        # Apply dynamic task weighting
        total_loss = self.dynamic_weighting(losses)
        
        # Combine all losses
        result = {'loss': total_loss}
        result.update(individual_losses)
        
        return result