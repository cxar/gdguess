#!/usr/bin/env python3
"""
Loss functions for the Grateful Dead show dating model.
"""

from typing import Dict

import torch
import torch.nn.functional as F


def uncertainty_aware_loss(
    outputs: Dict[str, torch.Tensor],
    targets_label: torch.Tensor,
    targets_era: torch.Tensor,
    global_step: int = 0,
    total_steps: int = 180000,
) -> torch.Tensor:
    """
    Calculate loss that accounts for both prediction accuracy and uncertainty estimation.
    
    Args:
        outputs: Dictionary containing model outputs with keys 'days', 'log_variance', and 'era_logits'
        targets_label: Ground truth day labels
        targets_era: Ground truth era labels
        global_step: Current training step
        total_steps: Total training steps
        
    Returns:
        Combined loss value as a tensor
    """
    try:
        pred_days = outputs["days"]
        log_variance = outputs["log_variance"]
        
        # Ensure everything is on the same device
        target_device = targets_label.device
        if pred_days.device != target_device:
            pred_days = pred_days.to(target_device)
            
        if log_variance.device != target_device:
            log_variance = log_variance.to(target_device)

        if outputs["era_logits"].device != target_device:
            outputs["era_logits"] = outputs["era_logits"].to(target_device)

        # Handle NaN values in predictions
        if torch.isnan(pred_days).any():
            pred_days = torch.where(
                torch.isnan(pred_days), torch.zeros_like(pred_days), pred_days
            )
            
        if torch.isnan(log_variance).any():
            log_variance = torch.where(
                torch.isnan(log_variance), torch.ones_like(log_variance) * -3, log_variance
            )

        # Calculate precision (1/variance) from log_variance
        precision = torch.exp(-log_variance)
        
        # Calculate negative log likelihood loss
        # This balances prediction accuracy and well-calibrated uncertainty
        squared_error = (pred_days - targets_label) ** 2
        date_loss = 0.5 * precision * squared_error + 0.5 * log_variance
        date_loss = torch.mean(date_loss)
        
        # Use label smoothing for era classification to improve generalization
        era_loss = F.cross_entropy(
            outputs["era_logits"], targets_era, label_smoothing=0.1
        )

        # Dynamic weighting - gradually increase focus on date prediction
        # Start with more emphasis on era (easier to learn), then shift to date
        progress = min(1.0, global_step / (0.7 * total_steps))
        era_weight = max(0.1, 0.5 - 0.4 * progress)

        total_loss = date_loss + era_weight * era_loss

        if torch.isnan(total_loss):
            print("Warning: NaN detected in loss calculation!")
            return torch.tensor(0.1, device=target_device, requires_grad=True)

        return total_loss
    except Exception as e:
        print(f"Error in uncertainty-aware loss calculation: {e}")
        dummy_loss = torch.tensor(0.1, device=targets_label.device, requires_grad=True)
        return dummy_loss


def combined_loss(
    outputs: Dict[str, torch.Tensor],
    targets_label: torch.Tensor,
    targets_era: torch.Tensor,
    global_step: int = 0,
    total_steps: int = 180000,
) -> torch.Tensor:
    """
    Calculate the combined loss for date prediction and era classification
    with dynamic weighting based on training progress.

    Args:
        outputs: Dictionary containing model outputs with keys 'days' and 'era_logits'
        targets_label: Ground truth day labels
        targets_era: Ground truth era labels
        global_step: Current training step
        total_steps: Total training steps

    Returns:
        Combined loss value as a tensor
    """
    # If the model outputs uncertainty, use the uncertainty-aware loss
    if "log_variance" in outputs:
        return uncertainty_aware_loss(outputs, targets_label, targets_era, global_step, total_steps)
    
    try:
        pred_days = outputs["days"]

        # Ensure everything is on the same device
        target_device = targets_label.device
        if pred_days.device != target_device:
            pred_days = pred_days.to(target_device)

        if outputs["era_logits"].device != target_device:
            outputs["era_logits"] = outputs["era_logits"].to(target_device)

        # Handle NaN values in predictions
        if torch.isnan(pred_days).any():
            pred_days = torch.where(
                torch.isnan(pred_days), torch.zeros_like(pred_days), pred_days
            )

        # Calculate date loss with Huber loss which is more robust than MSE
        days_loss = F.huber_loss(pred_days, targets_label, delta=5.0)

        # Use label smoothing for era classification to improve generalization
        era_loss = F.cross_entropy(
            outputs["era_logits"], targets_era, label_smoothing=0.1
        )

        # Dynamic weighting - gradually increase focus on date prediction
        # Start with more emphasis on era (easier to learn), then shift to date
        progress = min(1.0, global_step / (0.7 * total_steps))
        era_weight = max(0.1, 0.5 - 0.4 * progress)

        total_loss = days_loss + era_weight * era_loss

        if torch.isnan(total_loss):
            print("Warning: NaN detected in loss calculation!")
            return torch.tensor(0.1, device=target_device, requires_grad=True)

        return total_loss
    except Exception as e:
        print(f"Error in loss calculation: {e}")
        dummy_loss = torch.tensor(0.1, device=targets_label.device, requires_grad=True)
        return dummy_loss
