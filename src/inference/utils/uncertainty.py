#!/usr/bin/env python3
"""
Uncertainty estimation functions for model predictions.
"""

import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference for Monte Carlo Dropout.
    
    Args:
        model: PyTorch model with dropout layers
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()  # Set to train mode to enable dropout


def monte_carlo_dropout(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    num_samples: int = 30,
    device: Union[str, torch.device] = "cuda",
) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Run Monte Carlo Dropout to estimate prediction uncertainty.
    
    Args:
        model: PyTorch model
        inputs: Input tensor dictionary
        num_samples: Number of stochastic forward passes
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction results and uncertainty metrics
    """
    # Put model in eval mode but enable dropout
    model.eval()
    enable_dropout(model)
    
    # Save all MC samples
    days_samples = []
    era_logits_samples = []
    
    # Run multiple stochastic forward passes
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(inputs)
            days_samples.append(outputs["days"])
            era_logits_samples.append(outputs["era_logits"])
    
    # Stack all samples
    days_samples_tensor = torch.stack(days_samples, dim=0)  # [num_samples, batch_size]
    era_logits_samples_tensor = torch.stack(era_logits_samples, dim=0)  # [num_samples, batch_size, num_eras]
    
    # Mean predictions (point estimates)
    days_mean = torch.mean(days_samples_tensor, dim=0)
    
    # Standard deviation for days (epistemic uncertainty)
    days_std = torch.std(days_samples_tensor, dim=0)
    
    # 95% confidence interval for days (approximately 2 std deviations)
    days_lower = days_mean - 1.96 * days_std
    days_upper = days_mean + 1.96 * days_std
    
    # Era probabilities (average softmax outputs)
    era_probs_samples = F.softmax(era_logits_samples_tensor, dim=-1)
    era_probs_mean = torch.mean(era_probs_samples, dim=0)
    
    # Entropy of averaged probabilities (predictive uncertainty)
    era_entropy = -torch.sum(era_probs_mean * torch.log(era_probs_mean + 1e-10), dim=-1)
    
    # Most likely era
    era_pred = torch.argmax(era_probs_mean, dim=-1)
    
    # Uncertainty in era prediction (probability spread)
    # Use the standard deviation of the probabilities for the predicted class
    predicted_classes_indices = era_pred.unsqueeze(0).expand(num_samples, -1)
    batch_indices = torch.arange(era_probs_samples.shape[1], device=device).unsqueeze(0).expand(num_samples, -1)
    sample_indices = torch.arange(num_samples, device=device).unsqueeze(1).expand(-1, era_probs_samples.shape[1])
    
    # Get probabilities for predicted class across all samples
    predicted_class_probs = era_probs_samples[sample_indices, batch_indices, predicted_classes_indices]
    era_std = torch.std(predicted_class_probs, dim=0)
    
    return {
        "days": days_mean,
        "days_std": days_std,
        "days_lower": days_lower,
        "days_upper": days_upper,
        "era_logits": era_logits_samples_tensor[-1],  # Use last sample for logits
        "era_probs": era_probs_mean,
        "era_entropy": era_entropy,
        "era_std": era_std,
        "samples": {
            "days": days_samples_tensor,
            "era_probs": era_probs_samples
        }
    }


def estimate_date_uncertainty(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    base_date: datetime.date,
    num_samples: int = 30, 
    device: Union[str, torch.device] = "cuda",
) -> Dict:
    """
    Estimate uncertainty for date and era predictions.
    
    Args:
        model: PyTorch model
        inputs: Input tensor dictionary
        base_date: Base date for conversion
        num_samples: Number of Monte Carlo samples
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction results and uncertainty estimates
    """
    # Get Monte Carlo predictions
    mc_results = monte_carlo_dropout(model, inputs, num_samples, device)
    
    # Process results for each sample in the batch
    results = []
    batch_size = mc_results["days"].shape[0]
    
    for i in range(batch_size):
        # Extract per-sample results
        days_mean = mc_results["days"][i].item()
        days_std = mc_results["days_std"][i].item()
        days_lower = mc_results["days_lower"][i].item()
        days_upper = mc_results["days_upper"][i].item()
        
        # Convert days to dates
        mean_date = base_date + datetime.timedelta(days=int(days_mean))
        lower_date = base_date + datetime.timedelta(days=int(days_lower))
        upper_date = base_date + datetime.timedelta(days=int(days_upper))
        
        # Era prediction and uncertainty
        era_probs = mc_results["era_probs"][i]
        era_idx = torch.argmax(era_probs).item()
        era_conf = era_probs[era_idx].item()
        era_entropy = mc_results["era_entropy"][i].item()
        era_std = mc_results["era_std"][i].item()
        
        # Map index to era
        era_map = {0: "early", 1: "seventies", 2: "eighties", 3: "nineties", 4: "later"}
        era = era_map[era_idx]
        
        # Construct result dictionary
        result = {
            "predicted_date": mean_date,
            "days_offset": days_mean,
            "date_uncertainty": {
                "std_days": days_std,
                "lower_bound": lower_date,
                "upper_bound": upper_date,
                "confidence_days": days_upper - days_lower
            },
            "predicted_era": era,
            "era_confidence": era_conf,
            "era_uncertainty": {
                "entropy": era_entropy,
                "std": era_std
            }
        }
        
        results.append(result)
    
    # Return single result for single sample, list for batch
    if batch_size == 1:
        return results[0]
    return results 