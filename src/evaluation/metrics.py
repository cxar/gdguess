#!/usr/bin/env python3
"""
Comprehensive evaluation metrics for the Grateful Dead show dating model.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def calculate_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: List of prediction dictionaries containing 'days', 'era', etc.
        ground_truth: List of ground truth dictionaries
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Extract arrays for calculations
    pred_days = np.array([p['days_offset'] for p in predictions])
    true_days = np.array([g['days'] for g in ground_truth])
    
    pred_eras = np.array([p['raw_era_logits'].argmax() for p in predictions])
    true_eras = np.array([g['era'] for g in ground_truth])
    
    # Basic regression metrics
    metrics['mae'] = np.mean(np.abs(pred_days - true_days))
    metrics['rmse'] = np.sqrt(np.mean((pred_days - true_days) ** 2))
    metrics['r2'] = r2_score(true_days, pred_days)
    
    # Percentile errors
    errors = np.abs(pred_days - true_days)
    for p in [50, 75, 90, 95]:
        metrics[f'error_p{p}'] = np.percentile(errors, p)
    
    # Era classification metrics
    era_acc = (pred_eras == true_eras).mean()
    metrics['era_accuracy'] = era_acc
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_eras, pred_eras)
    era_f1s = []
    for i in range(5):  # 5 eras
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        era_f1s.append(f1)
        metrics[f'era_{i}_f1'] = f1
    
    metrics['era_macro_f1'] = np.mean(era_f1s)
    
    # Uncertainty metrics
    if 'date_uncertainty' in predictions[0]:
        # Calculate calibration score
        in_interval = 0
        total = 0
        for pred, true in zip(predictions, ground_truth):
            uncertainty = pred['date_uncertainty']
            if (uncertainty['lower_bound'].toordinal() <= true['date'].toordinal() <= 
                uncertainty['upper_bound'].toordinal()):
                in_interval += 1
            total += 1
        metrics['uncertainty_calibration'] = in_interval / total
        
        # Uncertainty correlation with error
        uncertainties = np.array([p['date_uncertainty']['std_days'] for p in predictions])
        error_uncertainty_corr = np.corrcoef(errors, uncertainties)[0, 1]
        metrics['error_uncertainty_correlation'] = error_uncertainty_corr
    
    return metrics

def plot_era_performance(
    predictions: List[Dict],
    ground_truth: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization of model performance across different eras.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        save_path: Optional path to save the plot
    """
    era_names = ['Early', '72-74', '75-79', '80-90', 'Later']
    
    # Prepare data
    era_errors = {era: [] for era in range(5)}
    for pred, true in zip(predictions, ground_truth):
        error = abs(pred['days_offset'] - true['days'])
        era_errors[true['era']].append(error)
    
    # Calculate statistics
    era_stats = []
    for era in range(5):
        errors = era_errors[era]
        if errors:
            stats = {
                'Era': era_names[era],
                'Mean Error': np.mean(errors),
                'Median Error': np.median(errors),
                'Std Error': np.std(errors)
            }
            era_stats.append(stats)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame(era_stats)
    
    # Bar plot with error bars
    plt.bar(df['Era'], df['Mean Error'], yerr=df['Std Error'], capsize=5)
    plt.title('Model Performance Across Eras')
    plt.xlabel('Era')
    plt.ylabel('Mean Absolute Error (days)')
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_uncertainty_analysis(
    predictions: List[Dict],
    ground_truth: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization of uncertainty analysis.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        save_path: Optional path to save the plot
    """
    if 'date_uncertainty' not in predictions[0]:
        return
    
    # Extract data
    uncertainties = np.array([p['date_uncertainty']['std_days'] for p in predictions])
    errors = np.abs([p['days_offset'] - g['days'] for p, g in zip(predictions, ground_truth)])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot of uncertainty vs error
    ax1.scatter(uncertainties, errors, alpha=0.5)
    ax1.set_xlabel('Predicted Uncertainty (days)')
    ax1.set_ylabel('Absolute Error (days)')
    ax1.set_title('Uncertainty vs Error Correlation')
    
    # Add trend line
    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    ax1.plot(uncertainties, p(uncertainties), "r--", alpha=0.8)
    
    # Calibration plot
    confidence_levels = np.arange(0.1, 1.1, 0.1)
    empirical_coverage = []
    
    for conf in confidence_levels:
        intervals = [(p['date_uncertainty']['lower_bound'].toordinal(),
                     p['date_uncertainty']['upper_bound'].toordinal())
                    for p in predictions]
        true_dates = [g['date'].toordinal() for g in ground_truth]
        
        # Calculate coverage at this confidence level
        in_interval = sum(1 for (lower, upper), true in zip(intervals, true_dates)
                         if lower <= true <= upper)
        empirical_coverage.append(in_interval / len(true_dates))
    
    ax2.plot([0, 1], [0, 1], 'r--', label='Ideal calibration')
    ax2.plot(confidence_levels, empirical_coverage, 'b-', label='Model calibration')
    ax2.set_xlabel('Expected Confidence Level')
    ax2.set_ylabel('Empirical Coverage')
    ax2.set_title('Uncertainty Calibration Plot')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_temporal_error_distribution(
    predictions: List[Dict],
    ground_truth: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization of error distribution across time.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        save_path: Optional path to save the plot
    """
    # Prepare data
    dates = [g['date'] for g in ground_truth]
    errors = np.abs([p['days_offset'] - g['days'] for p, g in zip(predictions, ground_truth)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Error': errors
    })
    df = df.sort_values('Date')
    
    # Calculate rolling statistics
    window = 30  # 30-day window
    rolling_mean = df['Error'].rolling(window=window, center=True).mean()
    rolling_std = df['Error'].rolling(window=window, center=True).std()
    
    # Create plot
    plt.figure(figsize=(15, 6))
    
    # Plot rolling mean with confidence interval
    plt.plot(df['Date'], rolling_mean, 'b-', label='30-day Rolling Mean Error')
    plt.fill_between(df['Date'],
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    alpha=0.2,
                    color='b',
                    label='±1 Std Dev')
    
    plt.title('Temporal Distribution of Prediction Error')
    plt.xlabel('Date')
    plt.ylabel('Mean Absolute Error (days)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close() 