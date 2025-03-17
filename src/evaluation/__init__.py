"""
Evaluation module for the Grateful Dead show dating model.

This module provides comprehensive evaluation metrics and visualization tools
for analyzing model performance, including:

1. Advanced metrics beyond MAE:
   - RMSE
   - R² score
   - Percentile errors
   - Era-specific F1 scores

2. Era and year-based performance analysis:
   - Era classification accuracy
   - Era-specific error distributions
   - Temporal error patterns

3. Uncertainty correlation analysis:
   - Uncertainty calibration
   - Error-uncertainty correlation
   - Confidence interval coverage
"""

from .metrics import (
    calculate_metrics,
    plot_era_performance,
    plot_uncertainty_analysis,
    plot_temporal_error_distribution
)

from .evaluate import evaluate_model

__all__ = [
    "calculate_metrics",
    "plot_era_performance",
    "plot_uncertainty_analysis",
    "plot_temporal_error_distribution",
    "evaluate_model"
] 