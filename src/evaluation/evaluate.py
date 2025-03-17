#!/usr/bin/env python3
"""
Evaluation script for the Grateful Dead show dating model.
"""

import argparse
import datetime
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from data.dataset import DeadShowDataset, optimized_collate_fn
from inference.base_inference import predict_date
from inference.utils.model_loader import load_model
from evaluation.metrics import (
    calculate_metrics,
    plot_era_performance,
    plot_uncertainty_analysis,
    plot_temporal_error_distribution
)

def evaluate_model(
    model_path: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = "cuda",
    enable_uncertainty: bool = True,
    uncertainty_samples: int = 30,
) -> Dict:
    """
    Evaluate model performance on a dataset.
    
    Args:
        model_path: Path to model checkpoint
        data_dir: Path to evaluation data directory
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        enable_uncertainty: Whether to enable uncertainty estimation
        uncertainty_samples: Number of Monte Carlo samples for uncertainty
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, base_date = load_model(model_path, device)
    model.eval()
    
    # Load dataset
    dataset = DeadShowDataset(
        root_dir=data_dir,
        base_date=base_date,
        transform=None,  # No augmentation during evaluation
        preload=False,  # Load on demand to save memory
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=optimized_collate_fn,
        pin_memory=True
    )
    
    # Lists to store predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Run inference
    print("Running inference...")
    for batch in dataloader:
        audio_paths = batch['audio_path']
        labels = batch['label']
        
        # Process each audio file in the batch
        for audio_path, label in zip(audio_paths, labels):
            prediction = predict_date(
                model=model,
                audio_path=audio_path,
                base_date=base_date,
                device=device,
                enable_uncertainty=enable_uncertainty,
                uncertainty_samples=uncertainty_samples
            )
            
            all_predictions.append(prediction)
            all_ground_truth.append(label)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(all_predictions, all_ground_truth)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    print("Generating visualizations...")
    plot_era_performance(
        all_predictions,
        all_ground_truth,
        save_path=os.path.join(output_dir, 'era_performance.png')
    )
    
    plot_uncertainty_analysis(
        all_predictions,
        all_ground_truth,
        save_path=os.path.join(output_dir, 'uncertainty_analysis.png')
    )
    
    plot_temporal_error_distribution(
        all_predictions,
        all_ground_truth,
        save_path=os.path.join(output_dir, 'temporal_error.png')
    )
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"MAE: {metrics['mae']:.2f} days")
    print(f"RMSE: {metrics['rmse']:.2f} days")
    print(f"R² Score: {metrics['r2']:.3f}")
    print(f"Era Classification Accuracy: {metrics['era_accuracy']:.3f}")
    print(f"Era Macro F1: {metrics['era_macro_f1']:.3f}")
    
    if 'uncertainty_calibration' in metrics:
        print(f"Uncertainty Calibration Score: {metrics['uncertainty_calibration']:.3f}")
        print(f"Error-Uncertainty Correlation: {metrics['error_uncertainty_correlation']:.3f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Grateful Dead show dating model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to evaluation data directory")
    parser.add_argument("--output", required=True, help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device to run evaluation on")
    parser.add_argument("--disable-uncertainty", action="store_true", help="Disable uncertainty estimation")
    parser.add_argument("--uncertainty-samples", type=int, default=30, help="Number of MC samples for uncertainty")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
        enable_uncertainty=not args.disable_uncertainty,
        uncertainty_samples=args.uncertainty_samples
    )

if __name__ == "__main__":
    main() 