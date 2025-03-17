#!/usr/bin/env python3
"""
Example demonstrating how to use the uncertainty estimation functionality 
for the Grateful Dead show dating model.
"""

import argparse
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ..models.dead_model import DeadShowDatingModel
from ..inference.base_inference import predict_date


def main():
    parser = argparse.ArgumentParser(description="Demonstrate uncertainty estimation")
    parser.add_argument("audio_path", help="Path to audio file to analyze")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda, cpu, mps)")
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Loading model from {args.checkpoint} on {device}...")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DeadShowDatingModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Set base date (adjust if your model uses a different date)
    base_date = datetime.date(1968, 1, 1)
    
    # Run prediction with direct uncertainty output
    print(f"Analyzing {args.audio_path} with direct uncertainty output...")
    result_direct = predict_date(
        model=model,
        audio_path=args.audio_path,
        base_date=base_date,
        device=device
    )
    
    # Run prediction with Monte Carlo Dropout uncertainty
    print(f"Analyzing {args.audio_path} with Monte Carlo Dropout uncertainty...")
    result_mc = predict_date(
        model=model,
        audio_path=args.audio_path,
        base_date=base_date,
        device=device,
        enable_uncertainty=True,
        uncertainty_samples=30
    )
    
    # Display results
    print("\n====== PREDICTION RESULTS ======")
    print(f"Predicted date: {result_direct['predicted_date'].strftime('%Y-%m-%d')}")
    
    # Display direct uncertainty if available
    if "date_uncertainty" in result_direct:
        uncertainty = result_direct["date_uncertainty"]
        print("\n----- Direct Uncertainty -----")
        print(f"Standard deviation: ±{uncertainty['std_days']:.1f} days")
        print(f"95% confidence interval: {uncertainty['lower_bound'].strftime('%Y-%m-%d')} to {uncertainty['upper_bound'].strftime('%Y-%m-%d')}")
        print(f"Confidence score: {uncertainty['confidence_score']}%")
    
    # Display Monte Carlo uncertainty
    if "date_uncertainty" in result_mc:
        uncertainty = result_mc["date_uncertainty"]
        print("\n----- Monte Carlo Uncertainty -----")
        print(f"Standard deviation: ±{uncertainty['std_days']:.1f} days")
        print(f"95% confidence interval: {uncertainty['lower_bound'].strftime('%Y-%m-%d')} to {uncertainty['upper_bound'].strftime('%Y-%m-%d')}")
        print(f"Confidence score: {uncertainty['confidence_score'] if 'confidence_score' in uncertainty else 'N/A'}")
    
    if "predicted_era" in result_direct:
        print(f"\nPredicted era: {result_direct['predicted_era']}")
        print(f"Era confidence: {result_direct['era_confidence']:.2f}")
        
        if "era_uncertainty" in result_mc:
            print(f"Era entropy (uncertainty): {result_mc['era_uncertainty']['entropy']:.3f}")
    
    if "samples" in result_mc:
        # Visualize uncertainty with histogram of predictions
        plt.figure(figsize=(12, 6))
        
        # Get samples
        days_samples = result_mc["samples"]["days"].cpu().numpy()
        
        # Plot histogram
        plt.hist(days_samples, bins=20, alpha=0.7, color='blue')
        
        # Add vertical line for mean prediction
        mean_days = result_mc["days_offset"]
        plt.axvline(x=mean_days, color='red', linestyle='--', 
                   label=f'Mean prediction: {result_mc["predicted_date"].strftime("%Y-%m-%d")}')
        
        # Add vertical lines for confidence interval
        plt.axvline(x=result_mc["date_uncertainty"]["lower_bound"].toordinal() - base_date.toordinal(), 
                   color='green', linestyle=':', 
                   label=f'95% CI Lower: {result_mc["date_uncertainty"]["lower_bound"].strftime("%Y-%m-%d")}')
        plt.axvline(x=result_mc["date_uncertainty"]["upper_bound"].toordinal() - base_date.toordinal(), 
                   color='green', linestyle=':', 
                   label=f'95% CI Upper: {result_mc["date_uncertainty"]["upper_bound"].strftime("%Y-%m-%d")}')
        
        plt.xlabel('Days since base date')
        plt.ylabel('Frequency')
        plt.title('Monte Carlo Dropout Predictions Histogram')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_path = 'uncertainty_visualization.png'
        plt.savefig(output_path)
        print(f"\nUncertainty visualization saved to {output_path}")
    
    print("===============================")


if __name__ == "__main__":
    main() 