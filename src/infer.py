#!/usr/bin/env python3
"""
Command-line tool for running inference with the Grateful Dead show dating model.
This script provides a convenient interface to both file-based and interactive inference.
"""

import argparse
import sys
import datetime
import torch
from pathlib import Path

from inference.utils.model_loader import load_model
from inference.base_inference import predict_date, batch_predict


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Grateful Dead Show Dating Inference"
    )
    
    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Inference mode")
    
    # File mode
    file_parser = subparsers.add_parser("file", help="Predict date from audio file")
    file_parser.add_argument(
        "--input", type=str, required=True, help="Path to audio file"
    )
    file_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    file_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    file_parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Enable test-time augmentation for more robust predictions",
    )
    file_parser.add_argument(
        "--tta-transforms",
        type=int,
        default=5,
        help="Number of transforms to use for test-time augmentation (default: 5)",
    )
    file_parser.add_argument(
        "--tta-intensity",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Intensity of test-time augmentation (default: medium)",
    )
    file_parser.add_argument(
        "--display_uncertainty",
        action="store_true",
        help="Display uncertainty information in the output if available",
    )
    
    # Batch mode
    batch_parser = subparsers.add_parser("batch", help="Process multiple audio files")
    batch_parser.add_argument("input", help="Directory containing audio files")
    batch_parser.add_argument(
        "--output", "-o", help="Output CSV file path", default="predictions.csv"
    )
    batch_parser.add_argument(
        "--pattern", help="Glob pattern for audio files", default="*.wav"
    )
    batch_parser.add_argument(
        "--checkpoint",
        help="Model checkpoint path",
        required=True,
    )
    batch_parser.add_argument(
        "--device",
        help="Device to run inference on (cuda, cpu, mps)",
        default="cuda",
    )
    batch_parser.add_argument(
        "--batch_size", help="Batch size for processing", type=int, default=16
    )
    batch_parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test-time augmentation for more robust predictions",
    )
    batch_parser.add_argument(
        "--tta_transforms",
        type=int,
        default=5,
        help="Number of TTA transforms to use (default: 5)",
    )
    batch_parser.add_argument(
        "--tta_intensity",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Intensity of test-time augmentation (default: medium)",
    )
    batch_parser.add_argument(
        "--display_uncertainty",
        action="store_true",
        help="Display uncertainty information in the output if available",
    )
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive inference with microphone")
    interactive_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    interactive_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    interactive_parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Audio sample rate (default: 24000)",
    )
    interactive_parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Enable test-time augmentation for more robust predictions",
    )
    interactive_parser.add_argument(
        "--tta-transforms",
        type=int,
        default=3,
        help="Number of transforms to use for test-time augmentation (default: 3)",
    )
    interactive_parser.add_argument(
        "--tta-intensity",
        type=str,
        default="light",
        choices=["light", "medium", "heavy"],
        help="Intensity of test-time augmentation (default: light)",
    )
    
    return parser.parse_args()


def determine_device(device_arg):
    """Determine which device to use based on argument and availability."""
    if device_arg == "cpu":
        device = torch.device("cpu")
        print("Using CPU for inference as requested")
    elif device_arg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for inference")
    elif device_arg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) for inference")
    elif device_arg == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon) for inference")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA for inference")
        else:
            device = torch.device("cpu")
            print("Using CPU for inference (no GPU acceleration available)")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference")
        
    return device


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.mode == "file":
        # File-based inference
        device = determine_device(args.device)
        
        try:
            # Load model
            model = load_model(args.checkpoint, device)
            
            # Show TTA info if enabled
            if args.use_tta:
                print(f"Using test-time augmentation with {args.tta_transforms} transforms at {args.tta_intensity} intensity")
            
            # Run prediction
            result = predict_date(
                model=model,
                audio_path=args.input,
                device=device,
                use_tta=args.use_tta,
                tta_transforms=args.tta_transforms,
                tta_intensity=args.tta_intensity
            )
            
            # Display results
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("\n====== PREDICTION RESULTS ======")
                print(f"Predicted date: {result['predicted_date'].strftime('%Y-%m-%d')}")
                
                # Display uncertainty if available and requested
                if args.display_uncertainty and "date_uncertainty" in result:
                    uncertainty = result["date_uncertainty"]
                    print("\n----- Date Uncertainty -----")
                    print(f"Standard deviation: ±{uncertainty['std_days']:.1f} days")
                    print(f"95% confidence interval: {uncertainty['lower_bound'].strftime('%Y-%m-%d')} to {uncertainty['upper_bound'].strftime('%Y-%m-%d')}")
                    print(f"Confidence score: {uncertainty['confidence_score']}%")
                
                if "predicted_era" in result:
                    print(f"Predicted era: {result['predicted_era']}")
                    print(f"Era confidence: {result['era_confidence']:.2f}")
                print("===============================\n")
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.mode == "batch":
        # Batch inference
        device = determine_device(args.device)
        
        try:
            # Load model
            model = load_model(args.checkpoint, device)
            
            # Find audio files in directory
            input_dir = Path(args.input)
            audio_files = list(input_dir.glob(args.pattern))
            
            if not audio_files:
                print(f"No audio files found in {args.input}")
                return
            
            print(f"Found {len(audio_files)} audio files for processing")
            
            # Show TTA info if enabled
            if args.use_tta:
                print(f"Using test-time augmentation with {args.tta_transforms} transforms at {args.tta_intensity} intensity")
            
            # Run batch prediction
            results = batch_predict(
                model=model,
                audio_paths=[str(f) for f in audio_files],
                device=device,
                batch_size=args.batch_size,
                use_tta=args.use_tta,
                tta_transforms=args.tta_transforms,
                tta_intensity=args.tta_intensity
            )
            
            # Write results to CSV
            import csv
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'predicted_date']
                
                # Add uncertainty fields if present in results
                has_uncertainty = results and 'date_uncertainty' in results[0]
                if has_uncertainty:
                    fieldnames.extend(['std_days', 'lower_bound', 'upper_bound', 'confidence_score'])
                
                # Add era field if present in results
                if results and 'predicted_era' in results[0]:
                    fieldnames.append('predicted_era')
                    fieldnames.append('era_confidence')
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, result in enumerate(results):
                    row = {
                        'filename': audio_files[i].name,
                        'predicted_date': result['predicted_date'].strftime('%Y-%m-%d')
                    }
                    
                    # Add uncertainty information if available
                    if 'date_uncertainty' in result:
                        uncertainty = result['date_uncertainty']
                        row['std_days'] = f"{uncertainty['std_days']:.1f}"
                        row['lower_bound'] = uncertainty['lower_bound'].strftime('%Y-%m-%d')
                        row['upper_bound'] = uncertainty['upper_bound'].strftime('%Y-%m-%d')
                        row['confidence_score'] = f"{uncertainty['confidence_score']}"
                    
                    if 'predicted_era' in result:
                        row['predicted_era'] = result['predicted_era']
                        row['era_confidence'] = f"{result['era_confidence']:.2f}"
                    
                    writer.writerow(row)
            
            print(f"Results written to {args.output}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.mode == "interactive":
        # Interactive inference
        try:
            from inference.interactive.mic_interface import run_interactive_inference
            
            device = determine_device(args.device)
            
            run_interactive_inference(
                checkpoint_path=args.checkpoint,
                device=device,
                sample_rate=args.sample_rate,
                use_tta=args.use_tta,
                tta_transforms=args.tta_transforms,
                tta_intensity=args.tta_intensity
            )
        except ImportError:
            print("Error: Unable to import interactive module. Make sure PyAudio is installed.")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Please specify a mode: file, batch, or interactive")
        sys.exit(1)


if __name__ == "__main__":
    main() 