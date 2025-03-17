#!/usr/bin/env python3
"""
Interactive inference with microphone input for the Grateful Dead show dating model.
This script loads a trained model and uses the laptop microphone to capture audio for real-time inference.
"""

import argparse
import datetime
import sys
import time
import torch
import torch.nn.functional as F
import pygame
from pathlib import Path

# Import local modules
from .mic_capture import MicrophoneCapture
from .visualization import initialize_pygame, draw_results
from ..utils.model_loader import load_model
from ..base_inference import predict_date


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Grateful Dead Show Dating Inference with Microphone"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto will use MPS on Apple Silicon if available)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Audio sample rate (default: 24000)",
    )
    parser.add_argument(
        "--inference_interval",
        type=float,
        default=1.0,
        help="Interval between inferences in seconds (default: 1.0)",
    )

    return parser.parse_args()


def predict_from_audio(model, audio_array, base_date, device):
    """Run inference on audio array directly."""
    # Convert to tensor and add batch dimension if needed
    if isinstance(audio_array, torch.Tensor):
        audio = audio_array
    else:
        audio = torch.tensor(audio_array, dtype=torch.float)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        
    audio = audio.to(device)

    # Use the more detailed predict_date function from base_inference but with direct audio input
    prediction = predict_date(
        model=model,
        audio_path=None,  # Not using a file path
        base_date=base_date,
        target_sr=24000,
        device=device,
        audio_tensor=audio,  # Passing the audio tensor directly
    )
    
    return prediction


def main():
    """Main entry point for interactive inference."""
    args = parse_arguments()

    # Determine device
    if args.device == "cpu":
        device = torch.device("cpu")
        print("Using CPU for inference as requested")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for inference")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) for inference")
    elif args.device == "auto":
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

    # Load the model
    try:
        model, base_date = load_model(args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Initialize visualization
    screen, width, height, title_font, header_font, text_font = initialize_pygame()

    # Start microphone capture
    mic = MicrophoneCapture(sample_rate=args.sample_rate)
    mic.start_recording()

    # Wait a moment to let the buffer fill
    print("Letting audio buffer fill for 3 seconds...")
    time.sleep(3)

    # Main loop
    prediction = None
    last_inference_time = 0
    running = True

    print("Ready! Listening for Grateful Dead music...")

    try:
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Run inference at specified intervals
            current_time = time.time()
            if current_time - last_inference_time >= args.inference_interval:
                # Get current audio from buffer
                audio_buffer = mic.get_audio()

                try:
                    # Run prediction
                    prediction = predict_from_audio(
                        model, audio_buffer, base_date, device
                    )
                    last_inference_time = current_time

                    # Print results to console
                    print("\n--- Prediction Results ---")
                    print(f"Date: {prediction['predicted_date'].strftime('%B %d, %Y')}")
                    print(f"Era: {prediction['era_name']}")
                    print("Era Probabilities:")
                    for era, prob in prediction["era_probabilities"].items():
                        print(f"  - {era}: {prob}")
                    print("-------------------------\n")

                except Exception as e:
                    print(f"Error during inference: {e}")

            # Update display
            draw_results(
                screen,
                width,
                height,
                title_font,
                header_font,
                text_font,
                prediction,
                audio_buffer,
            )

            # Limit frame rate
            pygame.time.wait(50)  # ~20 FPS

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Clean up
        mic.stop_recording()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main()) 