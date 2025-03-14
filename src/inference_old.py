#!/usr/bin/env python3
"""
Grateful Dead Show Dating - Microphone Inference Script
===============================================
This script loads a trained DeadShowDatingModel checkpoint and uses the laptop microphone
to capture audio for real-time inference.

Usage:
    python dead_show_mic_inference.py --checkpoint path/to/checkpoint.pt
"""

import argparse
import datetime
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pyaudio
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


# ====================== MODEL DEFINITION ======================
class DeadShowDatingModel(nn.Module):
    def __init__(self, sample_rate: int = 24000):
        super(DeadShowDatingModel, self).__init__()

        # Audio preprocessing config
        self.sample_rate = sample_rate

        # Mel spectrogram transformation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000,
        )

        # Convolutional feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Era detection module
        self.era_classifier = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5),  # 5 main Grateful Dead eras
        )

        # Self-attention for temporal patterns
        self.self_attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)

        # Recurrent layer for temporal modeling
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Final regression for date prediction
        self.regressor = nn.Sequential(
            nn.Linear(256 * 2 + 5, 256),  # GRU output + era features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Predicts number of days since base date
        )

    def forward(self, x: Union[torch.Tensor, Dict]) -> Dict[str, torch.Tensor]:
        """Forward pass for the model, handles both raw audio and batch dictionary inputs"""
        try:
            # Get the device of the model
            model_device = next(self.parameters()).device

            # Handle both tensor inputs and dictionary batch inputs
            if isinstance(x, dict):
                # Check if we're using precomputed mel spectrograms
                if "mel_spec" in x:
                    # Use the precomputed mel spectrogram directly
                    mel_spec = x["mel_spec"].to(model_device, non_blocking=True)
                    batch_size = mel_spec.shape[0]

                    # Extract features directly from mel spec
                    features = self.feature_extractor(mel_spec)
                # Otherwise use audio
                elif "audio" in x:
                    audio = x["audio"].to(model_device, non_blocking=True)
                    batch_size = audio.shape[0]

                    # Convert to mel spectrogram
                    with torch.no_grad():
                        audio_mel = self.mel_spec(audio)
                        audio_mel = torch.log(audio_mel + 1e-4)
                        audio_mel = audio_mel.unsqueeze(1)  # Add channel dimension

                    features = self.feature_extractor(audio_mel)
                elif "raw_audio" in x:
                    audio = x["raw_audio"].to(model_device, non_blocking=True)
                    batch_size = audio.shape[0]

                    # Convert to mel spectrogram
                    with torch.no_grad():
                        audio_mel = self.mel_spec(audio)
                        audio_mel = torch.log(audio_mel + 1e-4)
                        audio_mel = audio_mel.unsqueeze(1)  # Add channel dimension

                    features = self.feature_extractor(audio_mel)
                else:
                    # Fallback case for older code paths
                    batch_size = len(x["label"])
                    dummy_days = torch.zeros((batch_size,), device=model_device)
                    dummy_logits = torch.zeros((batch_size, 5), device=model_device)
                    return {"days": dummy_days, "era_logits": dummy_logits}
            else:
                # Direct tensor input (e.g., from inference) - assume raw audio
                audio = x.to(model_device, non_blocking=True)
                batch_size = audio.shape[0]

                # Convert to mel spectrogram
                with torch.no_grad():
                    audio_mel = self.mel_spec(audio)
                    audio_mel = torch.log(audio_mel + 1e-4)
                    audio_mel = audio_mel.unsqueeze(1)  # Add channel dimension

                features = self.feature_extractor(audio_mel)

            # Classify era
            era_logits = self.era_classifier(features)
            era_features = F.softmax(era_logits, dim=1)

            if torch.isnan(era_features).any():
                era_features = torch.where(
                    torch.isnan(era_features),
                    torch.zeros_like(era_features),
                    era_features,
                )

            # Reshape for temporal processing
            b, c, h, w = features.shape
            features_temporal = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

            # Apply self-attention
            attn_output, _ = self.self_attention(
                features_temporal,
                features_temporal,
                features_temporal,
                need_weights=False,
            )

            # Process with GRU
            gru_out, hidden = self.gru(attn_output)

            # Get final hidden state
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)

            # Combine with era features for regression
            combined = torch.cat([hidden_concat, era_features], dim=1)
            days_prediction = self.regressor(combined)

            if torch.isnan(days_prediction).any():
                days_prediction = torch.where(
                    torch.isnan(days_prediction),
                    torch.zeros_like(days_prediction),
                    days_prediction,
                )

            return {"days": days_prediction.squeeze(-1), "era_logits": era_logits}
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Use the model's device for dummy tensors
            model_device = next(self.parameters()).device
            dummy_days = torch.zeros((batch_size,), device=model_device)
            dummy_logits = torch.zeros((batch_size, 5), device=model_device)
            return {"days": dummy_days, "era_logits": dummy_logits}


# ====================== AUDIO CAPTURE CLASS ======================
class MicrophoneCapture:
    """Class to handle microphone audio capture with a buffer."""

    def __init__(
        self, sample_rate=24000, chunk_size=1024, channels=1, buffer_seconds=15
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paFloat32
        self.buffer_seconds = buffer_seconds
        self.buffer_size = self.sample_rate * self.buffer_seconds

        # Initialize buffer with zeros
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # PyAudio instance
        self.p = pyaudio.PyAudio()

        # Stream variable will be set when we start streaming
        self.stream = None

        # Threading control
        self.is_recording = False
        self.thread = None

    def start_recording(self):
        """Start capturing audio from the microphone."""
        if self.is_recording:
            print("Already recording!")
            return

        self.is_recording = True

        # Open stream
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )

        print("Started recording from microphone")

        # Start stream
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Roll the buffer and add new data
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data) :] = audio_data

        return (in_data, pyaudio.paContinue)

    def get_audio(self):
        """Get the current audio buffer."""
        return self.audio_buffer.copy()

    def stop_recording(self):
        """Stop recording and close the audio stream."""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        print("Stopped recording")

    def __del__(self):
        """Clean up resources."""
        self.stop_recording()
        if self.p is not None:
            self.p.terminate()


# ====================== INFERENCE FUNCTIONS ======================
def load_model(checkpoint_path, device="cpu"):
    """Load the model from a checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Handle different possible checkpoint formats
    try:
        # First try with PyTorch 2.6+ approach using safe globals
        import torch.serialization

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([datetime.date])
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            # Fall back to older PyTorch versions
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
    except (TypeError, AttributeError):
        # Final fallback for even older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    model = DeadShowDatingModel(sample_rate=24000)

    # Handle different key formats in checkpoints
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint

    # Load state dict, handling DataParallel prefix if present
    if list(model_state.keys())[0].startswith("module."):
        # Model was saved using DataParallel
        model = nn.DataParallel(model)
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # Move to the specified device
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Get config from checkpoint if it exists
    config = checkpoint.get("config", {})
    base_date = config.get("base_date", datetime.date(1968, 1, 1))

    return model, base_date


def predict_from_audio(model, audio_array, base_date, device):
    """Run inference on an audio array."""
    model.eval()

    # Convert to tensor and add batch dimension
    audio = torch.tensor(audio_array, dtype=torch.float).unsqueeze(0)
    audio = audio.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(audio)

    # Process results
    pred_days = outputs["days"].item()
    pred_date = base_date + datetime.timedelta(days=int(pred_days))

    era_logits = outputs["era_logits"][0]
    pred_era = torch.argmax(era_logits).item()
    era_probs = F.softmax(era_logits, dim=0).cpu().numpy()

    era_names = {
        0: "Primal Dead (1965-1971)",
        1: "Europe 72 through Wall of Sound (1972-1974)",
        2: "Hiatus Return through Egypt (1975-1979)",
        3: "Brent Era (1980-1990)",
        4: "Bruce/Vince Era (1990-1995)",
    }

    return {
        "predicted_date": pred_date,
        "predicted_days_since_base": pred_days,
        "predicted_era": pred_era,
        "era_name": era_names[pred_era],
        "era_probabilities": {
            era_names[i]: f"{prob:.2%}" for i, prob in enumerate(era_probs)
        },
    }


# ====================== VISUALIZATION ======================
def initialize_pygame():
    """Initialize Pygame for visualization."""
    pygame.init()
    pygame.font.init()

    # Set up the window
    info = pygame.display.Info()
    width, height = min(1024, info.current_w), min(768, info.current_h)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Grateful Dead Show Dating")

    # Fonts
    title_font = pygame.font.SysFont("Arial", 28, bold=True)
    header_font = pygame.font.SysFont("Arial", 24, bold=True)
    text_font = pygame.font.SysFont("Arial", 20)

    return screen, width, height, title_font, header_font, text_font


def draw_results(
    screen, width, height, title_font, header_font, text_font, prediction, audio_buffer
):
    """Draw prediction results on the screen."""
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    DARK_GRAY = (100, 100, 100)
    BLUE = (50, 100, 200)
    GREEN = (50, 200, 100)

    # Background
    screen.fill(WHITE)

    # Title
    title = title_font.render("Grateful Dead Show Dating", True, BLACK)
    screen.blit(title, (width // 2 - title.get_width() // 2, 20))

    # Date prediction
    if prediction:
        date_text = header_font.render(
            f"Predicted Date: {prediction['predicted_date'].strftime('%B %d, %Y')}",
            True,
            BLUE,
        )
        screen.blit(date_text, (width // 2 - date_text.get_width() // 2, 70))

        era_text = header_font.render(f"Era: {prediction['era_name']}", True, BLUE)
        screen.blit(era_text, (width // 2 - era_text.get_width() // 2, 110))

        # Era probabilities
        prob_header = text_font.render("Era Probabilities:", True, BLACK)
        screen.blit(prob_header, (width // 4, 160))

        y_pos = 190
        for era, prob in prediction["era_probabilities"].items():
            era_prob_text = text_font.render(f"{era}: {prob}", True, BLACK)
            screen.blit(era_prob_text, (width // 4, y_pos))
            y_pos += 30

    # Draw audio waveform
    if audio_buffer is not None:
        # Downsample for display
        downsample_factor = 500
        downsampled = audio_buffer[::downsample_factor]

        # Scale for display
        if np.max(np.abs(downsampled)) > 0:
            downsampled = downsampled / np.max(np.abs(downsampled)) * (height // 4)

        # Draw waveform
        pygame.draw.line(screen, DARK_GRAY, (0, height - 100), (width, height - 100), 1)

        for i in range(1, len(downsampled)):
            x1 = (i - 1) * width / len(downsampled)
            y1 = height - 100 + downsampled[i - 1]
            x2 = i * width / len(downsampled)
            y2 = height - 100 + downsampled[i]
            pygame.draw.line(screen, GREEN, (x1, y1), (x2, y2), 2)

    # Instructions
    instructions = text_font.render("Press ESC to quit", True, BLACK)
    screen.blit(instructions, (width - instructions.get_width() - 20, height - 40))

    pygame.display.flip()


# ====================== MAIN APPLICATION ======================
def main():
    # Parse command-line arguments
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

    args = parser.parse_args()

    # Determine device - prioritize MPS on Apple Silicon by default
    if args.device == "cpu":
        device = torch.device("cpu")
        print("Using CPU for inference as requested")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for inference")
    elif torch.backends.mps.is_available():
        # Use MPS automatically on Apple Silicon
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) for inference")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference (no GPU acceleration available)")

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
