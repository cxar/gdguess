#!/usr/bin/env python
import argparse
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# --- Configuration ---
TARGET_SR = 24000  # Target sample rate (24KHz)
DEFAULT_SEGMENT_DURATION = 5  # Duration (seconds) for each segment
DEFAULT_TOTAL_DURATION = 15  # Total duration to record (seconds)
BASE_YEAR = 1968  # Base year used during training


# --- Regression Head Definition ---
class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super(RegressionHead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def load_model(checkpoint_path, device):
    """
    Loads the Wav2Vec2 feature extractor, base model, and regression head from a checkpoint.
    """
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    base_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    hidden_dim = base_model.config.hidden_size
    regression_head = RegressionHead(hidden_dim)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint["base_model_state_dict"])
    regression_head.load_state_dict(checkpoint["regression_head_state_dict"])

    base_model.to(device)
    regression_head.to(device)
    base_model.eval()
    regression_head.eval()

    return processor, base_model, regression_head


def record_audio(duration, sample_rate):
    """
    Records audio from the microphone for a given duration (seconds) at sample_rate.
    Returns a 1D numpy array.
    """
    print(f"Recording {duration} seconds of audio at {sample_rate} Hz...")
    recording = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()  # Wait for the recording to complete
    return recording.flatten()


def preprocess_audio_segment(y, sample_rate, segment_duration):
    """
    Ensures the audio segment is exactly segment_duration seconds long.
    Pads with zeros if too short or truncates if too long.
    """
    desired_length = sample_rate * segment_duration
    if len(y) < desired_length:
        y = np.pad(y, (0, desired_length - len(y)))
    elif len(y) > desired_length:
        y = y[:desired_length]
    return y


def predict_segment(y_segment, processor, base_model, regression_head, device):
    """
    Processes a single audio segment (numpy array) and queries the model.
    Returns the predicted year for that segment.
    """
    # Pass the raw numpy array (in a list) to the feature extractor
    inputs = processor(
        [y_segment], sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = base_model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_length, hidden_dim)
        pooled = torch.mean(hidden_states, dim=1)
        pred_offset = regression_head(pooled).squeeze(-1)

    predicted_year = BASE_YEAR + pred_offset.item()
    return predicted_year


def main():
    parser = argparse.ArgumentParser(
        description="Mic Inference with Averaging and Saving Segments"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (.pt file)"
    )
    parser.add_argument(
        "--total_duration",
        type=float,
        default=DEFAULT_TOTAL_DURATION,
        help="Total recording duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=DEFAULT_SEGMENT_DURATION,
        help="Duration in seconds for each segment (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save recorded audio segments (WAV files)",
    )
    args = parser.parse_args()

    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection: Prefer MPS if available, then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model components from checkpoint.
    processor, base_model, regression_head = load_model(args.checkpoint, device)

    # Record total audio from the microphone.
    total_audio = record_audio(args.total_duration, TARGET_SR)

    # Split total audio into segments.
    segment_length = int(TARGET_SR * args.segment_duration)
    num_segments = int(len(total_audio) // segment_length)
    if num_segments == 0:
        print("Error: Recorded audio is shorter than one segment. Exiting.")
        return

    predictions = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = total_audio[start:end]
        segment = preprocess_audio_segment(segment, TARGET_SR, args.segment_duration)

        # Save each segment as a WAV file.
        output_filename = os.path.join(args.output_dir, f"segment_{i+1}.wav")
        sf.write(output_filename, segment, TARGET_SR)
        print(f"Saved segment {i+1} to {output_filename}")

        # Run prediction on the segment.
        pred_year = predict_segment(
            segment, processor, base_model, regression_head, device
        )
        predictions.append(pred_year)
        print(f"Segment {i+1}/{num_segments} predicted year: {pred_year:.2f}")

    avg_prediction = sum(predictions) / len(predictions)
    print(
        f"\nAverage predicted year over {num_segments} segments: {avg_prediction:.2f}"
    )


if __name__ == "__main__":
    main()
