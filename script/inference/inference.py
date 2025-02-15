#!/usr/bin/env python
import argparse

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# --- Configuration ---
TARGET_SR = 24000  # Target sample rate (24KHz)
DURATION = 5  # Duration in seconds (5 seconds)
BASE_YEAR = (
    1968  # The base year used in training (so predictions are offset by this value)
)


# --- Regression Head Definition ---
class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super(RegressionHead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def load_model(checkpoint_path, device):
    """
    Loads the feature extractor, base model, and regression head from checkpoint.
    """
    # Load the Wav2Vec2 processor (feature extractor)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    # Load the base model
    base_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    hidden_dim = base_model.config.hidden_size
    # Create the regression head
    regression_head = RegressionHead(hidden_dim)

    # Load checkpoint state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint["base_model_state_dict"])
    regression_head.load_state_dict(checkpoint["regression_head_state_dict"])

    base_model.to(device)
    regression_head.to(device)
    base_model.eval()
    regression_head.eval()

    return processor, base_model, regression_head


def preprocess_audio(audio_path, target_sr=TARGET_SR, duration=DURATION):
    """
    Loads an audio file, resamples it to target_sr,
    and ensures it is exactly `duration` seconds long.
    """
    y, sr = librosa.load(audio_path, sr=target_sr)
    desired_length = target_sr * duration
    if len(y) < desired_length:
        # Pad with zeros if too short
        y = np.pad(y, (0, desired_length - len(y)))
    elif len(y) > desired_length:
        # Truncate if too long
        y = y[:desired_length]
    return y


def predict_audio(audio_path, processor, base_model, regression_head, device):
    """
    Processes an audio file and queries the model.
    Returns the predicted year.
    """
    # Preprocess the audio (resample and adjust duration)
    y = preprocess_audio(audio_path, TARGET_SR, DURATION)

    # Convert the audio to a tensor; note that the processor expects a list of audio samples.
    audio_tensor = torch.tensor(y)

    # Process the audio with the feature extractor.
    inputs = processor(
        [audio_tensor], sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # Inference: Pass through the base model and regression head.
    with torch.no_grad():
        outputs = base_model(**inputs)
        hidden_states = (
            outputs.last_hidden_state
        )  # Shape: (batch, seq_length, hidden_dim)
        pooled = torch.mean(hidden_states, dim=1)
        pred_offset = regression_head(pooled).squeeze(
            -1
        )  # Regression output: predicted offset

    # The prediction is the offset (e.g., years since BASE_YEAR). Add BASE_YEAR to get the predicted year.
    predicted_year = BASE_YEAR + pred_offset.item()
    return predicted_year


def main():
    parser = argparse.ArgumentParser(description="Audio Regression Inference Script")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (.pt file)"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to audio file (e.g. .mp3)"
    )
    args = parser.parse_args()

    # Device selection: Prefer MPS, then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model components from the checkpoint.
    processor, base_model, regression_head = load_model(args.checkpoint, device)

    # Run inference on the provided audio file.
    predicted_year = predict_audio(
        args.audio, processor, base_model, regression_head, device
    )
    print("Predicted Year:", predicted_year)


if __name__ == "__main__":
    main()
