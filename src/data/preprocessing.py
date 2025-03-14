#!/usr/bin/env python3
"""
Data preprocessing utilities for the Grateful Dead show dating model.
"""

import glob
import os
import platform
from typing import Dict

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DeadShowDataset, identity_collate


def preprocess_dataset(
    config: Dict, force_preprocess: bool = False, store_audio: bool = False
) -> str:
    """
    Preprocess the entire dataset once to avoid repeated CPU work.

    Args:
        config: Configuration dictionary
        force_preprocess: Whether to force preprocessing even if files exist
        store_audio: Whether to store audio files alongside preprocessed data

    Returns:
        Path to the preprocessed data directory
    """
    preprocessed_dir = os.path.join(config["input_dir"], "preprocessed")

    # Skip if already preprocessed
    if (
        os.path.exists(preprocessed_dir)
        and os.listdir(preprocessed_dir)
        and not force_preprocess
    ):
        print(f"Using existing preprocessed data at {preprocessed_dir}")
        return preprocessed_dir

    print(f"Preprocessing dataset to {preprocessed_dir}...")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Create original dataset for preprocessing
    orig_dataset = DeadShowDataset(
        root_dir=config["input_dir"],
        base_date=config["base_date"],
        target_sr=config["target_sr"],
        augment=False,  # No augmentation during preprocessing
        verbose=True,
    )

    # Set up preprocessing dataloader with platform awareness
    is_mac = platform.system() == "Darwin"
    num_workers = 0 if is_mac else max(1, os.cpu_count() - 2)

    preprocess_loader = DataLoader(
        orig_dataset,
        batch_size=1,  # Process one at a time for simplicity
        num_workers=num_workers,
        shuffle=False,
        collate_fn=identity_collate,
        persistent_workers=num_workers > 0,
    )

    # Create mel spectrogram transform
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=config["target_sr"],
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=20,
        f_max=8000,
    )

    # Set up audio storage (if needed)
    audio_dir = os.path.join(preprocessed_dir, "audio_files") if store_audio else None
    if store_audio:
        os.makedirs(audio_dir, exist_ok=True)

    # Process each item
    for idx, items in enumerate(tqdm(preprocess_loader, desc="Preprocessing")):
        item = items[0]  # Get the single item from batch

        try:
            # Convert audio to tensor
            audio = torch.as_tensor(item["audio"], dtype=torch.float)

            # Pad or trim to exact length
            if audio.size(0) < config["target_sr"] * 15:
                audio = F.pad(audio, (0, config["target_sr"] * 15 - audio.size(0)))
            else:
                audio = audio[: config["target_sr"] * 15]

            # Pre-compute mel spectrogram
            with torch.no_grad():
                # First get raw mel spectrogram [freq_bins, time_frames]
                mel_spec_data = mel_spec(audio)
                mel_spec_data = torch.log(mel_spec_data + 1e-4)
                # Reshape to [channels=1, freq_bins, time_frames]
                mel_spec_data = mel_spec_data.unsqueeze(0)

                # Debug printout occasionally
                if idx % 1000 == 0:
                    print(f"Mel spectrogram shape: {mel_spec_data.shape}")

            # Store data
            output_data = {
                "mel_spec": mel_spec_data.half(),  # Half precision for storage efficiency
                "label": item["label"],
                "era": item["era"],
                "file": item["file"],
            }

            # Only store audio if explicitly requested
            if store_audio:
                audio_path = f"{audio_dir}/{idx:06d}.pt"
                torch.save(audio, audio_path)
                output_data["audio_file"] = audio_path

            # Save preprocessed data
            torch.save(output_data, f"{preprocessed_dir}/{idx:06d}.pt")

        except Exception as e:
            print(f"Error preprocessing item {idx}: {e}")
            continue

    print(
        f"Preprocessing completed. {len(glob.glob(f'{preprocessed_dir}/*.pt'))} items saved."
    )
    return preprocessed_dir
