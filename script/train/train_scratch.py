#!/usr/bin/env python3
"""
Grateful Dead show dating training script with fixes for numerical stability, using fixed precision.
Optimized for efficient GPU utilization on H100 with cross-platform compatibility.
"""

import datetime
import glob
import io
import os
import platform
import time
from typing import Dict, List, Optional, Tuple, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Ensure torch is imported at the global scope
import torch.backends.cudnn
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


# ====================== UTILITY FUNCTIONS ======================
def reset_parameters(model: nn.Module) -> None:
    """More aggressive parameter initialization to prevent nans"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=0.5)
                elif "bias" in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.MultiheadAttention):
            for name, param in m.named_parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.5)


def debug_shape(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Debug helper to print tensor shapes"""
    print(f"Shape of {name}: {tensor.shape}")
    return tensor


def combined_loss(
    outputs: Dict[str, torch.Tensor],
    targets_label: torch.Tensor,
    targets_era: torch.Tensor,
) -> torch.Tensor:
    """Calculate the combined loss for date prediction and era classification"""
    try:
        pred_days = outputs["days"]

        # Ensure everything is on the same device
        target_device = targets_label.device
        if pred_days.device != target_device:
            pred_days = pred_days.to(target_device)

        if outputs["era_logits"].device != target_device:
            outputs["era_logits"] = outputs["era_logits"].to(target_device)

        # Handle NaN values in predictions
        if torch.isnan(pred_days).any():
            pred_days = torch.where(
                torch.isnan(pred_days), torch.zeros_like(pred_days), pred_days
            )

        days_loss = F.mse_loss(pred_days, targets_label)
        era_loss = F.cross_entropy(outputs["era_logits"], targets_era)
        total_loss = days_loss + 0.3 * era_loss

        if torch.isnan(total_loss):
            print("Warning: NaN detected in loss calculation!")
            return torch.tensor(0.1, device=target_device, requires_grad=True)

        return total_loss
    except Exception as e:
        print(f"Error in loss calculation: {e}")
        dummy_loss = torch.tensor(0.1, device=targets_label.device, requires_grad=True)
        return dummy_loss


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


# ====================== DATA AUGMENTATION ======================
class DeadShowAugmenter:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply Grateful Dead specific augmentations"""
        augmentations = []

        # Audience simulation
        if np.random.random() > 0.7:

            def audience_sim(x):
                impulse = np.exp(-np.linspace(0, 10, int(0.3 * self.sample_rate)))
                reverb = np.convolve(x, impulse, mode="full")[: len(x)]
                noise_level = np.random.uniform(0.01, 0.05)
                noise = np.random.randn(len(x)) * noise_level
                mix_ratio = np.random.uniform(0.5, 0.8)
                return x * mix_ratio + reverb * (1 - mix_ratio) + noise

            augmentations.append(audience_sim)

        # Tape wear simulation
        if np.random.random() > 0.6:

            def tape_wear(x):
                window_size = int(self.sample_rate * 0.002)
                window = np.ones(window_size) / window_size
                filtered_x = np.convolve(x, window, mode="same")
                return filtered_x

            augmentations.append(tape_wear)

        # Era-specific EQ
        if np.random.random() > 0.5:
            era = np.random.choice(["early", "seventies", "eighties", "nineties"])

            def era_eq(x):
                if era == "early":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.logical_and(freq >= 50, freq <= 7000)
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    x_filtered = np.tanh(x_filtered * 1.2)
                    return x_filtered
                elif era == "seventies":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.ones_like(X, dtype=float)
                    midrange_mask = np.logical_and(freq >= 300, freq <= 2500)
                    mask[midrange_mask] = 1.3
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    x_filtered = (
                        np.sign(x_filtered)
                        * np.log(1 + 5 * np.abs(x_filtered))
                        / np.log(6)
                    )
                    return x_filtered
                elif era == "eighties":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.ones_like(X, dtype=float)
                    high_mask = freq >= 5000
                    mask[high_mask] = 1.2
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    bits = np.random.randint(10, 16)
                    x_filtered = np.round(x_filtered * (2**bits)) / (2**bits)
                    return x_filtered
                elif era == "nineties":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.ones_like(X, dtype=float)
                    mask[freq < 100] = 1.1
                    mask[freq > 8000] = 1.1
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    return x_filtered
                return x

            augmentations.append(era_eq)

        # Apply all selected augmentations
        augmented_audio = audio.copy()
        for aug_func in augmentations:
            augmented_audio = aug_func(augmented_audio)

        # Normalize the output
        if np.max(np.abs(augmented_audio)) > 0:
            augmented_audio = augmented_audio / np.max(np.abs(augmented_audio))

        return augmented_audio


# ====================== DATASET IMPLEMENTATIONS ======================
def identity_collate(x):
    """Simple collate function that just returns the input unchanged (for Mac compatibility)"""
    return x


def preprocess_dataset(
    config: Dict, force_preprocess: bool = False, store_audio: bool = False
) -> str:
    """Preprocess the entire dataset once to avoid repeated CPU work"""
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


class DeadShowDataset(Dataset):
    """Original dataset implementation that loads audio files directly"""

    def __init__(
        self,
        root_dir: str,
        base_date: datetime.date,
        target_sr: int = 24000,
        augment: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.base_date = base_date
        self.files = []
        self.labels = []
        self.augmenter = DeadShowAugmenter(target_sr) if augment else None
        self.verbose = verbose

        self.load_data()

        if verbose:
            print(f"Dataset loaded: {len(self.files)} files")

    def load_data(self) -> None:
        """Load data files and parse dates/labels"""
        file_count = 0

        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    file_count += 1

                    # Skip files with placeholder dates
                    if file.startswith(f"{subdir}-00-00"):
                        continue

                    file_path = os.path.join(subdir_path, file)

                    # Parse date from filename
                    date_str = file[:10]
                    try:
                        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    except Exception as e:
                        print(f"Skipping {file} due to date parse error: {e}")
                        continue

                    days = (date_obj.date() - self.base_date).days

                    # Determine era based on year
                    year = date_obj.year
                    era = 0
                    if 1972 <= year <= 1974:
                        era = 1
                    elif 1975 <= year <= 1979:
                        era = 2
                    elif 1980 <= year <= 1990:
                        era = 3
                    elif year > 1990:
                        era = 4

                    self.files.append(file_path)
                    self.labels.append(
                        {
                            "days": days,
                            "era": era,
                            "year": year,
                            "date": date_obj.date(),
                        }
                    )

        if self.verbose:
            print(f"Dataset loaded: {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor, str]]:
        """Get an audio item and its associated metadata"""
        file_path = self.files[idx]
        label_info = self.labels[idx]

        try:
            # Load and process audio
            y, _ = librosa.load(file_path, sr=self.target_sr, mono=True)
            y = y[: self.target_sr * 15]

            # Apply augmentation if configured
            if self.augmenter is not None and np.random.random() > 0.5:
                y = self.augmenter.augment(y)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            y = np.zeros(int(self.target_sr * 15))

        return {
            "audio": y,
            "label": torch.as_tensor(label_info["days"], dtype=torch.float),
            "era": torch.as_tensor(label_info["era"], dtype=torch.long),
            "file": file_path,
        }


class PreprocessedDeadShowDataset(Dataset):
    """Dataset that works with preprocessed files for faster training"""

    def __init__(
        self, preprocessed_dir: str, augment: bool = False, target_sr: int = 24000
    ):
        self.preprocessed_dir = preprocessed_dir
        self.files = sorted(glob.glob(f"{preprocessed_dir}/*.pt"))
        self.augment = augment
        self.target_sr = target_sr
        self.augmenter = DeadShowAugmenter(target_sr) if augment else None

        print(f"Using preprocessed dataset with {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """Get a preprocessed item directly from storage"""
        try:
            # Load preprocessed data
            item = torch.load(self.files[idx])

            # Actually use the preprocessed mel spectrogram instead of reloading audio
            if "mel_spec" in item:
                return {
                    "mel_spec": item["mel_spec"].float(),  # Convert half to float
                    "label": item["label"],
                    "era": item["era"],
                    "file": item["file"],
                    "use_precomputed": True,  # Flag that we're using precomputed data
                }

            # Fallback to original audio path if needed
            if (
                self.augment
                and self.augmenter is not None
                and torch.rand(1).item() > 0.5
            ):
                # For augmentation, we need to load the raw audio
                y, _ = librosa.load(item["file"], sr=self.target_sr, mono=True)
                audio_np = y[: self.target_sr * 15]
                augmented_audio = self.augmenter.augment(audio_np)
                audio = torch.as_tensor(augmented_audio, dtype=torch.float)
            else:
                # Load original audio only when needed (for augmentation)
                y, _ = librosa.load(item["file"], sr=self.target_sr, mono=True)
                audio = torch.as_tensor(y[: self.target_sr * 15], dtype=torch.float)

            # Pad if needed
            if audio.size(0) < self.target_sr * 15:
                audio = F.pad(audio, (0, self.target_sr * 15 - audio.size(0)))

            return {
                "audio": audio,
                "label": item["label"],
                "era": item["era"],
                "file": item["file"],
                "use_precomputed": False,
            }
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # Fallback to zeros
            return {
                "audio": torch.zeros(self.target_sr * 15),
                "label": item.get("label", torch.zeros(1)),
                "era": item.get("era", torch.zeros(1, dtype=torch.long)),
                "file": item.get("file", "error"),
                "use_precomputed": False,
            }


def optimized_collate_fn(batch):
    """Efficient collation function for batching data items"""
    # Separate out different item types
    precomputed_items = [item for item in batch if item.get("use_precomputed", False)]
    audio_items = [
        item
        for item in batch
        if "audio" in item and not item.get("use_precomputed", False)
    ]

    # Prepare the final batched data
    result = {
        "label": torch.stack([item["label"] for item in batch]),
        "era": torch.stack([item["era"] for item in batch]),
        "file": [item["file"] for item in batch],
    }

    # Handle precomputed spectrograms if we have them
    if precomputed_items:
        try:
            result["mel_spec"] = torch.stack(
                [item["mel_spec"] for item in precomputed_items]
            )
        except Exception as e:
            print(f"Error stacking mel_spec: {e}")
            if precomputed_items:
                # Use the first item as fallback
                result["mel_spec"] = precomputed_items[0]["mel_spec"].unsqueeze(0)

    # Handle raw audio items
    if audio_items:
        try:
            result["audio"] = torch.stack([item["audio"] for item in audio_items])
        except Exception as e:
            print(f"Error stacking audio: {e}")
            if audio_items:
                result["audio"] = audio_items[0]["audio"].unsqueeze(0)

    return result


def collate_fn(batch):
    """Original simple collate function for backward compatibility"""
    desired_length = 360000  # 15s * 24000 Hz
    padded_audios = []
    labels = []
    eras = []
    files = []

    for item in batch:
        audio = torch.as_tensor(item["audio"], dtype=torch.float)
        if audio.size(0) < desired_length:
            audio = torch.nn.functional.pad(audio, (0, desired_length - audio.size(0)))
        else:
            audio = audio[:desired_length]
        padded_audios.append(audio)
        labels.append(item["label"])
        eras.append(item["era"])
        files.append(item["file"])

    return {
        "audio": torch.stack(padded_audios),
        "label": torch.stack(labels),
        "era": torch.stack(eras),
        "file": files,
    }


# ====================== VISUALIZATION AND LOGGING ======================
def log_prediction_samples(
    writer: SummaryWriter,
    true_days: List[float],
    pred_days: List[float],
    base_date: datetime.date,
    epoch: int,
    prefix: str = "train",
) -> None:
    """Log sample predictions to TensorBoard"""
    try:
        samples = min(10, len(true_days))
        indices = np.random.choice(len(true_days), samples, replace=False)
        headers = ["True Date", "Predicted Date", "Error (days)"]
        data = []

        for i in indices:
            try:
                true_date = base_date + datetime.timedelta(days=int(true_days[i]))
                pred_date = base_date + datetime.timedelta(days=int(pred_days[i]))
                error = abs(int(true_days[i]) - int(pred_days[i]))
                data.append(
                    [
                        true_date.strftime("%Y-%m-%d"),
                        pred_date.strftime("%Y-%m-%d"),
                        str(error),
                    ]
                )
            except (ValueError, OverflowError, TypeError) as e:
                print(f"Error processing prediction sample {i}: {e}")
                continue

        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in data:
            table += "| " + " | ".join(row) + " |\n"

        writer.add_text(f"{prefix}/date_predictions", table, epoch)
    except Exception as e:
        print(f"Error in log_prediction_samples: {e}")


def log_era_confusion_matrix(
    writer: SummaryWriter,
    true_eras: List[int],
    pred_eras: List[int],
    epoch: int,
    num_classes: int = 5,
) -> None:
    """Generate and log a confusion matrix for era classification"""
    try:
        cm = confusion_matrix(true_eras, pred_eras, labels=range(num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Era")
        plt.ylabel("True Era")
        plt.title("Era Classification Confusion Matrix")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1))
        writer.add_image("validation/era_confusion_matrix", img_tensor, epoch)
        plt.close()
    except Exception as e:
        print(f"Error in log_era_confusion_matrix: {e}")


def log_error_by_era(
    writer: SummaryWriter,
    true_days: List[float],
    pred_days: List[float],
    true_eras: List[int],
    epoch: int,
) -> None:
    """Log error metrics broken down by era"""
    try:
        era_errors = {era: [] for era in range(5)}

        for days_true, days_pred, era in zip(true_days, pred_days, true_eras):
            try:
                era = int(era)
                error = abs(float(days_true) - float(days_pred))
                if not np.isnan(error) and not np.isinf(error):
                    era_errors[era].append(error)
            except (ValueError, TypeError):
                continue

        era_mean_errors = {}
        era_names = {
            0: "Primal (65-71)",
            1: "Europe 72-74",
            2: "Hiatus Return (75-79)",
            3: "Brent Era (80-90)",
            4: "Bruce/Vince (90-95)",
        }

        for era, errors in era_errors.items():
            if errors:
                era_mean_errors[era] = sum(errors) / len(errors)
            else:
                era_mean_errors[era] = 0

        for era, mean_error in era_mean_errors.items():
            writer.add_scalar(f"metrics/era_{era}_mae", mean_error, epoch)

        plt.figure(figsize=(10, 6))
        eras_keys = list(era_mean_errors.keys())
        errors = list(era_mean_errors.values())
        x_pos = range(len(eras_keys))

        plt.bar(x_pos, errors)
        plt.xticks(x_pos, [era_names[era] for era in eras_keys])
        plt.xlabel("Era")
        plt.ylabel("Mean Absolute Error (days)")
        plt.title("Dating Error by Era")
        plt.xticks(rotation=45)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1))
        writer.add_image("validation/error_by_era", img_tensor, epoch)
        plt.close()
    except Exception as e:
        print(f"Error in log_error_by_era: {e}")


def find_learning_rate(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: callable,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_steps: int = 100,
) -> float:
    """Find optimal learning rate using learning rate range test"""
    model.train()
    lrs = []
    losses = []
    lr = start_lr

    mult = (end_lr / start_lr) ** (1 / num_steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    for i, batch in enumerate(train_loader):
        if i >= num_steps:
            break

        audio = batch["audio"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        era = batch["era"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(audio)
        loss = criterion(outputs, label, era)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in loss - skipping batch")
            continue

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

        lr *= mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.savefig("lr_finder.png")

    # Find optimal learning rate (point with steepest negative gradient)
    smoothed_losses = np.array(losses)
    gradients = np.gradient(smoothed_losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]

    print(f"Suggested learning rate: {optimal_lr:.1e}")
    return optimal_lr


# ====================== TRAINING FUNCTION ======================
def train_model(config: Dict) -> None:
    """Main training function"""
    import torch  # Ensure torch is available in this scope

    # Preprocess dataset
    preprocessed_dir = preprocess_dataset(
        config, force_preprocess=False, store_audio=False
    )

    # Set up logging
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Log configuration
    config_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
    config_df.to_csv(os.path.join(log_dir, "config.csv"), index=False)
    writer.add_text("Configuration", config_df.to_markdown(), 0)

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        print(f"Using CUDA. Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            writer.add_text("System", f"GPU {i}: {torch.cuda.get_device_name(i)}", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set up dataset
    full_dataset = PreprocessedDeadShowDataset(
        preprocessed_dir=preprocessed_dir,
        augment=config["use_augmentation"],
        target_sr=config["target_sr"],
    )

    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(config["valid_split"] * dataset_size)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(
        f"Total samples: {dataset_size}, Training: {train_size}, Validation: {val_size}"
    )

    # Adjust DataLoader settings based on platform
    is_mac = platform.system() == "Darwin"
    if is_mac:
        print("Running on macOS, adjusting DataLoader settings for compatibility")
        num_workers = 0  # No multiprocessing on Mac
        persistent_workers = False
        dataloader_kwargs = {
            "batch_size": config["batch_size"],
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": True,
            "collate_fn": optimized_collate_fn,
        }
        # Note: No prefetch_factor or persistent_workers when num_workers=0
    else:
        num_workers = min(4, os.cpu_count() - 2)
        persistent_workers = False
        prefetch_factor = 2
        dataloader_kwargs = {
            "batch_size": config["batch_size"],
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": True,
            "collate_fn": optimized_collate_fn,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs["shuffle"] = False
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)

    # Initialize model
    model = DeadShowDatingModel(sample_rate=config["target_sr"])
    reset_parameters(model)

    # Apply JIT compilation if configured
    if torch.__version__ >= "1.10.0" and device.type == "cuda" and config["use_jit"]:
        try:
            model = torch.jit.script(model)
            print("Successfully applied JIT compilation to model")
        except Exception as e:
            print(f"JIT compilation failed, using regular model: {e}")

    # Set up multi-GPU if available
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
    writer.add_text(
        "Model",
        f"Total parameters: {total_params:,} (Trainable: {trainable_params:,})",
        0,
    )

    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["initial_lr"],
        weight_decay=config["weight_decay"],
        eps=1e-8,
    )

    # Run learning rate finder if configured
    if config["run_lr_finder"]:
        print("Running learning rate finder...")
        optimal_lr = find_learning_rate(
            model,
            train_loader,
            optimizer,
            combined_loss,
            device,
            start_lr=1e-6,
            end_lr=1e-1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = optimal_lr
        config["initial_lr"] = optimal_lr
        print(f"Updated learning rate to {optimal_lr}")

    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["initial_lr"],
        total_steps=config["total_training_steps"],
        pct_start=0.05,
        div_factor=25,
        final_div_factor=1000,
    )

    # Initialize training state
    global_step = 0
    start_epoch = 0
    best_val_mae = float("inf")
    patience_counter = 0

    # Resume from checkpoint if configured
    if config["resume_checkpoint"] and os.path.exists(config["resume_checkpoint"]):
        print(f"Loading checkpoint: {config['resume_checkpoint']}")
        # Add datetime.date to allowed globals for PyTorch 2.6+ compatibility
        try:
            # First try with PyTorch 2.6+ approach using safe globals
            import torch.serialization

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([datetime.date])
                checkpoint = torch.load(
                    config["resume_checkpoint"], map_location=device
                )
            else:
                # Fall back to older PyTorch versions
                checkpoint = torch.load(
                    config["resume_checkpoint"], map_location=device, weights_only=False
                )
        except (TypeError, AttributeError):
            # Final fallback for even older PyTorch versions
            checkpoint = torch.load(config["resume_checkpoint"], map_location=device)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        best_val_mae = checkpoint.get("best_val_mae", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)

        print(f"Resumed from step {global_step}, epoch {start_epoch}")

    # Training loop
    epoch = start_epoch
    ema_loss = None
    ema_alpha = 0.98
    training_start_time = time.time()

    print(f"Starting training for {config['total_training_steps']} steps...")

    while global_step < config["total_training_steps"]:
        print(f"Starting epoch {epoch+1}")
        epoch_loss = 0.0
        batch_count = 0
        epoch_mae = 0.0
        epoch_era_correct = 0
        epoch_samples = 0

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
        epoch_start_time = time.time()

        for batch in pbar:
            try:
                # Move batch data to device
                label = batch["label"].to(device, non_blocking=True)
                era = batch["era"].to(device, non_blocking=True)

                # For MPS device, we need to handle audio explicitly
                if "audio" in batch and device.type == "mps":
                    batch["audio"] = batch["audio"].to(device, non_blocking=True)
                elif "raw_audio" in batch and device.type == "mps":
                    batch["raw_audio"] = batch["raw_audio"].to(
                        device, non_blocking=True
                    )

                # Forward pass
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                loss = combined_loss(outputs, label, era)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: NaN/Inf detected in loss - skipping batch")
                    continue

                # Backward pass
                loss.backward()

                # Check for valid gradients
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if (
                            torch.isnan(param.grad).any()
                            or torch.isinf(param.grad).any()
                        ):
                            print(f"NaN/Inf gradient detected in {name}")
                            valid_gradients = False
                            break

                if valid_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    print("Skipping optimizer step due to invalid gradients")

                scheduler.step()

                # Update metrics
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss

                epoch_loss += current_loss
                batch_count += 1
                global_step += 1

                # Calculate batch metrics
                with torch.no_grad():
                    pred_days = outputs["days"].detach()
                    true_days = label.detach()
                    abs_error = torch.abs(pred_days - true_days)
                    batch_mae = torch.mean(abs_error).item()
                    epoch_mae += batch_mae * label.size(0)

                    _, pred_era = torch.max(outputs["era_logits"].detach(), 1)
                    batch_correct = (pred_era == era).sum().item()
                    epoch_era_correct += batch_correct
                    epoch_samples += label.size(0)

                # Log metrics
                writer.add_scalar("loss/train", current_loss, global_step)
                writer.add_scalar("loss/train_ema", ema_loss, global_step)
                writer.add_scalar(
                    "learning_rate", optimizer.param_groups[0]["lr"], global_step
                )

                # Log GPU usage
                if device.type == "cuda" and global_step % 100 == 0:
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
                    writer.add_scalar(
                        "system/gpu_memory_allocated_mb", memory_allocated, global_step
                    )
                    writer.add_scalar(
                        "system/gpu_memory_reserved_mb", memory_reserved, global_step
                    )

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "ema": f"{ema_loss:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                    }
                )

                # Periodic metric logging
                if global_step % 50 == 0:
                    writer.add_scalar("metrics/train_mae_days", batch_mae, global_step)
                    writer.add_scalar(
                        "metrics/train_era_accuracy",
                        100 * batch_correct / label.size(0),
                        global_step,
                    )

                # Periodic prediction samples
                if global_step % 500 == 0:
                    pred_days_cpu = pred_days.cpu().numpy()
                    true_days_cpu = true_days.cpu().numpy()
                    log_prediction_samples(
                        writer,
                        true_days_cpu,
                        pred_days_cpu,
                        config["base_date"],
                        global_step,
                        "train",
                    )

                # Check if we've reached total training steps
                if global_step >= config["total_training_steps"]:
                    break

            except Exception as e:
                print(f"Error processing batch: {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        avg_epoch_mae = epoch_mae / epoch_samples if epoch_samples > 0 else 0
        epoch_era_acc = (
            100 * epoch_era_correct / epoch_samples if epoch_samples > 0 else 0
        )

        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
            f"avg loss: {avg_epoch_loss:.4f}, "
            f"MAE: {avg_epoch_mae:.2f} days, "
            f"era acc: {epoch_era_acc:.2f}%"
        )

        writer.add_scalar("epoch/train_loss", avg_epoch_loss, epoch + 1)
        writer.add_scalar("epoch/train_mae", avg_epoch_mae, epoch + 1)
        writer.add_scalar("epoch/train_era_accuracy", epoch_era_acc, epoch + 1)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_era_correct = 0
        val_count = 0
        all_true_days = []
        all_pred_days = []
        all_true_eras = []
        all_pred_eras = []

        print("Running validation...")
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                try:
                    val_label = val_batch["label"].to(device, non_blocking=True)
                    val_era = val_batch["era"].to(device, non_blocking=True)

                    val_outputs = model(val_batch)
                    val_batch_loss = combined_loss(val_outputs, val_label, val_era)

                    pred_days = val_outputs["days"]
                    abs_error = torch.abs(pred_days - val_label)
                    batch_mae = torch.mean(abs_error).item()

                    _, pred_era = torch.max(val_outputs["era_logits"], 1)
                    era_correct = (pred_era == val_era).sum().item()

                    val_loss += val_batch_loss.item() * val_label.size(0)
                    val_mae += batch_mae * val_label.size(0)
                    val_era_correct += era_correct
                    val_count += val_label.size(0)

                    all_true_days.extend(val_label.cpu().numpy())
                    all_pred_days.extend(pred_days.cpu().numpy())
                    all_true_eras.extend(val_era.cpu().numpy())
                    all_pred_eras.extend(pred_era.cpu().numpy())
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        # Calculate validation metrics
        avg_val_loss = val_loss / val_count if val_count > 0 else 0
        avg_val_mae = val_mae / val_count if val_count > 0 else 0
        val_era_acc = 100 * val_era_correct / val_count if val_count > 0 else 0

        # Log validation metrics
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)
        writer.add_scalar("epoch/val_mae_days", avg_val_mae, epoch + 1)
        writer.add_scalar("epoch/val_era_accuracy", val_era_acc, epoch + 1)

        # Generate visualization reports
        log_prediction_samples(
            writer, all_true_days, all_pred_days, config["base_date"], epoch + 1, "val"
        )
        log_era_confusion_matrix(writer, all_true_eras, all_pred_eras, epoch + 1)
        log_error_by_era(writer, all_true_days, all_pred_days, all_true_eras, epoch + 1)

        print(
            f"Validation - loss: {avg_val_loss:.4f}, "
            f"MAE: {avg_val_mae:.2f} days, "
            f"era acc: {val_era_acc:.2f}%"
        )

        # Early stopping check
        if epoch > 0:
            if best_val_mae - avg_val_mae > config["min_delta"]:
                print(
                    f"Validation MAE improved from {best_val_mae:.2f} to {avg_val_mae:.2f}"
                )
                best_val_mae = avg_val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation MAE did not improve. Patience: {patience_counter}/{config['patience']}"
                )

            if patience_counter >= config["patience"] and config["use_early_stopping"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            best_val_mae = avg_val_mae

        # Save checkpoints
        try:
            checkpoint_data = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
                "val_mae": avg_val_mae,
                "val_era_acc": val_era_acc,
                "patience_counter": patience_counter,
                "config": config,
            }

            # Save latest checkpoint
            torch.save(
                checkpoint_data,
                os.path.join("checkpoints", config["latest_checkpoint"]),
            )

            # Save best checkpoint if improved
            if avg_val_mae < best_val_mae or epoch == 0:
                best_val_mae = avg_val_mae
                torch.save(
                    checkpoint_data,
                    os.path.join("checkpoints", config["best_checkpoint"]),
                )
                print(f"New best model saved! Val MAE: {avg_val_mae:.2f}")

            # Save periodic checkpoints
            if (epoch + 1) % config["save_every_n_epochs"] == 0:
                torch.save(
                    checkpoint_data,
                    os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pt"),
                )
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

        epoch += 1

    # Calculate and log training summary
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation MAE: {best_val_mae:.2f} days")

    writer.add_text(
        "Training Summary",
        f"Total steps: {global_step}\n"
        f"Total epochs: {epoch}\n"
        f"Best validation MAE: {best_val_mae:.2f} days\n"
        f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s",
        0,
    )

    writer.close()
    print("Training complete!")


# ====================== INFERENCE FUNCTION ======================
def predict_date(
    model: nn.Module,
    audio_path: str,
    base_date: datetime.date = datetime.date(1968, 1, 1),
    target_sr: int = 24000,
    device: Union[str, torch.device] = "cuda",
) -> Dict:
    """Run inference on a single audio file to predict its date"""
    import torch  # Ensure torch is available in this scope

    """Run inference on a single audio file to predict its date"""
    model.eval()

    try:
        # Load and preprocess audio
        y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        y = y[: target_sr * 15]

        audio = torch.tensor(y, dtype=torch.float).unsqueeze(0)

        if audio.size(1) < target_sr * 15:
            audio = torch.nn.functional.pad(audio, (0, target_sr * 15 - audio.size(1)))

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
                era_names[i]: prob for i, prob in enumerate(era_probs)
            },
        }

    except Exception as e:
        return {"error": str(e)}


# ====================== MAIN FUNCTION ======================
def main():
    """Main entry point for training"""
    # Ensure all necessary modules are imported
    import datetime

    import torch
    import torch.backends.cudnn
    import torch.backends.mps

    base_date = datetime.date(1968, 1, 1)
    config = {
        "input_dir": "../../data/audsnippets-all",
        "target_sr": 24000,
        "base_date": base_date,
        "batch_size": 128,
        "initial_lr": 5e-5,
        "weight_decay": 0.01,
        "num_workers": 8,
        "valid_split": 0.1,
        "use_augmentation": True,
        "resume_checkpoint": "./checkpoints/checkpoint_latest.pt",
        "latest_checkpoint": "checkpoint_latest.pt",
        "best_checkpoint": "checkpoint_best.pt",
        "total_training_steps": 180000,
        "run_lr_finder": False,
        "use_early_stopping": True,
        "patience": 10,
        "min_delta": 0.5,
        "use_jit": False,
        "save_every_n_epochs": 1,
    }

    train_model(config)


if __name__ == "__main__":
    main()
