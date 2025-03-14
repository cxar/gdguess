#!/usr/bin/env python3
"""
Dataset implementations for the Grateful Dead show dating model.
"""

import datetime
import glob
import os
from typing import Dict, Union

import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.augmentation import DeadShowAugmenter


def identity_collate(x):
    """
    Simple collate function that just returns the input unchanged (for Mac compatibility).
    
    Args:
        x: Batch of data
        
    Returns:
        The input unchanged
    """
    return x


def optimized_collate_fn(batch):
    """
    Efficient collation function for batching data items.
    
    Args:
        batch: List of data items to collate
        
    Returns:
        Collated batch dictionary
    """
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
    """
    Original simple collate function for backward compatibility.
    
    Args:
        batch: List of data items to collate
        
    Returns:
        Collated batch dictionary
    """
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


class DeadShowDataset(Dataset):
    """
    Original dataset implementation that loads audio files directly.
    """

    def __init__(
        self,
        root_dir: str,
        base_date: datetime.date,
        target_sr: int = 24000,
        augment: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory containing the audio files
            base_date: Reference date for calculating the number of days
            target_sr: Target sample rate for audio
            augment: Whether to apply audio augmentations
            verbose: Whether to print verbose information
        """
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
        """
        Load data files and parse dates/labels.
        """
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
        """Get the number of items in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get an audio item and its associated metadata.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the audio, label, era, and file path
        """
        file_path = self.files[idx]
        label_info = self.labels[idx]

        try:
            # Load and process audio
            y, _ = librosa.load(file_path, sr=self.target_sr, mono=True)
            y = y[: self.target_sr * 15]

            # Apply augmentation if configured
            if self.augmenter is not None and torch.rand(1).item() > 0.5:
                y = self.augmenter.augment(y)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            y = torch.zeros(int(self.target_sr * 15), dtype=torch.float)

        return {
            "audio": y,
            "label": torch.as_tensor(label_info["days"], dtype=torch.float),
            "era": torch.as_tensor(label_info["era"], dtype=torch.long),
            "file": file_path,
        }


class PreprocessedDeadShowDataset(Dataset):
    """
    Dataset that works with preprocessed files for faster training.
    """

    def __init__(
        self, preprocessed_dir: str, augment: bool = False, target_sr: int = 24000
    ):
        """
        Initialize the preprocessed dataset.
        
        Args:
            preprocessed_dir: Directory containing preprocessed data files
            augment: Whether to apply audio augmentations
            target_sr: Target sample rate for audio
        """
        self.preprocessed_dir = preprocessed_dir
        self.files = sorted(glob.glob(f"{preprocessed_dir}/*.pt"))
        self.augment = augment
        self.target_sr = target_sr
        self.augmenter = DeadShowAugmenter(target_sr) if augment else None

        print(f"Using preprocessed dataset with {len(self.files)} files")

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """
        Get a preprocessed item directly from storage.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the data for the item
        """
        try:
            # Load preprocessed data
            item = torch.load(self.files[idx])

            # Actually use the preprocessed mel spectrogram instead of reloading audio
            if "mel_spec" in item:
                return {
                    "mel_spec": item["mel_spec
