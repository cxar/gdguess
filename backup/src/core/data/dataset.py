#!/usr/bin/env python3
"""
Simplified dataset implementation for the Grateful Dead show dating model.
"""

import datetime
import glob
import os
import re
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

def get_device():
    """
    Determine the optimal available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def collate_fn(batch, device=None):
    """
    Simple collate function that handles audio data batching.
    
    Args:
        batch: List of data items to collate
        device: Target device for tensors (default: None, will not move tensors)

    Returns:
        Collated batch dictionary
    """
    # Filter out error items
    valid_batch = [item for item in batch if not item.get('error', False)]
    
    if not valid_batch:
        return {'empty_batch': True}
    
    # Initialize output dictionary
    output = {}
    
    # Process each key in the batch
    for key in valid_batch[0].keys():
        if key == 'file' or key == 'path':
            # Keep strings as a list
            output[key] = [item[key] for item in valid_batch]
        else:
            # Handle tensor data
            tensors = []
            for item in valid_batch:
                if key in item and isinstance(item[key], torch.Tensor):
                    tensors.append(item[key].float())
            
            if tensors:
                # Stack tensors if possible
                try:
                    output[key] = torch.stack(tensors)
                except:
                    # Skip keys that can't be stacked
                    pass
    
    # Move tensors to device if specified
    if device is not None:
        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].to(device)
    
    return output

# Define a picklable collate function
class SimpleCollate:
    """A picklable collate function class for DataLoader multiprocessing."""
    
    def __init__(self, device=None):
        """Initialize with a device.
        
        Args:
            device: The device to use for tensors
        """
        self.device = device
    
    def __call__(self, batch):
        """Call the collate function.
        
        Args:
            batch: Batch of samples
            
        Returns:
            Collated batch
        """
        return collate_fn(batch, self.device)

class DeadShowDataset(Dataset):
    """Dataset class for Grateful Dead shows."""
    
    def __init__(
        self,
        root_dir: str,
        target_sr: int = 24000,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing audio files
            target_sr: Target sample rate
            device: Device to place tensors on (default: None, will use CPU)
        """
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.device = device if device is not None else torch.device('cpu')
        
        # Find all audio files recursively
        self.audio_files = []
        audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"]
        
        for ext in audio_extensions:
            self.audio_files.extend(sorted(glob.glob(os.path.join(root_dir, "**", ext), recursive=True)))
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {root_dir}. Make sure the directory contains supported audio files.")
            
        print(f"Found {len(self.audio_files)} audio files")
            
    def __len__(self) -> int:
        return len(self.audio_files)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing audio data and metadata
        """
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
            
        # Create result dictionary
        result = {
            "audio": audio.squeeze(0),  # Remove channel dimension
            "path": audio_path,
        }
        
        # Move tensors to the specified device
        if self.device is not None and self.device.type != 'cpu':
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device)
                    
        return result

class PreprocessedDeadShowDataset(Dataset):
    """Dataset class for preprocessed Grateful Dead shows."""
    
    def __init__(
        self,
        preprocessed_dir: str,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            preprocessed_dir: Directory containing preprocessed data
            device: Device to place tensors on (default: None, will use CPU)
        """
        self.preprocessed_dir = preprocessed_dir
        self.device = device if device is not None else torch.device('cpu')
        
        # Find preprocessed files
        print(f"Looking for preprocessed files in: {preprocessed_dir}")
        pt_files = glob.glob(os.path.join(preprocessed_dir, "**/*.pt"), recursive=True)
        
        # Sort and store the files
        self.files = sorted(pt_files)
        print(f"Found {len(self.files)} preprocessed files")
        
        if not self.files:
            raise ValueError(f"No preprocessed files (.pt) found in {preprocessed_dir}")
        
    def __len__(self) -> int:
        return len(self.files)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing audio features and metadata
        """
        file_path = self.files[idx]
        
        try:
            # Load on CPU first, then move to device
            data = torch.load(file_path, map_location='cpu')
            
            # Process the data
            processed_data = self._process_data(data, file_path)
            return processed_data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {'error': True, 'path': file_path}
    
    def _process_data(self, data: Dict, file_path: str) -> Dict:
        """
        Process loaded data to ensure it meets requirements.
        
        Args:
            data: Loaded data dictionary
            file_path: Path to the file
            
        Returns:
            Processed data dictionary
        """
        # Add the file path if not present
        if 'path' not in data:
            data['path'] = file_path
            
        # Extract date information from filename if needed
        if 'label' not in data:
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file_path)
            if date_match:
                year, month, day = map(int, date_match.groups())
                date_obj = datetime.date(year, month, day)
                base_date = datetime.date(1968, 1, 1)
                days = (date_obj - base_date).days
                data['label'] = torch.tensor(days, dtype=torch.float)
            else:
                data['label'] = torch.tensor(0.0, dtype=torch.float)
        
        # Extract year if not present
        if 'year' not in data:
            filename = os.path.basename(file_path)
            year_match = re.search(r'(\d{4})-\d{2}-\d{2}', filename)
            if year_match:
                year = int(year_match.group(1))
                data['year'] = torch.tensor(year, dtype=torch.long)
        
        # Ensure spectrograms are present
        if 'mel_spec' not in data and 'harmonic' not in data:
            print("WARNING: No spectrograms found, creating dummy tensor")
            data['mel_spec'] = torch.zeros((1, 128, 100), dtype=torch.float)
            
        # Ensure we have both harmonic and percussive spectrograms
        if 'harmonic' not in data and 'mel_spec' in data:
            data['harmonic'] = data['mel_spec']
            
        if 'percussive' not in data and 'mel_spec_percussive' in data:
            data['percussive'] = data['mel_spec_percussive']
        elif 'percussive' not in data and 'harmonic' in data:
            # Just copy harmonic as percussive if not available
            print("WARNING: No percussive spectrogram found, using harmonic")
            data['percussive'] = data['harmonic'].clone()
            
        # Ensure we have chroma and spectral contrast
        if 'chroma' not in data:
            print("WARNING: No chroma found, creating dummy tensor")
            # Create a default shape based on harmonic time dimension
            if 'harmonic' in data:
                time_dim = data['harmonic'].shape[-1]
                data['chroma'] = torch.zeros((12, time_dim), dtype=torch.float)
            else:
                data['chroma'] = torch.zeros((12, 100), dtype=torch.float)
                
        if 'spectral_contrast' not in data and 'spectral_contrast_harmonic' in data:
            data['spectral_contrast'] = data['spectral_contrast_harmonic']
        elif 'spectral_contrast' not in data:
            print("WARNING: No spectral contrast found, creating dummy tensor")
            # Create a default shape based on harmonic time dimension
            if 'harmonic' in data:
                time_dim = data['harmonic'].shape[-1]
                data['spectral_contrast'] = torch.zeros((6, time_dim), dtype=torch.float)
            else:
                data['spectral_contrast'] = torch.zeros((6, 100), dtype=torch.float)
        
        # Move tensors to the target device if needed
        if self.device is not None and self.device.type != 'cpu':
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(device=self.device)
        
        return data