#!/usr/bin/env python3
"""
Dataset implementations for the Grateful Dead show dating model.
"""

import datetime
import glob
import os
import re
from typing import Dict, Union, List, Tuple, Any, Optional

import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import numpy as np

from .augmentation import DeadShowAugmenter


# Check for available devices
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


def identity_collate(x):
    """
    Simple collate function that just returns the input unchanged (for Mac compatibility).

    Args:
        x: Batch of data

    Returns:
        The input unchanged
    """
    return x


def ensure_tensor_consistency(data: Dict[str, Any], target_dtype=torch.float32, device=None) -> Dict[str, Any]:
    """
    Ensure all tensors in a dictionary are of consistent dtype and non-null.
    
    Args:
        data: Dictionary containing tensors or other data
        target_dtype: Target data type (default: torch.float32)
        device: Target device (default: None, will not move tensors to a specific device)
        
    Returns:
        Dictionary with consistent tensor dtypes
    """
    result = {}
    
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            # Convert tensor to target dtype and ensure it's not null
            if value.dtype != target_dtype:
                value = value.to(target_dtype)
            if torch.isnan(value).any() or torch.isinf(value).any():
                # Replace NaN/Inf with zeros
                value = torch.where(torch.isnan(value) | torch.isinf(value), 
                                    torch.zeros_like(value, dtype=target_dtype), 
                                    value)
            # Move to device if specified
            if device is not None:
                value = value.to(device)
            result[key] = value
        elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
            # List of tensors
            if device is not None:
                result[key] = [
                    item.to(device=device, dtype=target_dtype) if item.dtype != target_dtype else item.to(device=device)
                    for item in value
                ]
            else:
                result[key] = [
                    item.to(target_dtype) if item.dtype != target_dtype else item
                    for item in value
                ]
        else:
            # Non-tensor data
            result[key] = value
            
    return result


def optimized_collate_fn(batch: List[Dict[str, torch.Tensor]], device=None) -> Dict[str, torch.Tensor]:
    """
    Optimized collate function for batching data.
    
    Args:
        batch: List of dictionaries containing tensors
        device: Target device for tensors (default: None, will not move tensors)
        
    Returns:
        Dictionary containing batched tensors
    """
    # Filter out error items
    valid_batch = [item for item in batch if not item.get('error', False)]
    
    # If all items were filtered out, return an empty batch signal
    if not valid_batch:
        return {'empty_batch': True}
    
    # Initialize output dictionary
    output = {}
    batch_size = len(valid_batch)
    
    # Debug information
    if len(valid_batch) < len(batch):
        print(f"Filtered out {len(batch) - len(valid_batch)} error items from batch")
    
    # Process each key
    for key in valid_batch[0].keys():
        if key == 'file':
            # Keep strings as a list
            output[key] = [item[key] for item in valid_batch]
        else:
            # Collect valid tensors
            tensors = []
            for item in valid_batch:
                if key in item:
                    if isinstance(item[key], torch.Tensor):
                        tensors.append(item[key].float())
                    else:
                        try:
                            # Try to convert to tensor
                            tensors.append(torch.tensor(item[key], dtype=torch.float32))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {key} to tensor, skipping item")
                            # Skip this item for this key
            
            # Skip if no valid tensors
            if not tensors:
                continue
                
            # Process based on tensor type
            if key == 'mel_spec':
                # Pad mel spectrograms to the maximum time dimension in the batch
                # Check if tensors have at least 3 dimensions before accessing shape[2]
                valid_tensors = [t for t in tensors if isinstance(t, torch.Tensor) and len(t.shape) >= 3]
                if not valid_tensors:
                    # Skip if no valid tensors
                    continue
                    
                max_time = max(t.shape[2] for t in valid_tensors)
                padded = []
                for t in tensors:
                    if isinstance(t, torch.Tensor) and len(t.shape) >= 3:
                        if t.shape[2] < max_time:
                            padding = (0, 0, 0, max_time - t.shape[2])  # Pad only the time dimension
                            t = F.pad(t, padding)
                        padded.append(t.float())  # Ensure float32
                
                if padded:  # Only stack if we have valid tensors
                    if len(padded) != batch_size:
                        print(f"Warning: Only {len(padded)}/{batch_size} valid tensors for 'harmonic'")
                    output['harmonic'] = torch.stack(padded)
            elif key == 'mel_spec_percussive':
                # Similar processing for percussive
                valid_tensors = [t for t in tensors if isinstance(t, torch.Tensor) and len(t.shape) >= 3]
                if not valid_tensors:
                    continue
                    
                max_time = max(t.shape[2] for t in valid_tensors)
                padded = []
                for t in tensors:
                    if isinstance(t, torch.Tensor) and len(t.shape) >= 3:
                        if t.shape[2] < max_time:
                            padding = (0, 0, 0, max_time - t.shape[2])
                            t = F.pad(t, padding)
                        padded.append(t.float())
                
                if padded:
                    if len(padded) != batch_size:
                        print(f"Warning: Only {len(padded)}/{batch_size} valid tensors for 'percussive'")
                    output['percussive'] = torch.stack(padded)
            elif key == 'spectral_contrast_harmonic':
                # Pad spectral contrast to the maximum time dimension and reshape
                valid_tensors = [t for t in tensors if isinstance(t, torch.Tensor) and len(t.shape) >= 2]
                if not valid_tensors:
                    continue
                    
                max_time = max(t.shape[1] for t in valid_tensors)
                padded = []
                for t in tensors:
                    if isinstance(t, torch.Tensor) and len(t.shape) >= 2:
                        # First trim to 6 bands as that's what the model expects
                        t = t[:6]
                        if t.shape[1] < max_time:
                            padding = (0, max_time - t.shape[1])  # Pad the time dimension
                            t = F.pad(t, padding)
                        padded.append(t.float().unsqueeze(0))  # Add batch dimension, ensure float32
                
                if padded:
                    if len(padded) != batch_size:
                        print(f"Warning: Only {len(padded)}/{batch_size} valid tensors for 'spectral_contrast'")
                    output['spectral_contrast'] = torch.cat(padded, dim=0)
            elif key == 'chroma':
                # Pad chroma features to the maximum time dimension
                valid_tensors = [t for t in tensors if isinstance(t, torch.Tensor) and len(t.shape) >= 2]
                if not valid_tensors:
                    continue
                    
                max_time = max(t.shape[1] for t in valid_tensors)
                padded = []
                for t in tensors:
                    if isinstance(t, torch.Tensor) and len(t.shape) >= 2:
                        if t.shape[1] < max_time:
                            padding = (0, max_time - t.shape[1])  # Pad the time dimension
                            t = F.pad(t, padding)
                        padded.append(t.float().unsqueeze(0))  # Add batch dimension, ensure float32
                
                if padded:
                    if len(padded) != batch_size:
                        print(f"Warning: Only {len(padded)}/{batch_size} valid tensors for 'chroma'")
                    output['chroma'] = torch.cat(padded, dim=0)
            elif key in ['label', 'era', 'year', 'date']:
                # These are scalar values - should be present for all batch items
                tensors_float = []
                for i, item in enumerate(valid_batch):
                    if key in item:
                        t = item[key]
                        if isinstance(t, torch.Tensor):
                            tensors_float.append(t.float())
                        else:
                            try:
                                tensors_float.append(torch.tensor(t, dtype=torch.float32))
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid {key} value for batch item {i}")
                
                if tensors_float:
                    if len(tensors_float) != batch_size:
                        print(f"Warning: Only {len(tensors_float)}/{batch_size} valid tensors for '{key}'")
                        # This is critical - we need to ensure all batches have consistent sizes
                        # If we're missing values, it's better to skip this batch
                        return {'empty_batch': True, 'reason': f'Missing {key} values for some batch items'}
                    
                    output[key] = torch.stack(tensors_float)
                else:
                    # Critical fields are missing
                    if key in ['label', 'era', 'year']:
                        print(f"Error: No valid {key} values in batch, skipping batch")
                        return {'empty_batch': True, 'reason': f'No valid {key} values in batch'}
            elif key == 'onset_env':
                # We don't need this for the model
                continue
    
    # Final consistency check
    # Check that all required fields have consistent batch sizes
    required_fields = ['mel_spec', 'mel_spec_percussive', 'label', 'era', 'year']
    actual_batch_size = None
    
    for key in required_fields:
        alt_key = 'harmonic' if key == 'mel_spec' else ('percussive' if key == 'mel_spec_percussive' else key)
        
        # Check for either original or alternative key
        tensor_key = alt_key if alt_key in output else (key if key in output else None)
        
        if tensor_key and tensor_key in output and isinstance(output[tensor_key], torch.Tensor):
            if actual_batch_size is None:
                actual_batch_size = output[tensor_key].shape[0]
            elif output[tensor_key].shape[0] != actual_batch_size:
                print(f"Batch size mismatch: {tensor_key} has {output[tensor_key].shape[0]} items but expected {actual_batch_size}")
                return {'empty_batch': True, 'reason': 'Inconsistent batch sizes'}
    
    # Final consistency check to ensure all tensors are float32 and on the right device
    output = ensure_tensor_consistency(output, target_dtype=torch.float32, device=device)
    
    return output


def collate_fn(batch, device=None):
    """
    Original simple collate function for backward compatibility.

    Args:
        batch: List of data items to collate
        device: Target device for tensors (default: None, will not move tensors)

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

    result = {
        "audio": torch.stack(padded_audios),
        "label": torch.stack(labels),
        "era": torch.stack(eras),
        "file": files,
    }
    
    # Move to device if specified
    if device is not None:
        for key in result:
            if isinstance(result[key], torch.Tensor):
                result[key] = result[key].to(device)
    
    return result


def h200_optimized_collate_fn(batch: List[Dict[str, torch.Tensor]], device=None) -> Dict[str, torch.Tensor]:
    """
    Highly optimized collate function for H200 GPU training.
    
    This version:
    1. Uses non_blocking memory transfers
    2. Minimizes padding operations
    3. Avoids unnecessary deep copies
    4. Uses contiguous memory layouts for tensors
    5. Minimizes CPU-GPU synchronization points
    
    Args:
        batch: List of dictionaries containing tensors
        device: Target device for tensors (default: None)
        
    Returns:
        Dictionary containing batched tensors optimized for H200
    """
    # Filter out error items
    valid_batch = [item for item in batch if not item.get('error', False)]
    
    # If all items were filtered out, return an empty batch signal
    if not valid_batch:
        return {'empty_batch': True}
    
    # Initialize output dictionary
    output = {}
    
    # Ensure tensors are in float32 format
    for key in valid_batch[0].keys():
        if key == 'error':
            continue
            
        # Process tensors
        if all(isinstance(item.get(key), torch.Tensor) for item in valid_batch):
            # Get tensors for this key
            tensors = [item[key].float() for item in valid_batch if key in item]  # Ensure float32
            
            if tensors:
                # Stack or concatenate based on tensor dimensions
                if tensors[0].dim() == 0:
                    # Convert scalar tensors to a batch
                    output[key] = torch.stack(tensors).float()  # Ensure float32
                else:
                    try:
                        # Try to stack tensors of the same shape
                        output[key] = torch.stack(tensors).float()  # Ensure float32
                    except Exception:
                        # If shapes are different, keep as list
                        output[key] = [t.float() for t in tensors]  # Ensure all are float32
        else:
            # Non-tensor data - convert to tensor if possible, else keep as is
            output[key] = []
            for item in valid_batch:
                if key in item:
                    val = item[key]
                    if isinstance(val, (int, float)):
                        # Convert numeric values to float32 tensors
                        output[key].append(torch.tensor(val, dtype=torch.float32))
                    else:
                        output[key].append(val)
    
    # Final consistency check and move to device if specified
    return ensure_tensor_consistency(output, target_dtype=torch.float32, device=device)


class DeadShowDataset(torch.utils.data.Dataset):
    """Dataset class for Grateful Dead shows."""
    
    def __init__(
        self,
        root_dir: str,
        target_sr: int = 24000,
        augment: bool = False,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing audio files or directories with audio files
            target_sr: Target sample rate
            augment: Whether to apply audio augmentation
            verbose: Whether to print verbose information
            device: Device to place tensors on (default: None, will use CPU)
        """
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.augment = augment
        self.verbose = verbose
        self.device = device if device is not None else torch.device('cpu')
        
        # Find all audio files recursively
        self.audio_files = []
        # Common audio file extensions
        audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"]
        
        for ext in audio_extensions:
            self.audio_files.extend(sorted(glob.glob(os.path.join(root_dir, "**", ext), recursive=True)))
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {root_dir} or its subdirectories. Make sure the directory contains supported audio files (.mp3, .wav, .flac, .m4a, .ogg).")
            
        if verbose:
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
            
        # Apply augmentation if enabled
        if self.augment:
            # TODO: Implement audio augmentation
            pass
            
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


class PreprocessedDeadShowDataset(torch.utils.data.Dataset):
    """Dataset class for preprocessed Grateful Dead shows."""
    
    def __init__(
        self,
        preprocessed_dir: str,
        augment: bool = False,
        target_sr: int = 24000,
        device: Optional[torch.device] = None,
        memory_efficient: bool = True,  # Enable memory-efficient mode by default
    ):
        """
        Initialize the dataset.
        
        Args:
            preprocessed_dir: Directory containing preprocessed data
            augment: Whether to apply audio augmentation
            target_sr: Target sample rate
            device: Device to place tensors on (default: None, will use CPU)
            memory_efficient: If True, uses memory-efficient mode that doesn't cache loaded tensors
        """
        self.preprocessed_dir = preprocessed_dir
        self.augment = augment
        self.target_sr = target_sr
        self.device = device if device is not None else torch.device('cpu')
        self.memory_efficient = memory_efficient
        
        # Cache for loaded data (only used if memory_efficient=False)
        self.data_cache = {}
        
        # Detect available computing devices
        self.available_devices = {
            'cuda': torch.cuda.is_available(),
            'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        if self.device.type not in ['cpu', 'cuda', 'mps']:
            print(f"Warning: Unknown device type '{self.device.type}'. Falling back to CPU.")
            self.device = torch.device('cpu')
            
        if self.device.type == 'mps':
            print(f"Using MPS (Apple Silicon GPU) acceleration")
        elif self.device.type == 'cuda':
            print(f"Using CUDA (NVIDIA GPU) acceleration")
        else:
            print(f"Using CPU for computations")
            
        # Use multiple methods to find .pt files
        print(f"Looking for preprocessed files in: {preprocessed_dir}")
        
        # Method 1: Recursive glob pattern
        pt_files_recursive = glob.glob(os.path.join(preprocessed_dir, "**/*.pt"), recursive=True)
        print(f"Found {len(pt_files_recursive)} files with recursive glob")
        
        # Method 2: Direct subdirectory search
        pt_files_direct = []
        if os.path.exists(preprocessed_dir):
            subdirs = [d for d in os.listdir(preprocessed_dir) 
                      if os.path.isdir(os.path.join(preprocessed_dir, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(preprocessed_dir, subdir)
                subdir_files = glob.glob(os.path.join(subdir_path, "*.pt"))
                print(f"Found {len(subdir_files)} files in subdirectory {subdir}")
                pt_files_direct.extend(subdir_files)
        
        # Method 3: Two-level manual search (for year-based organization)
        pt_files_two_level = []
        if os.path.exists(preprocessed_dir):
            subdirs = [d for d in os.listdir(preprocessed_dir) 
                      if os.path.isdir(os.path.join(preprocessed_dir, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(preprocessed_dir, subdir)
                # Check for files in this subdir
                level1_files = glob.glob(os.path.join(subdir_path, "*.pt"))
                pt_files_two_level.extend(level1_files)
                
                # Also check for nested subdirectories
                level2_dirs = [d for d in os.listdir(subdir_path) 
                              if os.path.isdir(os.path.join(subdir_path, d))]
                for level2_dir in level2_dirs:
                    level2_path = os.path.join(subdir_path, level2_dir)
                    level2_files = glob.glob(os.path.join(level2_path, "*.pt"))
                    pt_files_two_level.extend(level2_files)
        
        # Choose the method that found the most files
        method_results = [
            (pt_files_recursive, "recursive glob"),
            (pt_files_direct, "direct subdirectory search"),
            (pt_files_two_level, "two-level search")
        ]
        
        # Sort methods by number of files found (descending)
        method_results.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Use the method that found the most files
        self.files = sorted(method_results[0][0])
        print(f"Using {method_results[0][1]} method which found {len(self.files)} files")
        
        if not self.files:
            # Detailed error message with directory information
            error_msg = f"No preprocessed files (.pt) found in {preprocessed_dir}"
            try:
                if os.path.exists(preprocessed_dir):
                    contents = os.listdir(preprocessed_dir)
                    error_msg += f"\nDirectory exists and contains {len(contents)} items"
                    if contents:
                        error_msg += f"\nSample items: {contents[:5]}"
                        
                    subdirs = [d for d in contents if os.path.isdir(os.path.join(preprocessed_dir, d))]
                    if subdirs:
                        error_msg += f"\nSubdirectories: {subdirs}"
                        # Check a sample subdirectory
                        if subdirs:
                            sample_subdir = os.path.join(preprocessed_dir, subdirs[0])
                            subdir_contents = os.listdir(sample_subdir)
                            error_msg += f"\nContents of {subdirs[0]}: {subdir_contents[:5] if subdir_contents else 'empty'}"
                else:
                    error_msg += " (directory does not exist)"
            except Exception as e:
                error_msg += f"\nError inspecting directory: {e}"
                
            raise ValueError(error_msg)
        
        print(f"Dataset initialized with {len(self.files)} preprocessed files")
        # Print the first few files to help with debugging
        if self.files:
            print("Sample files:")
            for path in self.files[:3]:
                print(f"  {path}")
        
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
        
        # Check if we have the data cached
        if not self.memory_efficient and idx in self.data_cache:
            # Return cached data directly
            return self.data_cache[idx]
        
        # Load PT file - handle MPS compatibility
        try:
            # On MPS devices, we need to be careful with tensor loading
            # Load on CPU first, then move to device
            data = torch.load(file_path, map_location='cpu')
            
            # Process the data
            processed_data = self._process_data(data, file_path)
            
            # Cache if not using memory-efficient mode
            if not self.memory_efficient:
                self.data_cache[idx] = processed_data
            else:
                # In memory-efficient mode, make sure to delete the raw data to free memory
                del data
                
                # Force garbage collection in memory-efficient mode
                if idx % 100 == 0:  # Do this periodically to avoid overhead
                    import gc
                    gc.collect()
                    
                    # Try to clear cache for MPS if applicable
                    try:
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                
            return processed_data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a simple empty dict if loading fails
            # This allows the dataloader to continue instead of crashing
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
            
        # Ensure all required fields are present with proper formats
        
        # 1. Required label field - extract from filename if not present
        if 'label' not in data:
            # Try to extract date from filename (YYYY-MM-DD format)
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file_path)
            if date_match:
                # Calculate days since 1968-01-01 (same baseline as in config)
                year, month, day = map(int, date_match.groups())
                date_obj = datetime.date(year, month, day)
                base_date = datetime.date(1968, 1, 1)
                days = (date_obj - base_date).days
                data['label'] = torch.tensor(days, dtype=torch.float)
            else:
                # Use default if no date found
                data['label'] = torch.tensor(0.0, dtype=torch.float)
        
        # 2. Required era field
        if 'era' not in data:
            # Default to era 0 if not present, or try to derive from year
            year = None
            if 'year' in data:
                year = data['year'].item() if isinstance(data['year'], torch.Tensor) else data['year']
            else:
                # Try to extract year from path
                year_match = re.search(r'/(\d{4})[-/]', file_path)
                if year_match:
                    year = int(year_match.group(1))
            
            # Determine era based on year
            if year is not None:
                if 1972 <= year <= 1974:
                    era = 1
                elif 1975 <= year <= 1979:
                    era = 2
                elif 1980 <= year <= 1990:
                    era = 3
                elif 1990 < year:
                    era = 4
                else:
                    era = 0  # Pre-1972
            else:
                era = 0  # Default
            
            data['era'] = torch.tensor(era, dtype=torch.long)
        
        # 3. Add 'year' field if not present
        if 'year' not in data:
            # Try to extract year from file path using more robust pattern
            # Look for YYYY-MM-DD pattern anywhere in the filename
            filename = os.path.basename(file_path)
            year_match = re.search(r'(\d{4})-\d{2}-\d{2}', filename)
            if year_match:
                year = int(year_match.group(1))
                data['year'] = torch.tensor(year, dtype=torch.long)
            else:
                # Try the older pattern as fallback
                year_match = re.search(r'/(\d{4})[-/]', file_path)
                if year_match:
                    year = int(year_match.group(1))
                    data['year'] = torch.tensor(year, dtype=torch.long)
                else:
                    # Default to calculating from era if year not found in path
                    if 'era' in data:
                        era = data['era'].item() if isinstance(data['era'], torch.Tensor) else data['era']
                        # Approximate year based on era
                        default_years = {0: 1970, 1: 1973, 2: 1977, 3: 1985, 4: 1990, 5: 1995}
                        data['year'] = torch.tensor(default_years.get(era, 1980), dtype=torch.long)
                    else:
                        # No era information available, default to a reasonable year
                        data['year'] = torch.tensor(1980, dtype=torch.long)
        
        # Ensure year is properly formatted as a tensor
        if 'year' in data:
            if not isinstance(data['year'], torch.Tensor):
                data['year'] = torch.tensor(data['year'], dtype=torch.long)
            elif data['year'].dim() > 1:
                # If year has extra dimensions, flatten it
                data['year'] = data['year'].view(-1)[0]
                
            # Clamp year to valid range to prevent issues in model
            min_year = 1965
            max_year = 1995
            data['year'] = torch.clamp(data['year'], min=min_year, max=max_year)
        
        # 4. Add 'date' field (day of year) if not present
        if 'date' not in data:
            # Try to extract from filename first
            filename = os.path.basename(file_path)
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
            if date_match:
                # Calculate actual day of year from date components
                year, month, day = map(int, date_match.groups())
                try:
                    date_obj = datetime.date(year, month, day)
                    # Calculate day of year (1-366)
                    day_of_year = date_obj.timetuple().tm_yday - 1  # Convert to 0-365
                    data['date'] = torch.tensor(day_of_year, dtype=torch.float)
                except ValueError:
                    # Invalid date in filename, log and skip
                    print(f"WARNING: Invalid date format in filename: {filename}")
                    return {'error': True, 'path': file_path, 'reason': 'invalid_date_format'}
            else:
                # No date in filename, log and skip
                print(f"WARNING: Could not extract date from filename: {filename}")
                return {'error': True, 'path': file_path, 'reason': 'missing_date'}
                
        # Make date field required now
        if 'date' not in data:
            print(f"WARNING: No date field in {file_path}")
            return {'error': True, 'path': file_path, 'reason': 'missing_date_field'}
        
        # 5. Check and normalize mel spectrograms
        required_fields = ['mel_spec', 'mel_spec_percussive']
        for field in required_fields:
            if field not in data:
                # Field is completely missing - this is a serious problem
                print(f"WARNING: Required field '{field}' missing in {file_path}")
                # Create empty tensor of expected shape - this should trigger obvious errors
                # rather than silent failures with dummy data
                data[field] = torch.zeros((1, 128, 100), dtype=torch.float)
            elif not isinstance(data[field], torch.Tensor):
                # Convert to tensor if it's not already
                data[field] = torch.tensor(data[field], dtype=torch.float)
        
        # 6. Ensure other feature fields exist if needed
        if 'spectral_contrast_harmonic' not in data and 'spectral_contrast' not in data:
            # Default to zeros - not ideal but better than crashing
            data['spectral_contrast_harmonic'] = torch.zeros((6, 100), dtype=torch.float)
        
        if 'chroma' not in data:
            # Default to zeros - not ideal but better than crashing
            data['chroma'] = torch.zeros((12, 100), dtype=torch.float)
        
        # Move tensors to the target device if needed
        if self.device is not None and self.device.type != 'cpu':
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    # Use non-blocking transfer for better performance
                    # Handle MPS-specific issues by ensuring contiguous tensors
                    if self.device.type == 'mps' and not value.is_contiguous():
                        value = value.contiguous()
                    data[key] = value.to(device=self.device, non_blocking=True)
        
        # Return the validated and normalized data item
        return data


class MemoryEfficientMPSDataset(PreprocessedDeadShowDataset):
    """Specialized version of PreprocessedDeadShowDataset optimized for MPS memory constraints."""
    
    def __init__(
        self,
        preprocessed_dir: str,
        augment: bool = False,
        target_sr: int = 24000,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the memory-efficient dataset for MPS.
        
        Args:
            preprocessed_dir: Directory containing preprocessed data
            augment: Whether to apply audio augmentation
            target_sr: Target sample rate
            device: Device to place tensors on (should be None or CPU for DataLoader compatibility)
        """
        # Force memory_efficient=True for MPS
        super().__init__(
            preprocessed_dir=preprocessed_dir, 
            augment=augment, 
            target_sr=target_sr, 
            device=None,  # Always load on CPU first
            memory_efficient=True
        )
        self.target_device = device
        print("Using MemoryEfficientMPSDataset with aggressive memory management")
        
        # Clear any existing cache immediately
        import gc
        gc.collect()
        try:
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear MPS cache during initialization: {e}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item with aggressive memory management.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing audio features and metadata
        """
        file_path = self.files[idx]
        
        try:
            # Load on CPU and immediately process
            data = torch.load(file_path, map_location='cpu')
            
            # Minimize tensor size - only keep essential fields and downsize where possible
            essential_fields = ['label', 'era', 'year', 'date', 'mel_spec', 'mel_spec_percussive']
            minimal_data = {k: data[k] for k in essential_fields if k in data}
            
            # Clear original data immediately
            del data
            
            # Process minimal data
            processed_data = self._process_data(minimal_data, file_path)
            
            # Clear minimal data
            del minimal_data
            
            # Force garbage collection periodically
            if idx % 10 == 0:
                import gc
                gc.collect()
                
                # Try to clear MPS cache
                try:
                    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                except Exception:
                    pass
                
            return processed_data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return error indicator
            return {'error': True, 'path': file_path}
    
    def _process_data(self, data: Dict, file_path: str) -> Dict:
        """Process data with memory optimizations for MPS."""
        # Get base processing done with parent method
        processed = super()._process_data(data, file_path)
        
        # Further optimize tensor sizes for MPS
        # 1. Use smaller data types where possible
        if 'mel_spec' in processed:
            # Convert to float16 to save memory (will be converted back to float32 when needed)
            processed['mel_spec'] = processed['mel_spec'].to(dtype=torch.float16)
            
        if 'mel_spec_percussive' in processed:
            processed['mel_spec_percussive'] = processed['mel_spec_percussive'].to(dtype=torch.float16)
        
        # 2. Keep tensors on CPU (DataLoader will handle the transfer)
        for key in list(processed.keys()):
            if isinstance(processed[key], torch.Tensor) and processed[key].device.type != 'cpu':
                processed[key] = processed[key].cpu()
        
        return processed

# Register the dataset type for easy access
MPS_EFFICIENT_DATASET = MemoryEfficientMPSDataset
