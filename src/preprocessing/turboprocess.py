#!/usr/bin/env python3
"""
Turbo-optimized preprocessing script for the Grateful Dead show dating model.
This version focuses on maximum speed using all available hardware acceleration.
"""

import argparse
import multiprocessing
import concurrent.futures
import os
import sys
import time
import datetime
import re
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as TAF
import torchaudio.transforms as transforms
from tqdm import tqdm
import warnings
import tempfile
import gc
import psutil

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set")
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')


def parse_arguments():
    """Parse command-line arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(description="Turbo-optimized preprocessing for Grateful Dead audio snippets")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/Users/charlie/projects/gdguess/data/audsnippets-all",
        help="Directory containing audio snippets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for preprocessed files (defaults to [input-dir]/preprocessed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit preprocessing to this many files (0 for no limit)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,  # Increased from 64 to 256 for high-memory systems
        help="Batch size for file processing (affects memory usage)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,  # Lower sample rate for faster processing
        help="Target sample rate for audio processing",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=15,  # Changed from 10 to 15 seconds Think of everyday life, widely recognized tools, household objects, common activities, or well-known categories from daily experience (e.g., common kitchen utensils, basic clothing items, widely recognized holiday decorations—if it's a major holiday).
        help="Length of audio clips in seconds",
    )
    parser.add_argument(
        "--ultrafast",
        action="store_true",
        help="Use simplified processing for ultra-fast speed",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of files even if they already exist",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="warning",  # Changed default from "info" to "warning" to reduce verbosity
        help="Set the logging level (debug, info, warning, error, critical)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging except errors and progress bar",
    )
    parser.add_argument(
        "--optimize-level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Optimization level (0=low, 1=medium, 2=aggressive)",
    )
    parser.add_argument(
        "--high-memory",
        action="store_true",
        help="Enable optimizations for high-memory systems (>64GB RAM)",
    )
    parser.add_argument(
        "--mps-optimize",
        action="store_true",
        help="Enable specific optimizations for Apple Silicon MPS",
    )
    return parser.parse_args()


def setup_logging(log_level, quiet=False):
    """Configure logging based on the specified level."""
    # Map string log levels to logging constants
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    numeric_level = level_map.get(log_level.lower(), logging.INFO)
    
    # If quiet mode is enabled, only show errors
    if quiet:
        numeric_level = logging.ERROR
    
    # Create a custom formatter that doesn't include the level name for DEBUG messages
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.DEBUG:
                self._style._fmt = '%(message)s'
            else:
                self._style._fmt = '%(levelname)s: %(message)s'
            return super().format(record)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)


def init_gpu():
    """Initialize and return GPU device if available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 on Ampere GPUs for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable automatic mixed precision for faster processing
        torch.cuda.amp.autocast(enabled=True)
        # Benchmark mode for faster runtime
        torch.backends.cudnn.benchmark = True
        
        # Get device properties
        prop = torch.cuda.get_device_properties(0)
        mem_gb = prop.total_memory / 1024 / 1024 / 1024
        
        # Set optimal memory settings based on available VRAM
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Set more aggressive memory management for GPUs with limited memory
        if mem_gb < 8:
            # For GPUs with less than 8GB memory
            logging.info(f"Using conservative memory settings for GPU with {mem_gb:.1f}GB VRAM")
            torch.backends.cudnn.benchmark = False  # Less memory usage but slower
            torch.backends.cuda.reserve_memory = False  # Don't reserve memory
        else:
            # For GPUs with more memory, use more aggressive settings
            logging.info(f"Using optimized memory settings for GPU with {mem_gb:.1f}GB VRAM")
            # Allow benchmarking for faster operations
            torch.backends.cudnn.benchmark = True
        
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)} with {mem_gb:.1f}GB VRAM")
        return device
    elif torch.backends.mps.is_available():
        return init_mps()
    else:
        logging.info("No GPU available, using CPU")
        return torch.device("cpu")


def init_mps():
    """Initialize optimized MPS device for Apple Silicon."""
    device = torch.device("mps")
    
    # Disable deterministic mode for better performance on M-series chips
    if hasattr(torch.backends.mps, 'deterministic_algorithms'):
        torch.backends.mps.deterministic_algorithms(False)
    
    # Force synchronization to ensure device is ready
    torch.mps.synchronize()
    
    # Empty cache to start with a clean slate
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        
    # Set environment variables for improved MPS performance, 
    # if they haven't already been set by the user
    if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' not in os.environ:
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # More aggressive memory reclamation
    
    if 'PYTORCH_MPS_ALLOCATOR_POLICY' not in os.environ:
        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'aggressive'
    
    # Get system info for logging
    import platform
    model_name = platform.machine()
    
    # Check if we're on Apple Silicon
    if model_name == 'arm64':
        try:
            # Try to get more detailed model info if possible
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                model_info = result.stdout.strip()
                # Check for M4 specifically to optimize
                if 'M4' in model_info:
                    logging.info(f"Detected Apple M4 processor, using optimized MPS settings")
                    # M4-specific optimizations can be added here if needed
                else:
                    logging.info(f"Using Apple Silicon GPU acceleration (MPS) on {model_info}")
            else:
                logging.info(f"Using Apple Silicon GPU acceleration (MPS) on {model_name}")
        except Exception:
            logging.info(f"Using Apple Silicon GPU acceleration (MPS) on {model_name}")
    else:
        logging.info("Using Apple Silicon GPU acceleration (MPS)")
    
    return device


def use_mixed_precision_for_mps(tensor, device):
    """Convert tensor to appropriate precision for MPS device.
    
    Apple Silicon MPS backend works better with float32 for some operations.
    This function ensures the tensor is in the right format for MPS.
    
    Args:
        tensor: Input tensor
        device: PyTorch device
        
    Returns:
        Tensor in appropriate precision for MPS
    """
    # Ensure tensor is on the correct device
    if tensor.device != device:
        tensor = tensor.to(device)
    
    # For MPS, ensure we're using float32 (MPS has issues with some operations in float16)
    if device.type == 'mps' and tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
        
    return tensor


def optimize_for_device(device):
    """Apply device-specific optimizations."""
    if device.type == 'mps':
        # Apple Silicon specific optimizations
        logging.info("Applying Apple Silicon specific optimizations")
        
        # No need to explicitly manage memory on Apple Silicon
        # The unified memory architecture handles this automatically
        return {
            "needs_manual_memory_management": False,
            "optimal_batch_size_multiplier": 2.0,  # M4 Max can handle larger batches
            "use_mixed_precision": True,  # Use mixed precision for better performance
            "avoid_memory_transfers": True  # Avoid unnecessary CPU-GPU transfers
        }
    elif device.type == 'cuda':
        # CUDA specific optimizations
        return {
            "needs_manual_memory_management": True,
            "optimal_batch_size_multiplier": 1.0,
            "use_mixed_precision": True,
            "avoid_memory_transfers": False
        }
    else:
        # CPU specific optimizations
        return {
            "needs_manual_memory_management": False,
            "optimal_batch_size_multiplier": 0.5,  # Smaller batches for CPU
            "use_mixed_precision": False,
            "avoid_memory_transfers": True
        }


def find_audio_files(root_dir, limit=0, fast_scan=True):
    """Find audio files with valid dates in the directory tree."""
    valid_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    logging.info(f"Scanning for audio files in: {root_dir}")
    print(f"DEBUG: Starting directory scan in {root_dir}")
    
    # Cache file for faster rescanning
    cache_path = os.path.join(root_dir, "audio_files_cache.json")
    
    # Try to use cache if available
    if fast_scan and os.path.exists(cache_path):
        try:
            print(f"DEBUG: Attempting to load cache from {cache_path}")
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                logging.info(f"Loaded {len(cached_data)} files from cache")
                print(f"DEBUG: Loaded {len(cached_data)} files from cache")
                
                if limit > 0:
                    cached_data = cached_data[:limit]
                    logging.info(f"Limited to {limit} files")
                    return cached_data
                else:
                    return cached_data
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            print(f"DEBUG: Cache error: {e}")
    else:
        print(f"DEBUG: No cache found or fast_scan disabled, performing full scan")
    
    # First pass: find all audio files with valid dates
    file_count = 0
    for root, _, files in os.walk(root_dir):
        print(f"DEBUG: Scanning directory: {root} ({len(files)} files)")
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in valid_extensions:
                continue
                
            file_path = os.path.join(root, file)
            
            # Try to extract date using regex for YYYY-MM-DD format
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file)
            if match:
                try:
                    year, month, day = map(int, match.groups())
                    date_obj = datetime.datetime(year, month, day).date()
                    
                    # Calculate era
                    era = 0
                    if 1972 <= year <= 1974:
                        era = 1
                    elif 1975 <= year <= 1979:
                        era = 2
                    elif 1980 <= year <= 1990:
                        era = 3
                    elif year > 1990:
                        era = 4
                    
                    # Add to list
                    base_date = datetime.date(1968, 1, 1)
                    days = (date_obj - base_date).days
                    
                    audio_files.append({
                        "file": file_path,
                        "date": date_obj.isoformat(),
                        "days": days,
                        "era": era,
                        "year": year
                    })
                    
                    file_count += 1
                    if file_count % 10000 == 0:
                        logging.info(f"Found {file_count} audio files...")
                    
                    # Apply limit if specified
                    if limit > 0 and file_count >= limit:
                        print(f"DEBUG: Reached limit of {limit} files, stopping scan")
                        break
                        
                except ValueError:
                    continue
        
        # Apply limit if specified
        if limit > 0 and file_count >= limit:
            break
    
    # Cache all files for faster future scans
    if len(audio_files) > 0 and fast_scan:
        try:
            print(f"DEBUG: Writing {len(audio_files)} files to cache")
            with open(cache_path, 'w') as f:
                # Only cache the full list, not the limited subset
                json.dump(audio_files, f)
                logging.info(f"Cached {len(audio_files)} files to {cache_path}")
        except Exception as e:
            logging.error(f"Error writing cache: {e}")
            print(f"DEBUG: Cache write error: {e}")
    
    logging.info(f"Found {len(audio_files)} audio files with valid dates")
    print(f"DEBUG: Found {len(audio_files)} audio files with valid dates")
    return audio_files


# GPU-optimized harmonic-percussive source separation
def gpu_hpss(audio_tensor, device, sr=16000, margin=2.0):
    """
    Perform harmonic-percussive source separation using GPU with optimized torch operations.
    Fully vectorized implementation with careful memory management.
    """
    # Ensure the input tensor is on the correct device
    if audio_tensor.device != device:
        audio_tensor = audio_tensor.to(device)
    
    # Move to device if needed
    audio = audio_tensor.to(device)
    
    # Parameters - smaller sizes for better memory usage
    n_fft = 1024
    hop_length = 256
    
    # Create window once for reuse
    # We'll reuse this same window for both STFT and ISTFT operations
    window = torch.hann_window(n_fft, device=device)
    
    # Compute STFT with no_grad to save memory
    with torch.no_grad():
        complex_spec = torch.stft(
            audio, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            return_complex=True
        )
    
    # Convert to magnitude
    mag_spec = torch.abs(complex_spec)
    
    # Get dimensions
    n_freq, n_time = mag_spec.shape
    
    # Compute horizontal and vertical differences in one step using vectorized operations
    h_diff = torch.zeros_like(mag_spec)
    v_diff = torch.zeros_like(mag_spec)
    
    # Use advanced indexing for fully vectorized operation
    if n_time > 1:
        h_diff[:, 1:] = torch.abs(mag_spec[:, 1:] - mag_spec[:, :-1])
    
    if n_freq > 1:
        v_diff[1:, :] = torch.abs(mag_spec[1:, :] - mag_spec[:-1, :])
    
    # Normalize the differences to avoid division by near-zero
    epsilon = 1e-8
    h_norm = h_diff / (torch.mean(h_diff) + epsilon)
    v_norm = v_diff / (torch.mean(v_diff) + epsilon)
    
    # Free memory early
    del h_diff, v_diff
    
    # Create masks - higher horizontal difference = more percussive
    # Use single operation for both masks
    percussive_mask = h_norm / (h_norm + v_norm + epsilon)
    harmonic_mask = 1.0 - percussive_mask
    
    # Free memory
    del h_norm, v_norm
    
    # Apply masks to complex spectrogram
    harmonic_spec = complex_spec * harmonic_mask
    percussive_spec = complex_spec * percussive_mask
    
    # Free memory
    del complex_spec, mag_spec, harmonic_mask, percussive_mask
    
    # Perform both istft operations with no_grad to save memory
    with torch.no_grad():
        # Convert back to time domain
        harmonic = torch.istft(
            harmonic_spec, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            length=audio.shape[0]
        )
        
        # Free memory
        del harmonic_spec
        
        percussive = torch.istft(
            percussive_spec, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            length=audio.shape[0]
        )
    
    # Free memory
    del percussive_spec, audio, window
    
    # Make sure the output tensors are on the correct device
    if harmonic.device != device:
        harmonic = harmonic.to(device)
    if percussive.device != device:
        percussive = percussive.to(device)
        
    return harmonic, percussive


# GPU-optimized spectral contrast implementation
def gpu_spectral_contrast(stft_magnitude, device, n_bands=6, quantile=0.02):
    """
    Compute spectral contrast using GPU with torch operations.
    Memory-optimized implementation to avoid peak memory usage.
    """
    # Ensure input tensor is on the correct device
    if stft_magnitude.device != device:
        stft_magnitude = stft_magnitude.to(device)
        
    if stft_magnitude.dim() == 2:
        stft_magnitude = stft_magnitude.unsqueeze(0)  # Add batch dimension if not present
    
    batch_size, freq_bins, time_frames = stft_magnitude.shape
    
    # Define frequency bands (similar to librosa)
    bands = torch.logspace(0, 1, n_bands + 1, base=2).to(device)
    band_edges = torch.round(bands * freq_bins / 2).long()
    band_edges = torch.clamp(band_edges, 0, freq_bins-1)
    
    # Free temporary tensor
    del bands
    
    # Pre-allocate output tensor
    contrast = torch.zeros(batch_size, n_bands, time_frames, device=device)
    
    # Calculate contrast for each band
    for i in range(n_bands):
        low_idx = band_edges[i]
        high_idx = band_edges[i + 1]
        
        if high_idx > low_idx:
            # Extract just this frequency band
            band_magnitudes = stft_magnitude[:, low_idx:high_idx, :]
            n_freqs = high_idx - low_idx
            
            # For memory efficiency, we'll compute percentiles more directly
            # Sort magnitudes within each band along frequency axis
            sorted_mags, _ = torch.sort(band_magnitudes, dim=1)
            
            # Calculate peak and valley indices
            valley_idx = max(0, min(int(n_freqs * quantile), n_freqs - 1))
            peak_idx = min(n_freqs - 1, max(0, int(n_freqs * (1 - quantile))))
            
            # Extract the specific percentile values
            valley = sorted_mags[:, valley_idx, :]
            peak = sorted_mags[:, peak_idx, :]
            
            # Calculate contrast (add a small number to avoid log(0))
            with torch.no_grad():  # No need to track gradients
                spec_contrast = torch.log(peak + 1e-5) - torch.log(valley + 1e-5)
            
            contrast[:, i, :] = spec_contrast
            
            # Free temporary tensors
            del band_magnitudes, sorted_mags, valley, peak, spec_contrast
    
    # Remove batch dimension if input didn't have it
    if batch_size == 1:
        contrast = contrast.squeeze(0)
    
    # Free the band edges tensor
    del band_edges
    
    # Ensure output is on the correct device
    if contrast.device != device:
        contrast = contrast.to(device)
        
    return contrast


# GPU-optimized chroma implementation
def gpu_chroma(stft_magnitude, device, sr=16000, n_chroma=12, use_precomputed=True):
    """
    Compute chromagram using GPU with torch operations.
    Memory-optimized implementation with explicit tensor cleanup.
    """
    # Ensure input tensor is on the correct device
    if stft_magnitude.device != device:
        stft_magnitude = stft_magnitude.to(device)
        
    if stft_magnitude.dim() == 2:
        stft_magnitude = stft_magnitude.unsqueeze(0)  # Add batch dimension if not present
    
    batch_size, freq_bins, time_frames = stft_magnitude.shape
    
    # Create a CQT-like filterbank for chroma
    n_fft = (freq_bins - 1) * 2
    
    # Create frequency bins
    freqs = torch.linspace(0, sr/2, freq_bins, device=device)
    
    # Create chroma map using vectorized operations
    chroma_map = torch.zeros(n_chroma, freq_bins, device=device)
    
    # Track valid frequencies to avoid computing for zeros
    valid_mask = freqs > 0
    valid_freqs = freqs[valid_mask]
    
    if len(valid_freqs) > 0:
        # Vectorized computation of MIDI notes
        with torch.no_grad():  # No need for gradients here
            midi_notes = 12 * torch.log2(valid_freqs / 440.0) + 69
        
        # Get chroma bins (0-11) with proper handling of negatives
        chroma_bins = torch.fmod(torch.round(midi_notes), 12)
        chroma_bins = torch.where(chroma_bins < 0, chroma_bins + 12, chroma_bins).long()
        
        # Free temporary tensor
        del midi_notes
        
        # Place frequencies in the correct chroma bins using scatter
        for i, freq_idx in enumerate(torch.nonzero(valid_mask).squeeze()):
            if i < len(chroma_bins):  # Safety check
                chroma_bin = chroma_bins[i]
                chroma_map[chroma_bin, freq_idx] = 1.0
    
    # Free temporary tensors
    del valid_mask, valid_freqs, freqs
    if 'chroma_bins' in locals():
        del chroma_bins
    
    # Normalize with proper handling of zeros
    chroma_map_sum = torch.sum(chroma_map, dim=1, keepdim=True)
    zero_mask = chroma_map_sum == 0
    chroma_map_sum = torch.where(zero_mask, torch.ones_like(chroma_map_sum), chroma_map_sum)
    chroma_map = chroma_map / chroma_map_sum
    
    # Free temporary tensors
    del chroma_map_sum, zero_mask
    
    # Apply filter bank to get chroma features
    chroma = torch.matmul(chroma_map, stft_magnitude)
    
    # Remove batch dimension if input didn't have it
    if batch_size == 1:
        chroma = chroma.squeeze(0)
    
    # Ensure output is on the correct device
    if chroma.device != device:
        chroma = chroma.to(device)
    
    if use_precomputed:
        # Ensure both return values are on the correct device
        if chroma_map.device != device:
            chroma_map = chroma_map.to(device)
        return chroma, chroma_map
    else:
        # Free filter bank if not returning it
        del chroma_map
        return chroma


# Create reusable chroma filter bank
def create_chroma_filter_bank(sr=16000, n_fft=2048, n_chroma=12, device=None):
    """Create a reusable chroma filter bank."""
    # Number of frequency bins in STFT
    freq_bins = n_fft // 2 + 1
    
    # Create frequency bins
    freqs = torch.linspace(0, sr/2, freq_bins, device=device)
    
    # Create chroma map
    chroma_map = torch.zeros(n_chroma, freq_bins, device=device)
    
    # Identify valid frequencies (> 0)
    valid_mask = freqs > 0
    valid_freqs = freqs[valid_mask]
    
    if len(valid_freqs) > 0:
        # Convert to MIDI note numbers
        midi_notes = 12 * torch.log2(valid_freqs / 440.0) + 69
        
        # Map to chroma bins (0-11)
        chroma_bins = torch.fmod(torch.round(midi_notes), 12)
        chroma_bins = torch.where(chroma_bins < 0, chroma_bins + 12, chroma_bins).long()
        
        # Populate the filter bank
        for i, (freq_idx, chroma_bin) in enumerate(zip(
                torch.nonzero(valid_mask).squeeze(), chroma_bins)):
            chroma_map[chroma_bin, freq_idx] = 1.0
    
    # Normalize
    chroma_map_sum = torch.sum(chroma_map, dim=1, keepdim=True)
    mask = chroma_map_sum > 0
    chroma_map_sum = torch.where(mask, chroma_map_sum, torch.ones_like(chroma_map_sum))
    chroma_map = chroma_map / chroma_map_sum
    
    return chroma_map


# GPU-optimized onset envelope implementation
def gpu_onset_strength(stft_magnitude, device):
    """
    Compute onset strength envelope using GPU with torch operations.
    Memory-optimized implementation with reduced allocations.
    """
    # Ensure input tensor is on the correct device
    if stft_magnitude.device != device:
        stft_magnitude = stft_magnitude.to(device)
        
    if stft_magnitude.dim() == 2:
        stft_magnitude = stft_magnitude.unsqueeze(0)  # Add batch dimension if not present
    
    # Get dimensions for pre-allocation
    batch_size, _, time_frames = stft_magnitude.shape
    
    # Pre-allocate output tensor
    onset_env = torch.zeros(batch_size, time_frames, device=device)
    
    # Compute onset strength if we have enough time frames
    if time_frames > 1:
        # Compute first-order difference along time axis in a single vectorized operation
        with torch.no_grad():  # No gradients needed here
            diff_spectrum = stft_magnitude[:, :, 1:] - stft_magnitude[:, :, :-1]
            
            # Only keep positive changes (rectification)
            diff_spectrum.clamp_(min=0.0)
            
            # Sum across frequency bins to get onset envelope
            onset_sum = torch.sum(diff_spectrum, dim=1)
            
            # Add a zero to beginning to match original length
            onset_env[:, 1:] = onset_sum
    
    # Remove batch dimension if input didn't have it
    if stft_magnitude.shape[0] == 1:
        onset_env = onset_env.squeeze(0)
    
    # Ensure output is on the correct device
    if onset_env.device != device:
        onset_env = onset_env.to(device)
        
    return onset_env


def get_feature_extractor(device, target_sr=16000, ultrafast=False, optimize_level=1, high_memory=False):
    """Create a fast feature extractor using GPU where possible."""
    
    # Create optimized mel spectrogram transform
    n_mels = 80 if ultrafast else 128  # Fewer mel bands in ultrafast mode
    n_fft = 1024 if ultrafast else 2048  # Smaller FFT in ultrafast mode
    hop_length = 256 if ultrafast else 512  # Smaller hop length in ultrafast mode
    
    # For high memory systems, we can use larger transform sizes and more aggressive optimizations
    if high_memory and not ultrafast:
        n_fft = 4096  # Larger FFT for better resolution with high memory
        hop_length = 1024  # Larger hop for faster processing
        # Pre-allocate larger buffers
        torch.backends.cuda.max_split_size_mb = 512  # Allow larger tensor splits
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear any existing allocations
            # Reserve some memory to prevent fragmentation
            torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Special handling for MPS devices
    if device.type == 'mps':
        # For MPS, we need to ensure all operations use the same device
        # Create the transform on CPU first, then move to MPS
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=8000 if target_sr >= 16000 else target_sr // 2 - 100,
            center=True,
            norm='slaney',
            mel_scale='htk'
        )
        # Move to device after creation
        mel_spec = mel_spec.to(device)
    else:
        # For CPU and CUDA, we can create directly on the device
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=8000 if target_sr >= 16000 else target_sr // 2 - 100,
            center=True,
            norm='slaney',
            mel_scale='htk'
        ).to(device)
    
    # Pre-compute hann window
    hann_window = torch.hann_window(n_fft).to(device)
    
    class FeatureExtractor:
        def __init__(self, mel_spec, device, ultrafast=False, optimize_level=1, high_memory=False):
            self.mel_spec = mel_spec
            self.device = device
            self.ultrafast = ultrafast
            self.optimize_level = optimize_level
            self.high_memory = high_memory
            self.hann_window = hann_window
            self.n_fft = n_fft
            self.hop_length = hop_length
            
            # Create resamplers dictionary to reuse them
            self.resamplers = {}
            
            # Create scaler for mixed precision
            self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
            
            # Precompute other transforms for high memory systems
            self.filter_banks = {}
            
            if high_memory:
                logging.info("Precomputing transforms for high-memory system")
                
                # Precompute chroma filter bank
                self.precompute_chroma_filterbank(target_sr)
                
                # Precompute frequency bands for spectral contrast
                self.precompute_spectral_contrast_bands()
                
                # Allocate buffers for batch processing
                self.allocate_buffers()
            
            # Precompute other transforms if aggressive optimization
            elif optimize_level >= 2:
                logging.info("Precomputing transforms for aggressive optimization")

            # Always precompute chroma filter bank (major bottleneck)
            self.chroma_filter_bank = create_chroma_filter_bank(
                sr=target_sr, n_fft=n_fft, device=device)
            logging.debug("Precomputed chroma filter bank")
        
        def precompute_chroma_filterbank(self, sr):
            """Precompute chroma filterbank for faster processing."""
            n_chroma = 12
            freq_bins = self.n_fft // 2 + 1
            
            # Create frequency bins
            freqs = torch.linspace(0, sr/2, freq_bins).to(self.device)
            
            # Create chroma map
            chroma_map = torch.zeros(n_chroma, freq_bins).to(self.device)
            
            for i in range(freq_bins):
                if freqs[i] > 0:
                    # Map frequency to chroma bin
                    midi = 12 * torch.log2(freqs[i] / 440.0) + 69
                    chroma_bin = torch.fmod(torch.round(midi), 12).long()
                    
                    # Ensure valid bin index
                    while chroma_bin < 0:
                        chroma_bin += 12
                        
                    # Add weight to the chroma bin
                    if 0 <= chroma_bin < n_chroma:
                        chroma_map[chroma_bin, i] = 1.0
            
            # Normalize chroma map
            chroma_map_sum = torch.sum(chroma_map, dim=1, keepdim=True)
            chroma_map_sum[chroma_map_sum == 0] = 1.0  # Avoid division by zero
            chroma_map = chroma_map / chroma_map_sum
            
            self.filter_banks['chroma'] = chroma_map
            
            # Free temporary tensors
            del freqs, chroma_map_sum
        
        def precompute_spectral_contrast_bands(self, n_bands=6):
            """Precompute spectral contrast frequency bands."""
            freq_bins = self.n_fft // 2 + 1
            
            # Define frequency bands (similar to librosa)
            bands = torch.logspace(0, 1, n_bands + 1, base=2).to(self.device)
            band_edges = torch.round(bands * freq_bins / 2).long()
            band_edges = torch.clamp(band_edges, 0, freq_bins-1)
            
            self.filter_banks['spectral_contrast_bands'] = band_edges
            
            # Free temporary tensors
            del bands
        
        def allocate_buffers(self):
            """Allocate reusable buffers for batch processing."""
            if self.high_memory:
                # Allocate buffers for common operations to avoid repeated allocations
                self.buffers = {
                    'stft': torch.zeros(1, self.n_fft // 2 + 1, 1000, device=self.device, dtype=torch.complex64),
                    'stft_mag': torch.zeros(1, self.n_fft // 2 + 1, 1000, device=self.device),
                    'mel': torch.zeros(1, 128, 1000, device=self.device),
                }
                logging.info("Allocated reusable buffers for batch processing")
            
        def extract_batch(self, batch_audio, target_sr, clip_length):
            """Process a batch of audio files at once for better performance."""
            batch_size = batch_audio.shape[0]
            results = []
            
            try:
                with torch.no_grad():
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                        # Ensure batch is on the correct device
                        if batch_audio.device != self.device:
                            batch_audio = batch_audio.to(self.device)
                        
                        # Convert to mono if needed
                        if batch_audio.shape[1] > 1:
                            batch_audio = torch.mean(batch_audio, dim=1, keepdim=True)
                        
                        # Process in smaller sub-batches to avoid memory issues
                        sub_batch_size = min(32, batch_size)
                        for i in range(0, batch_size, sub_batch_size):
                            end_idx = min(i + sub_batch_size, batch_size)
                            sub_batch = batch_audio[i:end_idx]
                            
                            # Apply HPSS to individual audio
                            sub_batch_results = []
                            for audio in sub_batch:
                                # Make sure the audio is contiguous for MPS
                                if not audio.is_contiguous():
                                    audio = audio.contiguous()
                                
                                # Explicit synchronization before HPSS for MPS
                                if self.device.type == 'mps':
                                    torch.mps.synchronize()
                                
                                # Apply HPSS
                                harmonic, percussive = gpu_hpss(audio.squeeze(0), self.device)
                                
                                # Extract features
                                if self.ultrafast:
                                    # Simple processing for ultrafast mode
                                    harmonic_mel = self.mel_spec(harmonic)
                                    harmonic_mel = torch.log(harmonic_mel + 1e-5)
                                    result = {
                                        "mel_spec": harmonic_mel.cpu().half()
                                    }
                                else:
                                    # Full processing with MPS synchronization
                                    if self.device.type == 'mps':
                                        torch.mps.synchronize()
                                    
                                    harmonic_mel = self.mel_spec(harmonic)
                                    harmonic_mel = torch.log(harmonic_mel + 1e-5)
                                    
                                    if self.device.type == 'mps':
                                        torch.mps.synchronize()
                                    
                                    percussive_mel = self.mel_spec(percussive)
                                    percussive_mel = torch.log(percussive_mel + 1e-5)
                                    
                                    if self.device.type == 'mps':
                                        torch.mps.synchronize()
                                    
                                    # STFT - MPS has issues with complex tensors, ensure proper handling
                                    try:
                                        stft = torch.stft(
                                            harmonic, 
                                            n_fft=self.n_fft, 
                                            hop_length=self.hop_length,
                                            window=self.hann_window,
                                            return_complex=True
                                        )
                                        stft_mag = torch.abs(stft)
                                    except RuntimeError as e:
                                        if 'MPS' in str(e):
                                            # Fallback for MPS
                                            logging.warning("Using STFT fallback for MPS")
                                            harmonic_cpu = harmonic.cpu()
                                            window_cpu = self.hann_window.cpu()
                                            stft = torch.stft(
                                                harmonic_cpu, 
                                                n_fft=self.n_fft, 
                                                hop_length=self.hop_length,
                                                window=window_cpu,
                                                return_complex=True
                                            )
                                            stft = stft.to(self.device)
                                            stft_mag = torch.abs(stft)
                                            del harmonic_cpu, window_cpu
                                    
                                    if self.device.type == 'mps':
                                        torch.mps.synchronize()
                                    
                                    # Extract other features
                                    spectral_contrast = gpu_spectral_contrast(stft_mag, self.device)
                                    chroma = gpu_chroma(stft_mag, self.device)[0]  # Only need the chroma, not the filter bank
                                    
                                    # Onset envelope
                                    try:
                                        p_stft = torch.stft(
                                            percussive, 
                                            n_fft=self.n_fft, 
                                            hop_length=self.hop_length,
                                            window=self.hann_window,
                                            return_complex=True
                                        )
                                        p_stft_mag = torch.abs(p_stft)
                                    except RuntimeError as e:
                                        if 'MPS' in str(e):
                                            # Fallback for MPS
                                            percussive_cpu = percussive.cpu()
                                            window_cpu = self.hann_window.cpu()
                                            p_stft = torch.stft(
                                                percussive_cpu, 
                                                n_fft=self.n_fft, 
                                                hop_length=self.hop_length,
                                                window=window_cpu,
                                                return_complex=True
                                            )
                                            p_stft = p_stft.to(self.device)
                                            p_stft_mag = torch.abs(p_stft)
                                            del percussive_cpu, window_cpu
                                    
                                    onset_env = gpu_onset_strength(p_stft_mag, self.device)
                                    
                                    # Create result dictionary - move all to CPU and convert to half precision
                                    result = {
                                        "mel_spec": harmonic_mel.cpu().half(),
                                        "mel_spec_percussive": percussive_mel.cpu().half(),
                                        "spectral_contrast_harmonic": spectral_contrast.cpu().half(),
                                        "chroma": chroma.cpu().half(),
                                        "onset_env": onset_env.cpu().half()
                                    }
                                    
                                    # Free memory
                                    del harmonic_mel, percussive_mel, spectral_contrast, chroma, onset_env
                                    del stft, stft_mag, p_stft, p_stft_mag, harmonic, percussive
                                
                                sub_batch_results.append(result)
                                
                                # Force garbage collection after each item
                                if self.device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                elif self.device.type == 'mps':
                                    torch.mps.empty_cache()
                        
                        results.extend(sub_batch_results)
                        
                        # Force garbage collection after each sub-batch
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        elif self.device.type == 'mps':
                            torch.mps.empty_cache()
    
            except Exception as e:
                logging.error(f"Error in batch feature extraction: {str(e)}")
                return [None] * batch_size
            
            return results
    
    return FeatureExtractor(mel_spec, device, ultrafast, optimize_level, high_memory)


def process_batch(args):
    """Process a batch of files in a worker process."""
    batch_id, file_batch, output_dir, target_sr, clip_length, ultrafast, use_gpu, force_reprocess, optimize_level, high_memory = args
    
    # Track memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    logging.debug(f"Batch {batch_id}: Initial memory usage: {initial_memory:.2f} MB")
    
    # Set up device
    if use_gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            # For M4 Max, we'll use true batch processing
            use_true_batch = True
            logging.info(f"Batch {batch_id}: Using true batch processing for Apple Silicon")
        else:
            device = init_gpu()
            use_true_batch = False
    else:
        device = torch.device("cpu")
        use_true_batch = False
    
    # Apply device-specific optimizations
    device_opts = optimize_for_device(device)
    
    # Adjust batch processing based on device type
    needs_manual_memory = device_opts["needs_manual_memory_management"]
    avoid_transfers = device_opts["avoid_memory_transfers"]
    
    # Create feature extractor
    logging.debug(f"Batch {batch_id}: Creating feature extractor")
    feature_extractor = get_feature_extractor(device, target_sr, ultrafast, optimize_level, high_memory)
    
    # Create a single resampler dictionary at the batch level
    resamplers = {}
    
    # With high memory, we can cache audio and features
    audio_cache = {}
    feature_cache = {}
    
    # For high memory mode, use batch file writing to reduce I/O operations
    pending_saves = []
    
    results = []
    start_time = time.time()
    batch_size = len(file_batch)
    completed = 0
    
    # For M4 Max with MPS, use true batch processing
    if use_true_batch and device.type == 'mps' and high_memory:
        logging.info(f"Batch {batch_id}: Using optimized batch processing for M4 Max")
        
        # Process in sub-batches to avoid memory issues
        max_files_in_memory = 16  # Process 16 files at once
        
        for i in range(0, batch_size, max_files_in_memory):
            sub_batch = file_batch[i:i + max_files_in_memory]
            sub_batch_size = len(sub_batch)
            
            logging.debug(f"Batch {batch_id}: Processing sub-batch {i//max_files_in_memory + 1} with {sub_batch_size} files")
            
            # Load audio files
            audio_batch = []
            file_infos = []
            
            for file_info in sub_batch:
                try:
                    # Load audio
                    audio_path = file_info["file"]
                    audio, orig_sr = torchaudio.load(audio_path)
                    
                    # Move audio to the correct device immediately after loading
                    audio = audio.to(device)
                    
                    # Resample if needed
                    if orig_sr != target_sr:
                        resampler_key = f"{orig_sr}_{target_sr}"
                        if resampler_key not in resamplers:
                            resamplers[resampler_key] = torchaudio.transforms.Resample(
                                orig_sr, target_sr
                            ).to(device)
                        
                        audio = resamplers[resampler_key](audio)
                    
                    # Add to batch
                    audio_batch.append(audio)
                    file_infos.append(file_info)
                    
                except Exception as e:
                    logging.error(f"Error loading {file_info['file']}: {e}")
                    results.append((file_info.get("idx", i), False))
            
            if not audio_batch:
                continue
                
            # Process the sub-batch
            batch_results = process_multiple_files(
                audio_batch, 
                file_infos, 
                device, 
                feature_extractor, 
                output_dir, 
                target_sr, 
                clip_length, 
                force_reprocess
            )
            
            # Add results
            results.extend(batch_results)
            completed += sum(1 for _, success in batch_results if success)
            
            # Clean up memory after each sub-batch
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            # Log progress
            sub_batch_time = time.time() - start_time
            if sub_batch_time > 0:
                files_per_second = completed / sub_batch_time
                logging.info(f"Batch {batch_id}: Processed {completed}/{batch_size} files at {files_per_second:.2f} files/s")
    
    else:
        # Original sequential processing
        # For high memory systems, we can preload batch data
        if high_memory:
            # Use a limited preload for better memory management
            # Only preload up to a certain number of files at once based on memory conditions
            max_preload = min(10, batch_size)  # Limit preloading to max 10 files at once
            logging.debug(f"Batch {batch_id}: Using high-memory mode with max {max_preload} preloaded files")
            
            # Only preload a few files at a time to avoid memory issues
            for file_info in file_batch[:max_preload]:
                try:
                    audio_path = file_info["file"]
                    audio_result = torchaudio.load(audio_path)
                    audio, orig_sr = audio_result
                    
                    # Move audio to device immediately after loading for high-memory mode
                    if device.type == 'mps':
                        audio = audio.to(device)
                        
                    audio_cache[audio_path] = (audio, orig_sr)
                except Exception as e:
                    logging.debug(f"Could not preload {audio_path}: {e}")
            logging.debug(f"Batch {batch_id}: Preloaded {len(audio_cache)} files")
        
        # Check memory after setup
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        logging.debug(f"Batch {batch_id}: Memory after setup: {current_memory:.2f} MB (+{current_memory - initial_memory:.2f} MB)")
        
        for i, file_info in enumerate(file_batch):
            file_idx = batch_id * batch_size + i
            
            try:
                file_start_time = time.time()
                
                # Get output path based on original file path for better organization
                rel_path = os.path.relpath(file_info["file"], os.path.dirname(output_dir))
                output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
                os.makedirs(output_subdir, exist_ok=True)
                
                filename = os.path.basename(file_info["file"])
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_subdir, f"{base_name}.pt")
                
                logging.debug(f"Batch {batch_id}: Processing file {i+1}/{batch_size}: {file_info['file']}")
                
                # Skip if file already exists and force_reprocess is False
                if os.path.exists(output_path) and not force_reprocess:
                    logging.debug(f"Batch {batch_id}: File {i+1} already exists, skipping")
                    results.append((file_idx, True))
                    completed += 1
                    continue
                
                # Extract features using preloaded audio if available
                features = None
                error = None
                
                # Check memory before processing file
                before_file_memory = process.memory_info().rss / 1024 / 1024  # MB
                logging.debug(f"Batch {batch_id}: Before processing file {i+1}/{batch_size}: {before_file_memory:.2f} MB")
                
                try:
                    # FILE LOADING
                    if high_memory and file_info["file"] in audio_cache:
                        # Use cached audio
                        audio, orig_sr = audio_cache[file_info["file"]]
                        # Ensure audio is on the correct device
                        if audio.device != device:
                            audio = audio.to(device)
                    else:
                        # Load audio directly
                        audio_result = torchaudio.load(file_info["file"])
                        audio, orig_sr = audio_result
                        # Move to device immediately after loading
                        audio = audio.to(device)
                    
                    # RESAMPLING - Modified approach
                    if orig_sr != target_sr:
                        # Check if we already have this resampler
                        resampler_key = f"{orig_sr}_{target_sr}"
                        if resampler_key not in resamplers:
                            resamplers[resampler_key] = torchaudio.transforms.Resample(
                                orig_sr, target_sr
                            ).to(device)
                        
                        # Use the cached resampler
                        resampled_audio = resamplers[resampler_key](audio)
                        del audio
                        audio = resampled_audio
                    
                    # Convert to mono if stereo
                    if audio.size(0) > 1:
                        mono_audio = torch.mean(audio, dim=0, keepdim=True)
                        del audio
                        audio = mono_audio
                    
                    # Remove channel dimension for processing
                    audio = audio.squeeze(0)
                    
                    # Standardize length
                    desired_length = target_sr * clip_length
                    if audio.size(0) < desired_length:
                        # Pad shorter audio
                        padded_audio = F.pad(audio, (0, desired_length - audio.size(0)))
                        del audio
                        audio = padded_audio
                    else:
                        # Trim longer audio
                        trimmed_audio = audio[:desired_length]
                        if trimmed_audio.size(0) != audio.size(0):  # Only replace if actually changed
                            del audio
                            audio = trimmed_audio
                    
                    # FEATURE EXTRACTION
                    if ultrafast or optimize_level >= 2:
                        # Simple processing for ultrafast mode
                        harmonic_mel = feature_extractor.mel_spec(audio)
                        harmonic_mel = torch.log(harmonic_mel + 1e-5)
                        
                        # Move to CPU
                        result = {
                            "mel_spec": harmonic_mel.cpu().half()
                        }
                        
                        # Free GPU tensors
                        del harmonic_mel, audio
                        
                        features = result
                    else:
                        # Full processing path
                        
                        # HPSS - Using high-memory efficient version with explicit scoping
                        if high_memory and use_gpu:
                            # For high-memory GPU mode, use a more efficient HPSS approach
                            with torch.no_grad():  # Use no_grad context for all operations
                                harmonic, percussive = gpu_hpss(audio, device)
                        else:
                            harmonic, percussive = gpu_hpss(audio, device)
                        
                        # Free original audio immediately
                        del audio
                        
                        # Immediately do garbage collection after HPSS which is a memory-intensive operation
                        if needs_manual_memory and use_gpu:
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Check memory after HPSS
                        if logging.getLogger().level <= logging.DEBUG:
                            if device.type == 'mps':
                                mem_gb, mem_msg = log_memory_apple()
                                logging.debug(f"After HPSS: {mem_msg}")
                            else:
                                post_hpss_memory = process.memory_info().rss / 1024 / 1024
                                logging.debug(f"After HPSS: {post_hpss_memory:.2f} MB")
                        
                        # MEL SPECTROGRAMS - using no_grad for memory efficiency
                        with torch.no_grad():  # Disable gradient calculation for inference
                            harmonic_mel = feature_extractor.mel_spec(harmonic)
                            harmonic_mel = torch.log(harmonic_mel + 1e-5)
                            
                            percussive_mel = feature_extractor.mel_spec(percussive)
                            percussive_mel = torch.log(percussive_mel + 1e-5)
                        
                        # Check memory after mel
                        post_mel_memory = process.memory_info().rss / 1024 / 1024
                        logging.debug(f"After mel: {post_mel_memory:.2f} MB")
                        
                        # STFT - use no_grad for all operations
                        with torch.no_grad():
                            stft = torch.stft(
                                harmonic, 
                                n_fft=feature_extractor.n_fft, 
                                hop_length=feature_extractor.hop_length,
                                window=feature_extractor.hann_window,
                                return_complex=True
                            )
                            stft_mag = torch.abs(stft)
                        
                        # Free harmonic immediately after use
                        del harmonic
                        
                        # Check memory after STFT
                        post_stft_memory = process.memory_info().rss / 1024 / 1024
                        logging.debug(f"After STFT: {post_stft_memory:.2f} MB")
                        
                        # SPECTRAL FEATURES
                        # Process each spectral feature independently to limit memory usage
                        # Spectral contrast
                        with torch.no_grad():
                            spectral_contrast = gpu_spectral_contrast(stft_mag, device)
                        
                        # Chroma features with better memory handling for high-memory mode
                        if high_memory and hasattr(feature_extractor, 'chroma_filter_bank'):
                            with torch.no_grad():
                                chroma = torch.matmul(feature_extractor.chroma_filter_bank, stft_mag)
                        else:
                            chroma, _ = gpu_chroma(stft_mag, device, sr=target_sr)
                        
                        # Free STFT tensors immediately after use
                        del stft
                        
                        # Onset envelope
                        with torch.no_grad():
                            p_stft = torch.stft(
                                percussive, 
                                n_fft=feature_extractor.n_fft, 
                                hop_length=feature_extractor.hop_length,
                                window=feature_extractor.hann_window,
                                return_complex=True
                            )
                            p_stft_mag = torch.abs(p_stft)
                            onset_env = gpu_onset_strength(p_stft_mag, device)
                        
                        # Free tensors we don't need immediately
                        del p_stft, p_stft_mag, percussive, stft_mag
                        gc.collect()
                        if use_gpu:
                            torch.cuda.empty_cache()
                        
                        # Check memory after spectral features
                        post_spectral_memory = process.memory_info().rss / 1024 / 1024
                        logging.debug(f"After spectral: {post_spectral_memory:.2f} MB")
                        
                        # MOVE TO CPU - do this in smaller chunks to avoid memory spikes
                        # Process one tensor at a time
                        result = {}
                        
                        # Move each tensor to CPU individually to avoid memory spikes
                        result["mel_spec"] = harmonic_mel.cpu().half()
                        del harmonic_mel
                        
                        result["mel_spec_percussive"] = percussive_mel.cpu().half()
                        del percussive_mel
                        
                        result["spectral_contrast_harmonic"] = spectral_contrast.cpu().half()
                        del spectral_contrast
                        
                        result["chroma"] = chroma.cpu().half()
                        del chroma
                        
                        result["onset_env"] = onset_env.cpu().half()
                        del onset_env
                        
                        # Force cleanup after CPU transfer
                        gc.collect()
                        if use_gpu:
                            torch.cuda.empty_cache()
                        
                        # Check memory after CPU transfer
                        post_cpu_memory = process.memory_info().rss / 1024 / 1024
                        logging.debug(f"After CPU transfer: {post_cpu_memory:.2f} MB")
                         
                        features = result
                    
                    # Force garbage collection after processing
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Error in processing file {file_info['file']}: {str(e)}")
                    features = None
                    error = f"Error: {str(e)}"
                
                # Check memory after processing file
                after_file_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = after_file_memory - before_file_memory
                logging.debug(f"Batch {batch_id}: After processing file {i+1}/{batch_size}: {after_file_memory:.2f} MB (+{memory_increase:.2f} MB)")
                
                if features:
                    # Add metadata
                    features["label"] = torch.tensor(file_info["days"], dtype=torch.float)
                    features["era"] = torch.tensor(file_info["era"], dtype=torch.long)
                    features["file"] = file_info["file"]
                    
                    # SAVE FEATURES
                    if high_memory:
                        # Store for batch saving
                        pending_saves.append((features, output_path))
                        
                        # Save in batches of 16 to reduce I/O operations
                        if len(pending_saves) >= 16:
                            for feat, path in pending_saves:
                                torch.save(feat, path)
                            pending_saves.clear()
                    else:
                        # Save the preprocessed data immediately
                        torch.save(features, output_path)
                    
                    # For high-memory mode with large batches, only cache a limited number
                    if high_memory and len(feature_cache) < 32:  # Limit cache size
                        feature_cache[file_info["file"]] = features
                    
                    results.append((file_info.get("idx", i), True))
                    completed += 1
                    
                    logging.debug(f"Batch {batch_id}: File {i+1} completed in {time.time() - file_start_time:.3f}s")
                    
                else:
                    logging.error(f"Error processing file {file_info['file']}: {error}")
                    results.append((file_info.get("idx", i), error))
                    
                # Manual garbage collection after each file
                gc.collect()
                
                # Release CUDA memory if using GPU
                if use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # In high-memory mode with large batches, we may need to dynamically load audio
                if high_memory and i >= max_preload - 1 and i + max_preload < batch_size:
                    # If we're approaching the end of our preloaded files, load the next one
                    next_idx = i + max_preload
                    if next_idx < batch_size and file_batch[next_idx]["file"] not in audio_cache:
                        try:
                            next_file = file_batch[next_idx]["file"]
                            audio_result = torchaudio.load(next_file)
                            audio_cache[next_file] = audio_result
                            
                            # If cache is getting too large, remove older entries
                            if len(audio_cache) > max_preload + 2:
                                # Remove the oldest entry (which should be processed by now)
                                if i > 0 and file_batch[i-1]["file"] in audio_cache:
                                    del audio_cache[file_batch[i-1]["file"]]
                        except Exception as e:
                            logging.debug(f"Could not preload next file {next_file}: {e}")
                
            except Exception as e:
                import traceback
                logging.error(f"Exception processing file {file_info['file']}: {str(e)}")
                logging.debug(traceback.format_exc())
                results.append((file_idx, f"Error: {str(e)}"))
        
        # Save any remaining files
        if pending_saves:
            for feat, path in pending_saves:
                torch.save(feat, path)
            pending_saves.clear()
    
    # Clear caches to free memory
    audio_cache.clear()
    feature_cache.clear()
    
    # Explicitly delete resamplers at the end of batch processing
    for key in list(resamplers.keys()):
        del resamplers[key]
    resamplers.clear()
    del resamplers
    
    # Force garbage collection
    gc.collect()
    
    # Check final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    logging.debug(f"Batch {batch_id}: Final memory usage: {final_memory:.2f} MB (Net change: {final_memory - initial_memory:.2f} MB)")
    
    # Calculate batch statistics
    elapsed = time.time() - start_time
    files_per_second = completed / elapsed if elapsed > 0 else 0
    
    if elapsed > 0:
        logging.info(f"Batch {batch_id} completed: {completed}/{batch_size} files in {elapsed:.2f}s ({files_per_second:.2f} files/s)")
    
    return results, completed, elapsed


def process_multiple_files(audio_batch, file_infos, device, feature_extractor, output_dir, target_sr, clip_length, force_reprocess=False):
    """Process multiple audio files in a single batch for better performance."""
    results = []
    
    try:
        # Calculate the desired length for all audio files
        desired_length = target_sr * clip_length
        
        # Pre-allocate memory for the entire batch
        if feature_extractor.high_memory:
            # For high memory systems, process larger chunks at once
            max_batch_size = 256 if device.type == 'cuda' else 64
            
            # Split into sub-batches if needed
            for i in range(0, len(audio_batch), max_batch_size):
                sub_batch = audio_batch[i:i + max_batch_size]
                sub_infos = file_infos[i:i + max_batch_size]
                
                # Process sub-batch
                with torch.no_grad():
                    # Pre-allocate tensors for the entire sub-batch
                    device_audio_batch = torch.zeros(
                        (len(sub_batch), 1, desired_length),
                        device=device,
                        dtype=torch.float32
                    )
                    
                    # Fill the pre-allocated tensor
                    for j, audio in enumerate(sub_batch):
                        if isinstance(audio, torch.Tensor):
                            # Convert to mono if stereo
                            if audio.dim() == 2 and audio.size(0) > 1:
                                audio = torch.mean(audio, dim=0, keepdim=True)
                            elif audio.dim() == 1:
                                audio = audio.unsqueeze(0)
                                
                            # Standardize length
                            if audio.size(1) > desired_length:
                                audio = audio[:, :desired_length]
                            elif audio.size(1) < desired_length:
                                padding = desired_length - audio.size(1)
                                audio = torch.nn.functional.pad(audio, (0, padding))
                                
                            device_audio_batch[j] = audio
                        
                    # Process the sub-batch
                    batch_features = feature_extractor.extract_batch(
                        device_audio_batch, target_sr, clip_length
                    )
                    
                    # Save results
                    for features, file_info in zip(batch_features, sub_infos):
                        if features is not None:
                            save_path = os.path.join(
                                output_dir,
                                os.path.splitext(os.path.basename(file_info["file"]))[0] + ".pt"
                            )
                            torch.save(features, save_path)
                            results.append((file_info.get("idx", len(results)), True))
                        else:
                            results.append((file_info.get("idx", len(results)), False))
                    
                    # Force memory cleanup after each sub-batch
                    del device_audio_batch
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif device.type == 'mps':
                        torch.mps.empty_cache()
        else:
            # For standard memory systems, use original processing
            normalized_audio_batch = []
            valid_indices = []
            valid_file_infos = []
            
            for i, (audio, file_info) in enumerate(zip(audio_batch, file_infos)):
                try:
                    if not isinstance(audio, torch.Tensor):
                        audio = torch.tensor(audio, device=device)
                    elif audio.device != device:
                        audio = audio.to(device)
                    
                    # Convert to mono if stereo
                    if audio.dim() == 2 and audio.size(0) > 1:
                        audio = torch.mean(audio, dim=0, keepdim=True)
                    elif audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    
                    # Standardize length
                    if audio.size(1) > desired_length:
                        audio = audio[:, :desired_length]
                    elif audio.size(1) < desired_length:
                        padding = desired_length - audio.size(1)
                        audio = torch.nn.functional.pad(audio, (0, padding))
                    
                    normalized_audio_batch.append(audio)
                    valid_indices.append(i)
                    valid_file_infos.append(file_info)
                except Exception as e:
                    logging.error(f"Error normalizing audio {file_info['file']}: {str(e)}")
                    results.append((file_info.get("idx", i), False))
            
            if normalized_audio_batch:
                device_audio_batch = torch.stack(normalized_audio_batch)
                batch_features = feature_extractor.extract_batch(
                    device_audio_batch, target_sr, clip_length
                )
                
                for features, file_info in zip(batch_features, valid_file_infos):
                    if features is not None:
                        save_path = os.path.join(
                            output_dir,
                            os.path.splitext(os.path.basename(file_info["file"]))[0] + ".pt"
                        )
                        torch.save(features, save_path)
                        results.append((file_info.get("idx", len(results)), True))
                    else:
                        results.append((file_info.get("idx", len(results)), False))
    
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")
        # Return failure for all files in batch
        results.extend((info.get("idx", i), False) for i, info in enumerate(file_infos))
    
    return results


def main():
    """Main preprocessing function."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level, args.quiet)
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set")
    
    start_time = time.time()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.input_dir, "preprocessed")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Print settings
    print(f"Settings:")
    print(f"  - Sample rate: {args.sample_rate} Hz")
    print(f"  - Clip length: {args.clip_length} seconds")
    print(f"  - Ultra-fast mode: {'Enabled' if args.ultrafast else 'Disabled'}")
    print(f"  - GPU acceleration: {'Enabled' if args.use_gpu else 'Disabled'}")
    print(f"  - Force reprocessing: {'Enabled' if args.force else 'Disabled'}")
    print(f"  - Optimization level: {args.optimize_level}")
    print(f"  - High memory mode: {'Enabled' if args.high_memory else 'Disabled'}")
    
    # Check for Apple Silicon and apply MPS optimizations
    if args.mps_optimize and torch.backends.mps.is_available():
        print(f"  - Apple Silicon MPS optimizations: Enabled")
        # Force GPU usage when MPS optimizations are requested
        args.use_gpu = True
        
        # Initialize MPS device for optimizations
        device = init_mps()
        logging.info("Initialized MPS device with optimizations")
    elif args.mps_optimize:
        print(f"  - Apple Silicon MPS optimizations: Requested but MPS not available")
    else:
        print(f"  - Apple Silicon MPS optimizations: Disabled")
    
    # Find all valid audio files
    audio_files = find_audio_files(args.input_dir, args.limit)
    
    if not audio_files:
        logging.error("No audio files found with valid dates. Exiting.")
        return
    
    # Determine optimal number of workers
    if args.workers <= 0:
        num_cores = multiprocessing.cpu_count()
        # With high memory, we can use more workers
        if args.high_memory:
            if args.use_gpu:
                # For GPU with high-memory, use fewer workers to avoid GPU memory issues
                num_workers = min(4, num_cores)
                print("Using fewer workers for GPU high-memory mode to avoid VRAM issues")
            else:
                num_workers = min(16, num_cores)  # Adjusted for high-memory CPU mode
        # Limit workers on lower optimization levels to avoid GPU memory issues
        elif args.optimize_level <= 1 and args.use_gpu:
            num_workers = min(2, num_cores)  # Even more conservative for GPU
        else:
            num_workers = min(8, num_cores)  # Reduced from 16 to 8 for better stability
    else:
        num_workers = args.workers
    
    print(f"Using {num_workers} parallel workers")
    
    # Dynamic batch size based on optimization level and available memory
    # Provide warnings but respect user's batch size choice
    if args.high_memory:
        if args.use_gpu and torch.backends.mps.is_available():
            print(f"Note: Large batch sizes on MPS may cause tensor size mismatches")
            print(f"If you experience errors, try reducing the batch size")
            batch_size = args.batch_size
        elif args.use_gpu:
            print(f"Note: Using GPU with batch size {args.batch_size}")
            print(f"If you experience VRAM issues, consider reducing the batch size")
            batch_size = args.batch_size
        else:
            batch_size = args.batch_size
    else:
        print(f"Note: Using batch size {args.batch_size} without high-memory optimizations")
        print(f"If you experience memory issues, consider enabling --high-memory")
        batch_size = args.batch_size
    
    batches = [audio_files[i:i + batch_size] for i in range(0, len(audio_files), batch_size)]
    print(f"Processing {len(audio_files)} files in {len(batches)} batches of {batch_size} files each")
    
    # For high-memory systems, we can use a memory cache to store processed results
    results_cache = {}
    if args.high_memory:
        print("High memory mode enabled: Using in-memory caching for processed files")
        
        # For high-memory systems, we can use memory mapping for large arrays
        try:
            # Create a temporary file to track processed files
            temp_dir = tempfile.mkdtemp()
            batch_results_file = os.path.join(temp_dir, "batch_results.npy")
            
            # Pre-allocate a memory-mapped array for batch results
            # This allows sharing results between processes without copying
            batch_results = np.memmap(
                batch_results_file,
                dtype=np.int32,
                mode='w+',
                shape=(len(audio_files), 3)  # [index, status, processing_time_ms]
            )
            print(f"Using memory-mapped array for batch results tracking")
        except Exception as e:
            logging.warning(f"Failed to create memory-mapped array: {e}")
            batch_results = None
    else:
        batch_results = None
    
    # For GPU optimization with high memory mode
    if args.high_memory and args.use_gpu:
        # Pre-initialize GPU to avoid first-time overhead
        print("Pre-initializing GPU for high-memory mode...")
        device = init_gpu()
        
        # Pre-allocate some GPU memory to avoid fragmentation
        if torch.cuda.is_available():
            try:
                # Warm up the GPU with a small tensor operation and clear it
                dummy_tensor = torch.zeros(100, 100, device=device)
                dummy_tensor = torch.matmul(dummy_tensor, dummy_tensor)
                del dummy_tensor
                torch.cuda.empty_cache()
                print("GPU initialized successfully")
            except Exception as e:
                logging.warning(f"GPU initialization warning: {e}")
    
    # Process in parallel
    successful = 0
    failed = 0
    total_processing_time = 0
    total_files_processed = 0
    
    # Shared cache for results between batches (high memory mode only)
    shared_file_info = {}
    if args.high_memory:
        # Precompute file paths and organize by common attributes to optimize batch processing
        for file_info in audio_files:
            year = file_info.get("year", 0)
            if year not in shared_file_info:
                shared_file_info[year] = []
            shared_file_info[year].append(file_info)
        
        print(f"Pre-indexed {len(audio_files)} files by year for optimized processing")
    
    with tqdm(total=len(audio_files), desc="Preprocessing") as pbar:
        # Create worker arguments
        worker_args = [
            (batch_id, batch, output_dir, args.sample_rate, args.clip_length, 
             args.ultrafast, args.use_gpu, args.force, args.optimize_level, args.high_memory)
            for batch_id, batch in enumerate(batches)
        ]
        
        # Process batches
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            futures = [executor.submit(process_batch, arg) for arg in worker_args]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    (batch_results, completed, elapsed) = future.result()
                    for _, status in batch_results:
                        if status is True:
                            successful += 1
                        else:
                            failed += 1
                    
                    # Update processing statistics
                    if completed > 0 and elapsed > 0:
                        total_processing_time += elapsed
                        total_files_processed += completed
                    
                    # Update progress bar
                    pbar.update(len(batch_results))
                    
                    # Dynamic update of estimated time
                    if total_files_processed > 0 and total_processing_time > 0:
                        current_rate = total_files_processed / total_processing_time
                        remaining = (len(audio_files) - pbar.n) / current_rate
                        pbar.set_postfix(rate=f"{current_rate:.2f} it/s", remaining=f"{remaining/3600:.1f}h")
                    
                except Exception as e:
                    logging.error(f"Batch processing error: {e}")
                    pbar.update(batch_size)  # Approximate update
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Preprocessing completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    
    # Calculate throughput
    if elapsed_time > 0:
        throughput = successful / elapsed_time
        print(f"Processing speed: {throughput:.2f} files/second")
        
        # Estimate full dataset time
        if args.limit > 0 and successful > 0:
            total_files = 974145
            estimated_total_time = total_files / throughput
            est_hours, remainder = divmod(estimated_total_time, 3600)
            est_minutes, _ = divmod(remainder, 60)
            print(f"Estimated time for full dataset: {int(est_hours)}h {int(est_minutes)}m")
    
    # Clean up resources in high-memory mode
    if args.high_memory and batch_results is not None:
        try:
            del batch_results
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        except Exception as e:
            logging.warning(f"Error cleaning up temporary files: {e}")
    
    # Final memory cleanup
    gc.collect()
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0) 