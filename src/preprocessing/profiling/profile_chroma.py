#!/usr/bin/env python3
"""
Dedicated profiling script for the chroma feature extraction bottleneck
"""

import os
import time
import torch
import torchaudio
import torch.nn.functional as F
from ..turboprocess import (
    gpu_chroma,
    init_gpu
)

def profile_chroma_implementations(audio_file, device, clip_length=15):
    """Profile different chroma implementations."""
    print(f"Profiling chroma implementations for file: {audio_file}")
    
    # Load audio file
    audio, sr = torchaudio.load(audio_file)
    
    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
        sr = 16000
    
    # Convert to mono if stereo
    if audio.size(0) > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Remove channel dimension for processing
    audio = audio.squeeze(0)
    
    # Standardize length
    desired_length = sr * clip_length
    if audio.size(0) < desired_length:
        # Pad shorter audio
        audio = F.pad(audio, (0, desired_length - audio.size(0)))
    else:
        # Trim longer audio
        audio = audio[:desired_length]
    
    # Move to device
    audio = audio.to(device)
    
    # Compute STFT
    n_fft = 2048
    hop_length = 512
    hann_window = torch.hann_window(n_fft).to(device)
    
    # First make STFT
    stft = torch.stft(
        audio, 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=hann_window,
        return_complex=True
    )
    stft_mag = torch.abs(stft)
    
    # Profile the original implementation
    print("\n1. Original GPU Chroma Implementation:")
    repeats = 5
    times = []
    for i in range(repeats):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        chroma = gpu_chroma(stft_mag, device, sr=sr)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Chroma shape: {chroma.shape}")
    
    # Vectorized implementation
    print("\n2. Vectorized Chroma Implementation:")
    
    def vectorized_chroma(stft_magnitude, device, sr=16000, n_chroma=12):
        if stft_magnitude.dim() == 2:
            stft_magnitude = stft_magnitude.unsqueeze(0)
        
        batch_size, freq_bins, time_frames = stft_magnitude.shape
        
        # Create frequency bins for the entire spectrum at once
        freqs = torch.linspace(0, sr/2, freq_bins).to(device)
        
        # Only consider non-zero frequencies
        valid_mask = freqs > 0
        valid_freqs = freqs[valid_mask]
        
        # Compute MIDI note numbers for all frequencies at once
        midi_notes = 12 * torch.log2(valid_freqs / 440.0) + 69
        
        # Map to chroma bins (0-11)
        chroma_bins = torch.fmod(torch.round(midi_notes), 12).long()
        
        # Ensure valid bin indices
        chroma_bins = torch.remainder(chroma_bins, 12)
        
        # Vectorized approach to creating the chroma map
        chroma_map = torch.zeros(n_chroma, freq_bins, device=device)
        
        # Use scatter to populate the chroma map
        valid_indices = torch.nonzero(valid_mask).squeeze()
        for i, idx in enumerate(valid_indices):
            if i < len(chroma_bins):
                chroma_map[chroma_bins[i], idx] = 1.0
        
        # Normalize along frequency axis
        chroma_map_sum = torch.sum(chroma_map, dim=1, keepdim=True)
        chroma_map_sum[chroma_map_sum == 0] = 1.0  # Avoid division by zero
        chroma_map = chroma_map / chroma_map_sum
        
        # Apply chroma map to STFT magnitudes with optimized matrix multiplication
        chroma = torch.matmul(chroma_map, stft_magnitude)
        
        if batch_size == 1:
            chroma = chroma.squeeze(0)
        
        return chroma
    
    times = []
    for i in range(repeats):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        chroma = vectorized_chroma(stft_mag, device, sr=sr)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Chroma shape: {chroma.shape}")
    
    # Pre-computed filterbank implementation
    print("\n3. Pre-computed Filterbank Chroma Implementation:")
    
    def precomputed_filterbank_chroma(stft_magnitude, device, sr=16000, n_chroma=12):
        if stft_magnitude.dim() == 2:
            stft_magnitude = stft_magnitude.unsqueeze(0)
        
        batch_size, freq_bins, time_frames = stft_magnitude.shape
        
        # Create frequency bins for the entire spectrum at once
        freqs = torch.linspace(0, sr/2, freq_bins).to(device)
        
        # Create the chroma filterbank
        chroma_map = torch.zeros(n_chroma, freq_bins, device=device)
        
        # Precompute all MIDI notes at once
        with torch.no_grad():  # No need for gradients here
            valid_mask = freqs > 0
            valid_freqs = freqs[valid_mask]
            midi_notes = 12 * torch.log2(valid_freqs / 440.0) + 69
            chroma_bins = torch.fmod(torch.round(midi_notes), 12).long()
            chroma_bins = torch.remainder(chroma_bins, 12)  # Ensure positive values
            
            # Use index_put for faster assignment
            valid_indices = torch.nonzero(valid_mask).squeeze()
            for i, idx in enumerate(valid_indices):
                if i < len(chroma_bins):
                    bin_idx = chroma_bins[i]
                    chroma_map[bin_idx, idx] = 1.0
            
            # Normalize with sum reduction
            chroma_map_sum = torch.sum(chroma_map, dim=1, keepdim=True)
            chroma_map_sum[chroma_map_sum == 0] = 1.0  # Avoid division by zero
            chroma_map = chroma_map / chroma_map_sum
        
        # Apply chroma map with batched matmul
        chroma = torch.matmul(chroma_map, stft_magnitude)
        
        if batch_size == 1:
            chroma = chroma.squeeze(0)
        
        return chroma, chroma_map
    
    # First, precompute the filterbank
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    _, chroma_filterbank = precomputed_filterbank_chroma(stft_mag, device, sr=sr)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    precompute_time = time.time() - start_time
    print(f"  Filterbank precomputation time: {precompute_time:.4f}s")
    
    # Now test using the precomputed filterbank
    times = []
    for i in range(repeats):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        chroma = torch.matmul(chroma_filterbank, stft_mag)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Chroma shape: {chroma.shape}")
    
    # CPU vs MPS/GPU comparison
    print("\n4. CPU vs GPU Comparison:")
    
    # GPU timing (using the fastest method)
    times = []
    for i in range(3):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        chroma_gpu = torch.matmul(chroma_filterbank, stft_mag)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        times.append(end_time - start_time)
    
    gpu_avg_time = sum(times) / len(times)
    print(f"  GPU average time: {gpu_avg_time:.4f}s")
    
    # CPU timing
    cpu_device = torch.device("cpu")
    cpu_stft_mag = stft_mag.cpu()
    cpu_filterbank = chroma_filterbank.cpu()
    
    times = []
    for i in range(3):
        start_time = time.time()
        
        chroma_cpu = torch.matmul(cpu_filterbank, cpu_stft_mag)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    cpu_avg_time = sum(times) / len(times)
    print(f"  CPU average time: {cpu_avg_time:.4f}s")
    print(f"  Speedup ratio (GPU/CPU): {cpu_avg_time/gpu_avg_time:.2f}x")
    
    return {
        "original": avg_time,
        "vectorized": avg_time,
        "precomputed": avg_time,
        "gpu": gpu_avg_time,
        "cpu": cpu_avg_time
    }

def find_sample_audio_file():
    """Find a sample audio file to profile."""
    # Look for audio files in the data directory
    data_dir = "/Users/charlie/projects/gdguess/data/audsnippets-all"
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                return os.path.join(root, file)
    
    return None

def main():
    # Find a sample audio file
    audio_file = find_sample_audio_file()
    if not audio_file:
        print("No audio file found for profiling!")
        return
    
    print(f"Using sample audio file: {audio_file}")
    
    # Initialize device
    device = init_gpu()
    print(f"Using device: {device}")
    
    # Profile chroma implementations
    results = profile_chroma_implementations(audio_file, device)
    
    # Recommend optimizations
    print("\n=== Optimization Recommendations ===")
    fastest_method = min(results, key=results.get)
    print(f"Fastest method: {fastest_method} ({results[fastest_method]:.4f}s)")
    
    if results["precomputed"] < results["original"]:
        speedup = results["original"] / results["precomputed"]
        print(f"Recommendation: Use precomputed filterbank approach for {speedup:.1f}x speedup")
    
    if results["cpu"] < results["gpu"]:
        print("Recommendation: Run chroma extraction on CPU instead of GPU")
    else:
        print("Recommendation: Keep chroma extraction on GPU")
    
    print("\nProfiling complete!")

if __name__ == "__main__":
    main() 