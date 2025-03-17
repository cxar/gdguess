#!/usr/bin/env python3
"""
Optimization script for chroma feature extraction
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

def create_test_audio(duration=15, sr=16000, device=None):
    """Create a test audio signal for benchmarking."""
    # Generate a simple sine wave for testing
    t = torch.arange(0, duration, 1/sr, device=device)
    # Create a mix of frequencies to simulate musical content
    freqs = [440, 880, 1320, 2640]  # A4, A5, E6, E7
    audio = torch.zeros_like(t)
    for i, f in enumerate(freqs):
        audio += torch.sin(2 * torch.pi * f * t) * (0.5 ** i)
    # Normalize
    audio = audio / audio.abs().max()
    return audio

def get_stft(audio, device, n_fft=2048, hop_length=512):
    """Compute STFT for the audio."""
    hann_window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(
        audio, 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=hann_window,
        return_complex=True
    )
    return torch.abs(stft)  # Return magnitude

def original_chroma(stft_magnitude, device, sr=16000, n_chroma=12):
    """Original chroma implementation from turboprocess.py."""
    if stft_magnitude.dim() == 2:
        stft_magnitude = stft_magnitude.unsqueeze(0)  # Add batch dimension if not present
    
    batch_size, freq_bins, time_frames = stft_magnitude.shape
    
    # Create a CQT-like filterbank for chroma
    n_fft = (freq_bins - 1) * 2
    
    # Create frequency bins
    freqs = torch.linspace(0, sr/2, freq_bins).to(device)
    
    # Convert frequencies to chroma bins (logarithmic mapping)
    chroma_map = torch.zeros(n_chroma, freq_bins).to(device)
    
    # Map frequencies to chroma bins
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
    
    # Apply chroma map to STFT magnitudes
    chroma = torch.matmul(chroma_map, stft_magnitude)
    
    # Remove batch dimension if input didn't have it
    if batch_size == 1:
        chroma = chroma.squeeze(0)
    
    return chroma

def optimized_chroma(stft_magnitude, device, sr=16000, n_chroma=12):
    """Optimized chroma implementation with vectorization."""
    if stft_magnitude.dim() == 2:
        stft_magnitude = stft_magnitude.unsqueeze(0)
    
    batch_size, freq_bins, time_frames = stft_magnitude.shape
    
    # Create frequency bins (vectorized)
    freqs = torch.linspace(0, sr/2, freq_bins).to(device)
    
    # Create filter bank (all at once)
    chroma_map = torch.zeros(n_chroma, freq_bins, device=device)
    
    # Compute all MIDI notes at once for frequencies > 0
    valid_mask = freqs > 0
    valid_freqs = freqs[valid_mask]
    
    if len(valid_freqs) > 0:  # Check to avoid empty tensor
        # Vectorized computation of MIDI notes
        midi_notes = 12 * torch.log2(valid_freqs / 440.0) + 69
        
        # Get chroma bins (0-11) with proper handling of negatives
        chroma_bins = torch.fmod(torch.round(midi_notes), 12)
        chroma_bins = torch.where(chroma_bins < 0, chroma_bins + 12, chroma_bins).long()
        
        # Place the frequencies in the correct chroma bins
        for i, (freq_idx, chroma_bin) in enumerate(zip(
                torch.nonzero(valid_mask).squeeze(), chroma_bins)):
            chroma_map[chroma_bin, freq_idx] = 1.0
    
    # Normalize with proper handling of zeros
    chroma_map_sum = torch.sum(chroma_map, dim=1, keepdim=True)
    mask = chroma_map_sum > 0
    chroma_map_sum = torch.where(mask, chroma_map_sum, torch.ones_like(chroma_map_sum))
    chroma_map = chroma_map / chroma_map_sum
    
    # Apply filter bank
    chroma = torch.matmul(chroma_map, stft_magnitude)
    
    if batch_size == 1:
        chroma = chroma.squeeze(0)
    
    return chroma

def precomputed_chroma(stft_magnitude, filter_bank):
    """Fastest chroma implementation using precomputed filter bank."""
    if stft_magnitude.dim() == 2:
        stft_magnitude = stft_magnitude.unsqueeze(0)
    
    batch_size = stft_magnitude.size(0)
    
    # Apply precomputed filter bank 
    chroma = torch.matmul(filter_bank, stft_magnitude)
    
    if batch_size == 1:
        chroma = chroma.squeeze(0)
    
    return chroma

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

def benchmark_implementations(duration=15, sr=16000, n_fft=2048, hop_length=512):
    """Benchmark different chroma implementations."""
    print(f"Benchmarking chroma implementations with {duration}s audio, sr={sr}")
    
    # Initialize device
    device = init_gpu()
    print(f"Using device: {device}")
    
    # Create test audio
    print("Generating test audio...")
    audio = create_test_audio(duration=duration, sr=sr, device=device)
    print(f"Audio shape: {audio.shape}")
    
    # Compute STFT
    print("Computing STFT...")
    stft_mag = get_stft(audio, device, n_fft=n_fft, hop_length=hop_length)
    print(f"STFT magnitude shape: {stft_mag.shape}")
    
    # Precompute chroma filter bank
    print("Creating reusable chroma filter bank...")
    start_time = time.time()
    filter_bank = create_chroma_filter_bank(sr=sr, n_fft=n_fft, device=device)
    precompute_time = time.time() - start_time
    print(f"Filter bank created in {precompute_time:.4f}s")
    
    # Benchmark original implementation
    print("\nBenchmarking original implementation...")
    repeats = 5
    times = []
    for i in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        chroma = original_chroma(stft_mag, device, sr=sr)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
    
    orig_avg = sum(times) / len(times)
    print(f"  Average time: {orig_avg:.4f}s")
    print(f"  Output shape: {chroma.shape}")
    
    # Benchmark optimized implementation
    print("\nBenchmarking optimized implementation...")
    times = []
    for i in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        chroma = optimized_chroma(stft_mag, device, sr=sr)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
    
    opt_avg = sum(times) / len(times)
    print(f"  Average time: {opt_avg:.4f}s")
    print(f"  Output shape: {chroma.shape}")
    print(f"  Speedup vs original: {orig_avg / opt_avg:.2f}x")
    
    # Benchmark precomputed implementation
    print("\nBenchmarking precomputed implementation...")
    times = []
    for i in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        chroma = precomputed_chroma(stft_mag, filter_bank)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
    
    precomp_avg = sum(times) / len(times)
    print(f"  Average time: {precomp_avg:.4f}s")
    print(f"  Output shape: {chroma.shape}")
    print(f"  Speedup vs original: {orig_avg / precomp_avg:.2f}x")
    
    # Compare with CPU implementation
    print("\nComparing GPU vs CPU performance...")
    cpu_device = torch.device("cpu")
    cpu_stft_mag = stft_mag.cpu()
    cpu_filter_bank = filter_bank.cpu()
    
    # Warm up
    precomputed_chroma(cpu_stft_mag, cpu_filter_bank)
    
    # Benchmark CPU
    times = []
    for i in range(repeats):
        start_time = time.time()
        
        chroma = precomputed_chroma(cpu_stft_mag, cpu_filter_bank)
        
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  CPU Run {i+1}: {times[-1]:.4f}s")
    
    cpu_avg = sum(times) / len(times)
    print(f"  CPU Average time: {cpu_avg:.4f}s")
    print(f"  GPU vs CPU speedup: {cpu_avg / precomp_avg:.2f}x")
    
    # Check correctness
    print("\nVerifying correctness of implementations...")
    original_output = original_chroma(stft_mag, device, sr=sr)
    optimized_output = optimized_chroma(stft_mag, device, sr=sr)
    precomputed_output = precomputed_chroma(stft_mag, filter_bank)
    
    # Check if results are similar
    diff_opt = torch.abs(original_output - optimized_output).max().item()
    diff_precomp = torch.abs(original_output - precomputed_output).max().item()
    
    print(f"  Max difference (original vs optimized): {diff_opt:.6f}")
    print(f"  Max difference (original vs precomputed): {diff_precomp:.6f}")
    
    # Summary
    print("\n=== Performance Summary ===")
    print(f"Original implementation: {orig_avg:.4f}s")
    print(f"Optimized implementation: {opt_avg:.4f}s (speedup: {orig_avg / opt_avg:.2f}x)")
    print(f"Precomputed implementation: {precomp_avg:.4f}s (speedup: {orig_avg / precomp_avg:.2f}x)")
    print(f"CPU implementation: {cpu_avg:.4f}s")
    
    fastest = min(orig_avg, opt_avg, precomp_avg, cpu_avg)
    if fastest == orig_avg:
        print("Original implementation is fastest (unexpected)")
    elif fastest == opt_avg:
        print("Optimized implementation is fastest")
    elif fastest == precomp_avg:
        print("Precomputed GPU implementation is fastest")
    else:
        print("CPU implementation is fastest")
    
    return {
        "original": orig_avg,
        "optimized": opt_avg,
        "precomputed": precomp_avg,
        "cpu": cpu_avg
    }

def main():
    """Run benchmarks for different audio durations."""
    results_by_duration = {}
    for duration in [5, 15, 30]:
        print(f"\n\n{'='*40}")
        print(f"Testing with {duration}s audio")
        print(f"{'='*40}\n")
        results_by_duration[duration] = benchmark_implementations(duration=duration)
    
    # Print final summary
    print("\n\n=== Final Summary ===")
    print("Average times (seconds) by duration:")
    print(f"{'Duration (s)':<12} {'Original':<10} {'Optimized':<10} {'Precomputed':<12} {'CPU':<10} {'Best Method':<15}")
    print(f"{'-'*65}")
    
    for duration, results in results_by_duration.items():
        best_method = min(results, key=results.get)
        best_time = results[best_method]
        print(f"{duration:<12} {results['original']:<10.4f} {results['optimized']:<10.4f} {results['precomputed']:<12.4f} {results['cpu']:<10.4f} {best_method:<15}")
    
    print("\nRecommendations:")
    if all(res["precomputed"] < res["original"] for res in results_by_duration.values()):
        avg_speedup = sum(res["original"] / res["precomputed"] for res in results_by_duration.values()) / len(results_by_duration)
        print(f"1. Use the precomputed filter bank approach for ~{avg_speedup:.1f}x speedup")
    
    if all(res["cpu"] < res["precomputed"] for res in results_by_duration.values()):
        print("2. Move chroma computation to CPU instead of GPU")
    else:
        print("2. Keep chroma computation on GPU with precomputed filter banks")
    
    print("\nThese optimizations would significantly reduce the bottleneck in the turboprocess.py script.")

if __name__ == "__main__":
    main() 