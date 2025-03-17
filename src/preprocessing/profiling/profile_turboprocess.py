#!/usr/bin/env python3
"""
Profiling script for turboprocess.py to identify bottlenecks
"""

import cProfile
import pstats
import io
import os
import sys
import time
import torch
import torchaudio
import torch.nn.functional as F
from ..turboprocess import (
    gpu_hpss, 
    gpu_spectral_contrast, 
    gpu_chroma, 
    gpu_onset_strength, 
    get_feature_extractor,
    init_gpu
)

def profile_feature_extraction(audio_file, clip_length=15, optimize_level=1, high_memory=True):
    """Profile the feature extraction process for a single file."""
    print(f"Profiling feature extraction for file: {audio_file}")
    
    # Initialize device
    device = init_gpu()
    print(f"Using device: {device}")
    
    # Load audio file
    print("Loading audio file...")
    start_time = time.time()
    audio, sr = torchaudio.load(audio_file)
    load_time = time.time() - start_time
    print(f"Audio loaded in {load_time:.4f}s, shape: {audio.shape}, sample rate: {sr}")
    
    # Create feature extractor
    print("Creating feature extractor...")
    feature_extractor = get_feature_extractor(
        device, target_sr=16000, ultrafast=False, 
        optimize_level=optimize_level, high_memory=high_memory
    )
    
    # Process audio
    print("Processing audio...")
    start_time = time.time()
    features, error = feature_extractor(audio_file, 16000, clip_length)
    process_time = time.time() - start_time
    print(f"Audio processed in {process_time:.4f}s")
    
    if features:
        print("Features extracted successfully:")
        for name, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            else:
                print(f"  - {name}: {type(tensor)}")
    else:
        print(f"Error extracting features: {error}")
    
    return features, process_time

def profile_hpss(audio_file, device, clip_length=15):
    """Profile just the HPSS process."""
    print(f"Profiling HPSS for file: {audio_file}")
    
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
    
    # Run HPSS
    print("Running HPSS...")
    start_time = time.time()
    harmonic, percussive = gpu_hpss(audio, device)
    hpss_time = time.time() - start_time
    print(f"HPSS completed in {hpss_time:.4f}s")
    
    return hpss_time

def profile_all_components(audio_file, device, clip_length=15):
    """Profile each component of the feature extraction process."""
    print(f"Profiling all components for file: {audio_file}")
    
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
    
    # Initialize components
    n_fft = 2048
    hop_length = 512
    hann_window = torch.hann_window(n_fft).to(device)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        f_min=20,
        f_max=8000,
        center=True,
        norm='slaney',
        mel_scale='htk'
    ).to(device)
    
    # Run and time each component
    results = {}
    
    # HPSS
    start_time = time.time()
    harmonic, percussive = gpu_hpss(audio, device)
    results['hpss_time'] = time.time() - start_time
    print(f"HPSS completed in {results['hpss_time']:.4f}s")
    
    # Mel spectrogram - harmonic
    start_time = time.time()
    harmonic_mel = mel_spec(harmonic)
    harmonic_mel = torch.log(harmonic_mel + 1e-5)
    results['harmonic_mel_time'] = time.time() - start_time
    print(f"Harmonic mel spectrogram completed in {results['harmonic_mel_time']:.4f}s")
    
    # Mel spectrogram - percussive
    start_time = time.time()
    percussive_mel = mel_spec(percussive)
    percussive_mel = torch.log(percussive_mel + 1e-5)
    results['percussive_mel_time'] = time.time() - start_time
    print(f"Percussive mel spectrogram completed in {results['percussive_mel_time']:.4f}s")
    
    # STFT
    start_time = time.time()
    stft = torch.stft(
        harmonic, 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=hann_window,
        return_complex=True
    )
    stft_mag = torch.abs(stft)
    results['stft_time'] = time.time() - start_time
    print(f"STFT completed in {results['stft_time']:.4f}s")
    
    # Spectral contrast
    start_time = time.time()
    spectral_contrast = gpu_spectral_contrast(stft_mag, device)
    results['spectral_contrast_time'] = time.time() - start_time
    print(f"Spectral contrast completed in {results['spectral_contrast_time']:.4f}s")
    
    # Chroma
    start_time = time.time()
    chroma = gpu_chroma(stft_mag, device, sr=sr)
    results['chroma_time'] = time.time() - start_time
    print(f"Chroma completed in {results['chroma_time']:.4f}s")
    
    # Onset envelope
    start_time = time.time()
    p_stft = torch.stft(
        percussive, 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=hann_window,
        return_complex=True
    )
    p_stft_mag = torch.abs(p_stft)
    onset_env = gpu_onset_strength(p_stft_mag, device)
    results['onset_env_time'] = time.time() - start_time
    print(f"Onset envelope completed in {results['onset_env_time']:.4f}s")
    
    # Move to CPU
    start_time = time.time()
    features = {
        "mel_spec": harmonic_mel.cpu().half(),
        "mel_spec_percussive": percussive_mel.cpu().half(),
        "spectral_contrast_harmonic": spectral_contrast.cpu().half(),
        "chroma": chroma.cpu().half(),
        "onset_env": onset_env.cpu().half()
    }
    results['to_cpu_time'] = time.time() - start_time
    print(f"Move to CPU completed in {results['to_cpu_time']:.4f}s")
    
    return results

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
    
    # Run profiling
    print("\n=== Profiling full feature extraction ===")
    features, extract_time = profile_feature_extraction(audio_file)
    
    print("\n=== Profiling HPSS component ===")
    hpss_time = profile_hpss(audio_file, device)
    
    print("\n=== Profiling all components separately ===")
    component_times = profile_all_components(audio_file, device)
    
    # Summarize results
    print("\n=== Profiling Summary ===")
    print(f"Total feature extraction time: {extract_time:.4f}s")
    print(f"HPSS time: {hpss_time:.4f}s ({hpss_time/extract_time*100:.1f}% of total)")
    
    total_component_time = sum(component_times.values())
    print(f"Component breakdown:")
    for component, time in component_times.items():
        print(f"  - {component}: {time:.4f}s ({time/total_component_time*100:.1f}% of component time)")
    
    # Check for CPU vs GPU performance
    try:
        # Try to move component to CPU and compare
        audio, sr = torchaudio.load(audio_file)
        audio = audio.squeeze(0)[:sr*15]  # 15 seconds
        
        # GPU timing
        audio_gpu = audio.to(device)
        start_time = time.time()
        harmonic_gpu, percussive_gpu = gpu_hpss(audio_gpu, device)
        gpu_time = time.time() - start_time
        
        # CPU timing
        audio_cpu = audio.cpu()
        device_cpu = torch.device("cpu")
        start_time = time.time()
        harmonic_cpu, percussive_cpu = gpu_hpss(audio_cpu, device_cpu)
        cpu_time = time.time() - start_time
        
        print(f"\nGPU vs CPU performance for HPSS:")
        print(f"  - GPU: {gpu_time:.4f}s")
        print(f"  - CPU: {cpu_time:.4f}s")
        print(f"  - Speedup: {cpu_time/gpu_time:.2f}x")
    except Exception as e:
        print(f"Error comparing CPU vs GPU performance: {e}")
    
    print("\nProfiling complete!")

if __name__ == "__main__":
    main() 