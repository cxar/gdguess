#!/usr/bin/env python3
"""
Fast, multi-processed preprocessing script for the Grateful Dead show dating model.
This version is optimized for speed on Apple Silicon (M4 Max).
"""

import argparse
import multiprocessing
import concurrent.futures
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from tqdm import tqdm


def parse_arguments():
    """Parse command-line arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(description="Fast preprocessing for Grateful Dead audio snippets")
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
        default=32,
        help="Batch size for file processing (affects memory usage)",
    )
    parser.add_argument(
        "--reduced-features",
        action="store_true",
        help="Extract fewer features for faster processing and smaller files",
    )
    return parser.parse_args()


def find_audio_files(root_dir, limit=0):
    """Find audio files with valid dates in the directory tree."""
    import datetime
    import re
    
    valid_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    print(f"Scanning for audio files in: {root_dir}")
    
    # First pass: find all audio files with valid dates
    file_count = 0
    for root, _, files in os.walk(root_dir):
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
                    base_date = datetime.date(1968, 1, 1)  # Same as in config
                    days = (date_obj - base_date).days
                    
                    audio_files.append({
                        "file": file_path,
                        "date": date_obj,
                        "days": days,
                        "era": era,
                        "year": year
                    })
                    
                    file_count += 1
                    if file_count % 10000 == 0:
                        print(f"Found {file_count} audio files...")
                    
                    # Apply limit if specified
                    if limit > 0 and file_count >= limit:
                        break
                        
                except ValueError:
                    continue
        
        # Apply limit if specified
        if limit > 0 and file_count >= limit:
            break
    
    print(f"Found {len(audio_files)} audio files with valid dates")
    return audio_files


def process_audio_file(file_info, target_sr=24000, reduced_features=False):
    """Process a single audio file and extract features."""
    try:
        # Load audio with librosa
        audio, _ = librosa.load(file_info["file"], sr=target_sr, mono=True)
        
        # Standardize length to 15 seconds
        desired_length = target_sr * 15
        if len(audio) < desired_length:
            # Pad shorter audio
            padding = desired_length - len(audio)
            audio = np.pad(audio, (0, padding))
        else:
            # Trim longer audio
            audio = audio[:desired_length]
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Create mel spectrogram transform
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000,
            center=True,
            norm='slaney',
            mel_scale='htk'
        )
        
        # Process audio - optimized to only compute what's needed
        # Harmonic-percussive source separation (most computationally expensive step)
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Process harmonic component
        harmonic_tensor = torch.from_numpy(harmonic).float()
        harmonic_mel = mel_spec(harmonic_tensor)
        harmonic_mel = torch.log(harmonic_mel + 1e-4)
        harmonic_mel = harmonic_mel.unsqueeze(0)  # Add channel dimension
        
        # Process percussive component
        percussive_tensor = torch.from_numpy(percussive).float()
        percussive_mel = mel_spec(percussive_tensor)
        percussive_mel = torch.log(percussive_mel + 1e-4)
        percussive_mel = percussive_mel.unsqueeze(0)
        
        # Create result with essential features
        result = {
            # Metadata
            "label": torch.tensor(file_info["days"], dtype=torch.float),
            "era": torch.tensor(file_info["era"], dtype=torch.long),
            "file": file_info["file"],
            
            # Primary features
            "mel_spec": harmonic_mel.half(),
            "mel_spec_percussive": percussive_mel.half(),
        }
        
        # Add additional features if not in reduced mode
        if not reduced_features:
            # Compute spectral contrast (harmonic only)
            harmonic_contrast = librosa.feature.spectral_contrast(
                y=harmonic, sr=target_sr, n_fft=2048, hop_length=512
            )
            
            # Compute chroma from harmonic component
            chroma = librosa.feature.chroma_stft(
                y=harmonic, sr=target_sr, n_fft=2048, hop_length=512
            )
            
            # Add to result
            result["spectral_contrast_harmonic"] = torch.from_numpy(harmonic_contrast).half()
            result["chroma"] = torch.from_numpy(chroma).half()
            
            # Add onset envelope from percussive component
            onset_env = librosa.onset.onset_strength(
                y=percussive, sr=target_sr, hop_length=512
            )
            result["onset_env"] = torch.from_numpy(onset_env).half()
        
        return result, None
    
    except Exception as e:
        return None, f"Error processing {file_info['file']}: {str(e)}"


def process_batch(batch_id, file_batch, output_dir, target_sr, reduced_features):
    """Process a batch of files in a single worker."""
    results = []
    
    for i, file_info in enumerate(file_batch):
        result, error = process_audio_file(file_info, target_sr, reduced_features)
        
        if result:
            # Save the preprocessed data
            file_idx = batch_id * len(file_batch) + i
            output_path = os.path.join(output_dir, f"{file_idx:06d}.pt")
            torch.save(result, output_path)
            results.append((file_idx, True))
        else:
            results.append((file_idx, error))
    
    return results


def main():
    """Main preprocessing function."""
    args = parse_arguments()
    start_time = time.time()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.input_dir, "preprocessed")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all valid audio files
    audio_files = find_audio_files(args.input_dir, args.limit)
    
    if not audio_files:
        print("No audio files found with valid dates. Exiting.")
        return
    
    # Determine optimal number of workers
    if args.workers <= 0:
        num_cores = multiprocessing.cpu_count()
        # For M4 Max or similar high-core-count machines
        num_workers = min(16, num_cores) 
    else:
        num_workers = args.workers
    
    print(f"Using {num_workers} parallel workers")
    
    # Split files into batches for parallel processing
    batch_size = args.batch_size
    batches = [audio_files[i:i + batch_size] for i in range(0, len(audio_files), batch_size)]
    print(f"Processing {len(audio_files)} files in {len(batches)} batches of up to {batch_size} files each")
    
    # Process in parallel
    successful = 0
    failed = 0
    
    with tqdm(total=len(audio_files), desc="Preprocessing") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            futures = []
            for batch_id, batch in enumerate(batches):
                future = executor.submit(
                    process_batch, 
                    batch_id, 
                    batch, 
                    output_dir, 
                    24000,  # target_sr 
                    args.reduced_features
                )
                futures.append(future)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    for _, status in batch_results:
                        if status is True:
                            successful += 1
                        else:
                            failed += 1
                    pbar.update(len(batch_results))
                except Exception as e:
                    print(f"Batch processing error: {e}")
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
            total_files = 974145  # From your earlier output
            estimated_total_time = total_files / throughput
            est_hours, remainder = divmod(estimated_total_time, 3600)
            est_minutes, _ = divmod(remainder, 60)
            print(f"Estimated time for full dataset: {int(est_hours)}h {int(est_minutes)}m")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0) 