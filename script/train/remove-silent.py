#!/usr/bin/env python3
"""
Silent Audio File Removal Script for Grateful Dead Dataset

This script scans a directory of audio files, identifies silent files,
and either deletes them or moves them to a separate directory.

Usage:
    python remove_silent_files.py --input_dir=./data/audsnippets-all --action=move

Parameters:
    --input_dir: Directory containing the audio files
    --action: Action to take for silent files ('delete' or 'move')
    --move_dir: Directory to move silent files to (if action is 'move')
    --silence_threshold: RMS threshold below which a file is considered silent
    --min_duration: Minimum valid audio duration in seconds
    --sample_rate: Sample rate for audio processing
    --log_file: Path to log file
    --verbose: Whether to print detailed progress
"""

import argparse
import datetime
import logging
import os
import shutil
import sys
import time
from multiprocessing import Pool, cpu_count

import librosa
import numpy as np
from tqdm import tqdm


def setup_logging(log_file):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def is_silent(file_path, silence_threshold=0.01, min_duration=1.0, sample_rate=24000):
    """
    Check if an audio file is silent or too short.

    Args:
        file_path: Path to the audio file
        silence_threshold: RMS threshold for silence detection
        min_duration: Minimum valid duration in seconds
        sample_rate: Sample rate for audio processing

    Returns:
        dict: {'silent': bool, 'reason': str or None, 'rms': float or None}
    """
    try:
        # Load first few seconds of audio to check
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=5)

        # Check duration
        duration = len(y) / sr
        if duration < min_duration:
            return {"silent": True, "reason": f"too_short_{duration:.2f}s", "rms": None}

        # Check if silent (using RMS energy)
        rms = np.sqrt(np.mean(y**2))
        if rms < silence_threshold:
            return {"silent": True, "reason": "low_rms", "rms": rms}

        return {"silent": False, "reason": None, "rms": rms}

    except Exception as e:
        return {"silent": True, "reason": f"error_{str(e)}", "rms": None}


def process_file(args):
    """Process a single file (for multiprocessing)."""
    file_path, config = args
    result = is_silent(
        file_path,
        silence_threshold=config["silence_threshold"],
        min_duration=config["min_duration"],
        sample_rate=config["sample_rate"],
    )

    return {
        "file_path": file_path,
        "silent": result["silent"],
        "reason": result["reason"],
        "rms": result["rms"],
    }


def find_silent_files(input_dir, config, logger):
    """
    Find all silent files in the dataset.

    Args:
        input_dir: Root directory containing audio files
        config: Configuration dict
        logger: Logger instance

    Returns:
        list: List of dicts with silent file info
    """
    logger.info(f"Scanning directory: {input_dir}")

    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac")):
                all_files.append(os.path.join(root, file))

    logger.info(f"Found {len(all_files)} audio files to check")

    # Process files in parallel
    silent_files = []
    processed_count = 0

    # Determine number of workers (use fewer for very large datasets to avoid memory issues)
    num_workers = min(cpu_count(), 8)  # Limit to max 8 workers

    with Pool(processes=num_workers) as pool:
        args_list = [(file_path, config) for file_path in all_files]

        # Process files in batches to show progress
        batch_size = 1000
        for i in range(0, len(args_list), batch_size):
            batch_args = args_list[i : i + batch_size]

            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(args_list) + batch_size - 1)//batch_size}"
            )

            results = list(
                tqdm(
                    pool.imap(process_file, batch_args),
                    total=len(batch_args),
                    desc="Checking files",
                    disable=not config["verbose"],
                )
            )

            # Count and collect silent files
            batch_silent = [r for r in results if r["silent"]]
            silent_files.extend(batch_silent)
            processed_count += len(batch_args)

            logger.info(
                f"Processed {processed_count}/{len(all_files)} files, "
                f"found {len(batch_silent)} silent files in this batch"
            )

    logger.info(
        f"Completed scan. Found {len(silent_files)} silent files out of {len(all_files)} total."
    )

    # Group silent files by reason
    reasons = {}
    for file in silent_files:
        reason = file["reason"]
        reasons[reason] = reasons.get(reason, 0) + 1

    logger.info("Silent files by reason:")
    for reason, count in reasons.items():
        logger.info(f"  {reason}: {count}")

    return silent_files


def handle_silent_files(silent_files, config, logger):
    """
    Process silent files according to specified action.

    Args:
        silent_files: List of silent file info dicts
        config: Configuration dictionary
        logger: Logger instance
    """
    action = config["action"]
    move_dir = config["move_dir"]

    if action == "move" and move_dir:
        # Create target directory if it doesn't exist
        os.makedirs(move_dir, exist_ok=True)

        # Create subdirectories for different silence reasons
        for file in silent_files:
            reason = file["reason"].split("_")[0]  # Get the main reason category
            reason_dir = os.path.join(move_dir, reason)
            os.makedirs(reason_dir, exist_ok=True)

    success_count = 0
    failed_count = 0

    for file_info in tqdm(
        silent_files,
        desc=f"{action.capitalize()}ing silent files",
        disable=not config["verbose"],
    ):
        file_path = file_info["file_path"]
        reason = file_info["reason"].split("_")[0]  # Get the main reason category

        try:
            if action == "delete":
                os.remove(file_path)
                logger.debug(f"Deleted: {file_path} (Reason: {file_info['reason']})")

            elif action == "move" and move_dir:
                # Preserve directory structure relative to input_dir
                rel_path = os.path.relpath(file_path, config["input_dir"])
                # Use just the filename for the target to avoid recreating the full structure
                filename = os.path.basename(file_path)
                # Add a timestamp to avoid name collisions
                name, ext = os.path.splitext(filename)
                timestamp = (
                    int(time.time() * 1000) % 10000
                )  # Use last 4 digits of timestamp
                new_filename = f"{name}_{timestamp}{ext}"

                target_path = os.path.join(move_dir, reason, new_filename)
                shutil.move(file_path, target_path)
                logger.debug(
                    f"Moved: {file_path} -> {target_path} (Reason: {file_info['reason']})"
                )

            success_count += 1

        except Exception as e:
            logger.error(f"Failed to {action} {file_path}: {str(e)}")
            failed_count += 1

    logger.info(f"Successfully {action}d {success_count} files")
    if failed_count > 0:
        logger.warning(f"Failed to {action} {failed_count} files")


def save_report(silent_files, config, logger):
    """Save a CSV report of all silent files."""
    report_path = os.path.join(
        os.path.dirname(config["log_file"]), "silent_files_report.csv"
    )

    with open(report_path, "w") as f:
        f.write("file_path,reason,rms\n")
        for file_info in silent_files:
            rms_value = file_info["rms"] if file_info["rms"] is not None else "NA"
            f.write(f"{file_info['file_path']},{file_info['reason']},{rms_value}\n")

    logger.info(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Find and remove silent audio files")

    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["delete", "move", "report_only"],
        default="report_only",
        help="Action to take for silent files",
    )
    parser.add_argument(
        "--move_dir",
        type=str,
        default="./silent_files",
        help='Directory to move silent files to (if action is "move")',
    )
    parser.add_argument(
        "--silence_threshold",
        type=float,
        default=0.01,
        help="RMS threshold for silence detection",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum valid audio duration in seconds",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Sample rate for audio processing",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for log files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )

    args = parser.parse_args()

    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)

    # Generate timestamp for log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"silent_removal_{timestamp}.log")

    # Set up logging
    logger = setup_logging(log_file)

    # Collect configuration
    config = {
        "input_dir": args.input_dir,
        "action": args.action,
        "move_dir": args.move_dir,
        "silence_threshold": args.silence_threshold,
        "min_duration": args.min_duration,
        "sample_rate": args.sample_rate,
        "log_file": log_file,
        "verbose": args.verbose,
    }

    # Log configuration
    logger.info("Starting silent file removal with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Find silent files
    start_time = time.time()
    silent_files = find_silent_files(args.input_dir, config, logger)

    # Save report
    save_report(silent_files, config, logger)

    # Handle files based on action
    if args.action != "report_only":
        handle_silent_files(silent_files, config, logger)
    else:
        logger.info("Report only mode - no files were modified")

    # Log completion time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    main()
