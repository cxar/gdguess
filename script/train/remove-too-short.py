#!/usr/bin/env python3
"""
Exact Length Audio Validation Script for Grateful Dead Dataset

This script scans a directory of audio files and:
1. Identifies files that aren't EXACTLY 15 seconds in length (zero tolerance)
2. Either deletes them or moves them to a separate directory

Usage:
    python exact_length_validation.py --input_dir=./data/audsnippets-all --action=move

Parameters:
    --input_dir: Directory containing the audio files
    --action: Action to take for invalid files ('delete', 'move', or 'report_only')
    --move_dir: Directory to move invalid files to (if action is 'move')
    --target_duration: Target duration in seconds (default: 15.0)
    --tolerance: Tolerance in seconds (default: 0.0 for exact matching)
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


def validate_length(file_path, config):
    """
    Validate an audio file for exact length.

    Args:
        file_path: Path to the audio file
        config: Dictionary containing validation parameters

    Returns:
        dict: Validation result with status and reason
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Check duration
        duration = len(y) / sr
        target_duration = config["target_duration"]
        tolerance = config["tolerance"]

        if abs(duration - target_duration) > tolerance:
            return {
                "valid": False,
                "reason": f"wrong_duration_{duration:.6f}s",
                "duration": duration,
            }

        # If it passes the check, it's valid
        return {"valid": True, "reason": None, "duration": duration}

    except Exception as e:
        return {"valid": False, "reason": f"error_{str(e)}", "duration": None}


def process_file(args):
    """Process a single file (for multiprocessing)."""
    file_path, config = args
    result = validate_length(file_path, config)

    return {
        "file_path": file_path,
        "valid": result["valid"],
        "reason": result["reason"],
        "duration": result["duration"],
    }


def find_invalid_length_files(input_dir, config, logger):
    """
    Find all files that don't match the exact length requirement.

    Args:
        input_dir: Root directory containing audio files
        config: Configuration dict
        logger: Logger instance

    Returns:
        list: List of dicts with invalid file info
    """
    logger.info(f"Scanning directory: {input_dir}")

    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac")):
                all_files.append(os.path.join(root, file))

    logger.info(f"Found {len(all_files)} audio files to check")

    # Process files in parallel
    invalid_files = []
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

            # Count and collect invalid files
            batch_invalid = [r for r in results if not r["valid"]]
            invalid_files.extend(batch_invalid)
            processed_count += len(batch_args)

            logger.info(
                f"Processed {processed_count}/{len(all_files)} files, "
                f"found {len(batch_invalid)} incorrect length files in this batch"
            )

    logger.info(
        f"Completed scan. Found {len(invalid_files)} incorrect length files out of {len(all_files)} total."
    )

    # Group invalid files by reason type
    too_short = 0
    too_long = 0
    errors = 0

    for file in invalid_files:
        if file["reason"] and file["reason"].startswith("wrong_duration"):
            if file["duration"] and file["duration"] < config["target_duration"]:
                too_short += 1
            else:
                too_long += 1
        else:
            errors += 1

    logger.info("Invalid files breakdown:")
    logger.info(f"  Too short: {too_short}")
    logger.info(f"  Too long: {too_long}")
    logger.info(f"  Errors: {errors}")

    return invalid_files


def handle_invalid_files(invalid_files, config, logger):
    """
    Process invalid files according to specified action.

    Args:
        invalid_files: List of invalid file info dicts
        config: Configuration dictionary
        logger: Logger instance
    """
    action = config["action"]
    move_dir = config["move_dir"]

    if action == "move" and move_dir:
        # Create target directory if it doesn't exist
        os.makedirs(move_dir, exist_ok=True)

        # Create subdirectories for too long/too short
        too_short_dir = os.path.join(move_dir, "too_short")
        too_long_dir = os.path.join(move_dir, "too_long")
        error_dir = os.path.join(move_dir, "error")

        os.makedirs(too_short_dir, exist_ok=True)
        os.makedirs(too_long_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)

    success_count = 0
    failed_count = 0

    for file_info in tqdm(
        invalid_files,
        desc=f"{action.capitalize()}ing invalid files",
        disable=not config["verbose"],
    ):
        file_path = file_info["file_path"]

        # Determine the appropriate category
        if file_info["reason"] and file_info["reason"].startswith("wrong_duration"):
            if (
                file_info["duration"]
                and file_info["duration"] < config["target_duration"]
            ):
                category = "too_short"
            else:
                category = "too_long"
        else:
            category = "error"

        try:
            if action == "delete":
                os.remove(file_path)
                logger.debug(f"Deleted: {file_path} (Reason: {file_info['reason']})")

            elif action == "move" and move_dir:
                # Use just the filename to avoid recreating directory structure
                filename = os.path.basename(file_path)
                # Add a timestamp to avoid name collisions
                name, ext = os.path.splitext(filename)
                timestamp = (
                    int(time.time() * 1000) % 10000
                )  # Use last 4 digits of timestamp
                new_filename = f"{name}_{timestamp}{ext}"

                # Target directory based on category
                if category == "too_short":
                    target_dir = too_short_dir
                elif category == "too_long":
                    target_dir = too_long_dir
                else:
                    target_dir = error_dir

                target_path = os.path.join(target_dir, new_filename)
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


def save_report(invalid_files, config, logger):
    """Save a CSV report of all invalid files."""
    report_path = os.path.join(
        os.path.dirname(config["log_file"]), "length_validation_report.csv"
    )

    with open(report_path, "w") as f:
        f.write("file_path,reason,duration\n")
        for file_info in invalid_files:
            duration_value = (
                file_info["duration"] if file_info["duration"] is not None else "NA"
            )
            f.write(
                f"{file_info['file_path']},{file_info['reason']},{duration_value}\n"
            )

    logger.info(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate audio files for exact length"
    )

    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["delete", "move", "report_only"],
        default="report_only",
        help="Action to take for invalid files",
    )
    parser.add_argument(
        "--move_dir",
        type=str,
        default="./incorrect_length_files",
        help='Directory to move invalid files to (if action is "move")',
    )
    parser.add_argument(
        "--target_duration",
        type=float,
        default=15.0,
        help="Target duration in seconds for valid files",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Tolerance in seconds (0.0 for exact match)",
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
    log_file = os.path.join(args.log_dir, f"length_validation_{timestamp}.log")

    # Set up logging
    logger = setup_logging(log_file)

    # Collect configuration
    config = {
        "input_dir": args.input_dir,
        "action": args.action,
        "move_dir": args.move_dir,
        "target_duration": args.target_duration,
        "tolerance": args.tolerance,
        "sample_rate": args.sample_rate,
        "log_file": log_file,
        "verbose": args.verbose,
    }

    # Log configuration
    logger.info("Starting exact length validation with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    if config["tolerance"] == 0.0:
        logger.info(
            "Running with ZERO tolerance - only exactly 15.000000s files will be kept"
        )

    # Find invalid files
    start_time = time.time()
    invalid_files = find_invalid_length_files(args.input_dir, config, logger)

    # Save report
    save_report(invalid_files, config, logger)

    # Handle files based on action
    if args.action != "report_only":
        handle_invalid_files(invalid_files, config, logger)
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
