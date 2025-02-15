import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from pydub import AudioSegment
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "../../data/fullshows-all"  # Path to the source directory containing MP3s
DEST_DIR = "../../data/audsnippets-all"  # Path to the destination directory
SNIPPET_LENGTH_MS = 15 * 1000  # Snippet length: 15 seconds (in milliseconds)
TARGET_SR = 24000  # Target sample rate (24KHz)


def process_file(source_file, source_dir, dest_dir):
    """
    Process a single MP3 file: split the entire audio into consecutive 15-second snippets,
    resampling the audio to TARGET_SR before splitting, and save them to the destination folder
    preserving the same folder structure.
    Files are renamed as <base_name>_<date>_snippet_<index>.<ext>, where <date> is extracted from
    the file name in the format YYYY-MM-DD.
    """
    try:
        # Compute the relative path from the source directory and create the destination folder.
        rel_path = os.path.relpath(os.path.dirname(source_file), source_dir)
        dest_folder = os.path.join(dest_dir, rel_path)
        os.makedirs(dest_folder, exist_ok=True)

        # Get the base filename and extension.
        base_name, ext = os.path.splitext(os.path.basename(source_file))

        # Extract the date from the filename using regex.
        match = re.search(r"(\d{4}-\d{2}-\d{2})", base_name)
        date_str = match.group(1) if match else "unknown_date"

        # Load the audio using pydub.
        audio = AudioSegment.from_mp3(source_file)
        # Resample the audio to TARGET_SR.
        audio = audio.set_frame_rate(TARGET_SR)
        duration = len(audio)  # Duration in milliseconds.

        # Calculate how many 15-second snippets can be extracted.
        snippet_count = duration // SNIPPET_LENGTH_MS
        if snippet_count == 0:
            snippet_count = (
                1  # If the file is shorter than 15 seconds, output one snippet.
            )

        for i in range(snippet_count):
            start = i * SNIPPET_LENGTH_MS
            end = start + SNIPPET_LENGTH_MS
            snippet = audio[start:end]
            # Build the new filename with the extracted date and snippet index.
            new_filename = f"{base_name}_{date_str}_snippet_{i+1}{ext}"
            dest_file = os.path.join(dest_folder, new_filename)
            snippet.export(dest_file, format="mp3")
        return True
    except Exception as e:
        print(f"Error processing {source_file}: {e}")
        return False


def get_all_mp3_files(source_dir):
    """
    Recursively collect all MP3 file paths under source_dir.
    """
    file_list = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".mp3"):
                file_list.append(os.path.join(root, file))
    return file_list


def main():
    # Gather all MP3 files from the source directory.
    mp3_files = get_all_mp3_files(SOURCE_DIR)
    total_files = len(mp3_files)
    print(f"Found {total_files} MP3 file(s) to process.")

    # Use ProcessPoolExecutor to parallelize processing of files.
    max_workers = 16  # Adjust based on your machine's capabilities.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, SOURCE_DIR, DEST_DIR): f for f in mp3_files
        }
        # Display progress using tqdm.
        for future in tqdm(
            as_completed(futures), total=total_files, desc="Processing Files"
        ):
            future.result()  # Optionally check the result.


if __name__ == "__main__":
    main()
