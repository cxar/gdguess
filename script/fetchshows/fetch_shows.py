import os
import re
import time
from datetime import datetime, timedelta

import requests

# --- Configuration ---
START_DATE_STR = "1968-01-01"  # default start date if no resume date is found
END_DATE_STR = "1995-12-31"  # end date in YYYY-MM-DD format
OUTPUT_DIR = "../../data/fullshows-all"  # where files are saved
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

DATE_REGEX = re.compile(r"(\d{4}-\d{2}-\d{2})_")  # to extract date from filename


def get_resume_date(output_dir):
    """
    Look through OUTPUT_DIR (including subfolders) for filenames that start with a date in YYYY-MM-DD format,
    then return the maximum date plus one day. If none found, return None.
    """
    max_date = None
    # Walk through all files in output_dir
    for root, _, files in os.walk(output_dir):
        for file in files:
            m = DATE_REGEX.search(file)
            if m:
                date_str = m.group(1)
                try:
                    cur_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue  # skip files with invalid dates
                if (max_date is None) or (cur_date > max_date):
                    max_date = cur_date
    if max_date:
        return max_date + timedelta(days=1)
    else:
        return None


def advanced_search_item_by_date(
    search_date, max_retries=MAX_RETRIES, delay=RETRY_DELAY
):
    """
    Uses Archive.org's Advanced Search endpoint to find an audio item for the specified date.
    Returns the top item (sorted by favorites descending) or None if no item is found.
    """
    base_url = "https://archive.org/advancedsearch.php"
    params = {
        "q": f"date:{search_date} AND collection:(GratefulDead) AND NOT collection:(stream_only)",
        "fl[]": ["identifier", "title", "year", "num_favorites"],
        "sort[]": "num_favorites desc",
        "rows": "50",  # up to 50 items
        "page": "1",
        "output": "json",
    }
    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"Searching for audio recordings for {search_date} (attempt {attempt})..."
            )
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            if docs:
                return docs[0]
            else:
                print(f"No items found for date {search_date}.")
                return None
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503:
                print(
                    f"503 error for {search_date} on attempt {attempt}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                print(f"HTTP error {response.status_code} for {search_date}: {e}")
                return None
    print(
        f"Failed to fetch search results for {search_date} after {max_retries} attempts"
    )
    return None


def get_item_metadata(identifier):
    """
    Retrieves metadata for a given item identifier.
    """
    url = f"https://archive.org/metadata/{identifier}"
    print(f"Retrieving metadata for item '{identifier}'...")
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Error retrieving metadata for {identifier}: HTTP {response.status_code}"
        )
        return None


def choose_all_mp3_files(metadata):
    """
    Returns a list of file infos for files ending with .mp3.
    """
    files = metadata.get("files", [])
    candidates = [f for f in files if f.get("name", "").lower().endswith(".mp3")]
    return candidates


def download_file(identifier, file_info, show_date, index, output_dir=OUTPUT_DIR):
    """
    Downloads a file from Archive.org. If a 404 error occurs, log and skip.
    Files are saved in a folder based on the year from show_date.
    """
    # Use the year part for folder organization.
    year = show_date.split("-")[0]
    year_folder = os.path.join(output_dir, year)
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)

    original_name = file_info.get("name")
    _, ext = os.path.splitext(original_name)
    new_filename = f"{show_date}_{index}{ext}"
    local_path = os.path.join(year_folder, new_filename)

    download_url = f"https://archive.org/download/{identifier}/{original_name}"
    print(f"Downloading file from {download_url} to {local_path} ...")
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Download complete:", local_path)
    except requests.exceptions.HTTPError as err:
        print(f"Failed to download {original_name} for {show_date}: {err}")


def main():
    # Determine resume date if any file exists
    resume_date = get_resume_date(OUTPUT_DIR)
    if resume_date:
        current_date = resume_date
        print(f"Resuming from {current_date.strftime('%Y-%m-%d')}")
    else:
        current_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
        print(f"Starting from default start date {START_DATE_STR}")

    end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d")
    total_count = 0

    while current_date <= end_date:
        search_date = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing date {search_date}...")
        item = advanced_search_item_by_date(search_date)
        if not item:
            current_date += timedelta(days=1)
            continue

        identifier = item.get("identifier")
        print(f"Found item: {identifier} for {search_date}")
        metadata = get_item_metadata(identifier)
        if not metadata:
            current_date += timedelta(days=1)
            continue

        # Use metadata's date if available; otherwise, use search_date.
        show_date = metadata.get("metadata", {}).get("date", search_date)
        print(f"Show date: {show_date}")

        mp3_files = choose_all_mp3_files(metadata)
        if mp3_files:
            print(f"Found {len(mp3_files)} mp3 file(s) for item {identifier}.")
            for index, file_info in enumerate(mp3_files, start=1):
                print(
                    f"Downloading file '{file_info.get('name')}' as {show_date}_{index} ..."
                )
                download_file(identifier, file_info, show_date, index)
            total_count += 1
        else:
            print(f"No mp3 files found for item {identifier}.")

        current_date += timedelta(days=1)

    print(f"\nCompleted processing. Total dates with downloads: {total_count}")


if __name__ == "__main__":
    main()
