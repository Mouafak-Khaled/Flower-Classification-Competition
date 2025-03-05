import os
import tarfile
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError, HTTPError
from utils.logging_configs import setup_logging


setup_logging()


def download_dataset(dataset_url: str, dir_path: Path, filename: str) -> bool:
    """
    Downloads a dataset from a given URL and saves it to a specified directory.

    If the dataset file already exists, the function skips the download and returns True.

    Args:
        dataset_url (str): The URL of the dataset to download.
        dir_path (Path): The directory where the dataset should be stored.
        filename (str): The name of the file to save the dataset as.

    Returns:
        bool: True if the dataset was downloaded successfully or already exists, False otherwise.

    Raises:
        HTTPError: If an HTTP error occurs during the download.
        URLError: If there is an issue with the URL.
        OSError: If there is a file system-related error.
        Exception: If any other unexpected error occurs.
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path / filename

        # Check if dataset already exists
        if file_path.exists():
            logging.info(f"Dataset already exists at {file_path}. Skipping download.")
            return True

        logging.info(f"Downloading dataset from {dataset_url}...")

        urllib.request.urlretrieve(dataset_url, file_path)

        logging.info(f"Dataset downloaded successfully: {file_path}")

    except HTTPError as e:
        logging.error(f"HTTP error occurred: {e.code} - {e.reason}")

    except URLError as e:
        logging.error(f"URL error occurred: {e.reason}")

    except OSError as e:
        logging.error(f"File system error: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False


def extract_data(data_path: Path, extract_to: Path) -> bool:
    """
    Extracts a .tgz dataset to the specified directory (`data/raw`).

    Args:
        data_path (Path): Path to the .tgz file.
        extract_to (Path): Directory where the dataset should be extracted (`data/raw`).

    Returns:
        bool: True if extraction was successful or already extracted, False otherwise.

    Raises:
        tarfile.TarError: If there is an issue extracting the archive.
        OSError: If a file system error occurs.
        Exception: If any other unexpected error occurs.
    """
    try:
        # Ensure extraction directory exists
        extract_to.mkdir(parents=True, exist_ok=True)

        # Check if already extracted (assumption: extracted files are inside the directory)
        if any(extract_to.iterdir()):
            logging.info(f"Dataset already extracted at {extract_to}. Skipping extraction.")
            return True

        logging.info(f"Extracting dataset from {data_path} to {extract_to}...")

        with tarfile.open(data_path, "r:gz") as tar:
            tar.extractall(path=extract_to)

        logging.info(f"Dataset extracted successfully to {extract_to}")
        return True

    except tarfile.TarError as e:
        logging.error(f"Error extracting tar file: {e}")

    except OSError as e:
        logging.error(f"File system error: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False


def read_data_from_txt_file(
    file_path: Path, images_per_class: Optional[int] = 80, extension: Optional[str] = "jpg"
) -> Dict[int, List[str]]:
    """
    Reads and parses a text file containing image names and organizes them into classes.

    The function assumes:
    - Images are listed sequentially and grouped into classes.
    - Each class contains a fixed number of images (`images_per_class`).
    - Only files with the specified extension (`extension`) are included.

    Args:
        file_path (Path): Path to the text file containing image names.
        images_per_class (Optional[int]): The number of images per class. Default is 80.
        extension (Optional[str]): The expected file extension (without a dot). Default is "jpg".

    Returns:
        Dict[int, List[str]]: A dictionary mapping class IDs to lists of image names.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or does not contain valid image names.
        Exception: If any other error occurs during file reading.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with file_path.open(mode="r") as file:
            lines = [line.strip() for line in file.readlines()]

        if not lines:
            raise ValueError("File is empty or does not contain valid image names.")

        extension = "." + extension.lstrip(".")  # Ensure extension starts with '.'
        # dataset = {}
        dataset = []

        for index, image_name in enumerate(lines):
            if image_name.endswith(extension):
                class_id = index // images_per_class + 1
                # dataset.setdefault(class_id, []).append(image_name)
                dataset.append((image_name, class_id))

        return dataset

    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise


def process_dataset_pipeline(
    dataset_url: str,
    download_dir: Path,
    archive_filename: str,
    extract_dir: Path,
    file_list_path: Path
) -> Optional[List[Tuple[str, int]]]:
    """
    Orchestrates the dataset preparation pipeline, including downloading, extracting, and parsing dataset metadata.

    This function:
    1. Downloads the dataset archive from a given URL.
    2. Extracts the archive contents to a specified directory.
    3. Reads and parses image metadata from a given text file.

    Args:
        dataset_url (str): The URL of the dataset to download.
        download_dir (Path): Directory where the dataset archive should be downloaded.
        archive_filename (str): The name of the downloaded dataset archive file.
        extract_dir (Path): Directory where the dataset should be extracted.
        file_list_path (Path): Path to the text file containing image names and metadata.

    Returns:
        Optional[List[Tuple[str, int]]]: A list of tuples where each tuple contains:
            - str: The image file name.
            - int: The assigned class ID.

        Returns `None` if any step in the pipeline fails.

    Raises:
        HTTPError: If an HTTP error occurs during dataset download.
        URLError: If there is an issue with the dataset URL.
        OSError: If file system-related errors occur.
        ValueError: If the dataset file is empty or invalid.
        Exception: If any other unexpected error occurs.
    """

    # Step 1: Download the dataset
    dataset_downloaded = download_dataset(dataset_url, download_dir, archive_filename)
    if not dataset_downloaded:
        logging.error(f"Failed to download dataset from {dataset_url}. Aborting pipeline.")
        return None

    # Step 2: Extract dataset
    archive_path = download_dir / archive_filename
    extraction_success = extract_data(archive_path, extract_dir)
    if not extraction_success:
        logging.error(f"Failed to extract dataset from {archive_path}. Aborting pipeline.")
        return None

    # Step 3: Read dataset metadata
    try:
        dataset_metadata = read_data_from_txt_file(file_list_path)
        return dataset_metadata

    except Exception as e:
        logging.error(f"Error reading dataset metadata from {file_list_path}: {e}")
        return None