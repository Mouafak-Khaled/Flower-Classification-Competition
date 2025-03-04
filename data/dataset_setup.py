import os
import tarfile
from pathlib import Path
import urllib.request
import logging
from urllib.error import URLError, HTTPError


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
            return True

        urllib.request.urlretrieve(dataset_url, file_path)

        return True

    except HTTPError as e:
        logging.error(f"HTTP error occurred: {e.code} - {e.reason}")

    except URLError as e:
        logging.error(f"URL error occurred: {e.reason}")

    except OSError as e:
        logging.error(f"File system error: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False


def extract_data(data_path: Path, output_path: Path) -> bool:
    """
    Extracts a .tgz dataset to the specified directory (`data/raw`).

    Args:
        archive_path (Path): Path to the .tgz file.
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
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if already extracted (assumption: extracted files are inside `jpg/`)
        if any(output_path.iterdir()):
            return True

        with tarfile.open(data_path, "r:gz") as tar:
            tar.extractall(path=output_path)

        return True

    except tarfile.TarError as e:
        logging.error(f"Error extracting tar file: {e}")

    except OSError as e:
        logging.error(f"File system error: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False
