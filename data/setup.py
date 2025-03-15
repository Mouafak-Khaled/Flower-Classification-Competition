import os
import tarfile
import shutil
import random
import logging
import urllib.request
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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
        dataset = {}

        for index, image_name in enumerate(lines):
            if image_name.endswith(extension):
                class_id = index // images_per_class + 1
                dataset.setdefault(class_id, []).append(image_name)

        return dataset

    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise


def process_dataset_pipeline(configurations: Dict[str, Any]) -> Optional[Dict[int, List[str]]]:
    """
    Executes the dataset preparation pipeline, including downloading, extracting, and parsing dataset metadata.

    This function performs the following steps:
    1. Download Dataset: Retrieves the dataset archive from a specified URL.
    2. Extract Dataset: Extracts the dataset contents into a specified directory.
    3. Read Metadata: Reads and parses image metadata from a text file.

    Args:
        configurations (Dict[str, Any]): Dictionary containing dataset configuration parameters.
            Expected keys:
            - "dataset_url" (str): URL of the dataset to download.
            - "root_dir" (Path): Directory where the dataset will be stored.
            - "archive_filename" (str): Name of the dataset archive file.
            - "extract_dir" (Path): Directory where extracted dataset contents will be stored.
            - "file_list_path" (Path): Path to the text file containing image metadata.

    Returns:
        Optional[Dict[int, List[str]]]: A dictionary mapping class IDs to lists of image names:
            - `int`: The corresponding class ID.
            - `List[str]`: A list of images belong to the corresponding class.

        Returns `None` if any step in the pipeline fails.

    Raises:
        FileNotFoundError: If the specified dataset metadata file is missing.
        HTTPError: If an HTTP error occurs during dataset download.
        URLError: If the dataset URL is unreachable or invalid.
        OSError: If file system-related errors occur (e.g., permission issues).
        ValueError: If the dataset metadata file is empty or incorrectly formatted.
        Exception: If any other unexpected error occurs.
    """
    try:
        dataset_url = configurations["dataset_url"]
        download_dir = Path(configurations["root_dir"]).resolve()
        archive_filename = configurations["archive_filename"]
        extract_dir = Path(configurations["extract_dir"]).resolve() # Fixed incorrect key spacing
        file_list_path = Path(configurations["txt_file"]).resolve()

        # Step 1: Download the dataset
        dataset_downloaded = download_dataset(dataset_url, download_dir, archive_filename)
        if not dataset_downloaded:
            logging.error(f"Failed to download dataset from {dataset_url}.")
            return None

        # Step 2: Extract dataset
        archive_path = download_dir / archive_filename
        extraction_success = extract_data(archive_path, extract_dir)
        if not extraction_success:
            logging.error(f"Failed to extract dataset from {archive_path}.")
            return None

        # Step 3: Read dataset metadata
        try:
            dataset_metadata = read_data_from_txt_file(file_list_path)
            return dataset_metadata

        except FileNotFoundError as e:
            logging.error(f"Metadata file not found at {file_list_path}: {e}")

        except ValueError as e:
            logging.error(f"Invalid or empty dataset metadata file {file_list_path}: {e}")

        except Exception as e:
            logging.error(f"Unexpected error while reading metadata from {file_list_path}: {e}")

        return None

    except KeyError as e:
        logging.error(f"Missing required configuration key: {e}")

    except Exception as e:
        logging.error(f"Unexpected error in dataset processing pipeline: {e}")

    return None


def split_organize_dataset(
        dataset_metadata: List[Tuple[str, int]],
        configurations: Dict[str, Any]
):
    """
    Splits a list of (image, label) tuples into training, validation, and test sets based on given ratios.
    Organizes images into structured directories `{processed_data_dir}/train/{class_id}/`, `{processed_data_dir}/val/{class_id}/`, etc.

    Args:
        dataset_metadata (List[Tuple[str, int]]): A list where each element is:
            - `str`: Image filename (relative to `extract_dir`).
            - `int`: Corresponding class label.
        configurations (Dict[str, Any]): Configuration dictionary containing:
            - "extract_dir" (str): Directory where extracted dataset is stored.
            - "processed_data_dir" (str): Directory where split dataset will be saved.
            - "train_split" (float): Proportion of dataset for training.
            - "val_split" (float): Proportion of dataset for validation.
            - "test_split" (float): Proportion of dataset for testing.
            - "seed" (int): Random seed for reproducibility.

    Returns:
        None

    Raises:
        ValueError: If split ratios do not sum to 1.
        FileNotFoundError: If an image file is missing.
        Exception: For unexpected errors.
    """

    # Load configuration values
    extract_dir = (Path(configurations["extract_dir"]) / "jpg").resolve()  # Image source directory
    processed_data_dir = Path(configurations["processed_data_dir"])  # Output directory
    train_ratio = configurations["train_split"]
    val_ratio = configurations["val_split"]
    test_ratio = configurations["test_split"]
    seed = configurations["seed"]

    # Ensure split ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-3):
        raise ValueError(f"Split ratios must sum to 1. Found: {total_ratio}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Create output directories
    for split in ["train", "val", "test"]:
        (processed_data_dir / split).mkdir(parents=True, exist_ok=True)

    #Split and copy images
    for class_id, images in dataset_metadata.items():
        random.shuffle(images)  # Shuffle images to ensure randomness

        num_total = len(images)
        num_train = int(train_ratio * num_total)
        num_val = int(val_ratio * num_total)

        # Ensure random sampling
        train_images = random.sample(images, num_train)
        remaining_images = list(set(images) - set(train_images))
        val_images = random.sample(remaining_images, num_val)
        test_images = list(set(remaining_images) - set(val_images))

        # Organize and copy images
        for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            class_dir = processed_data_dir / split_name / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)

            for image in split_images:
                source_path = extract_dir / image
                dest_path = class_dir / image

                if not source_path.exists():
                    logging.error(f"File not found: {source_path}")
                    continue

                shutil.copy(source_path, dest_path)

        logging.info(
            f"Processed Class {class_id}: Train {len(train_images)}, Val {len(val_images)}, Test {len(test_images)}"
        )

    logging.info("Dataset splitting and organization completed successfully.")


def read_processed_data(root_dir: Path, mode: str, extension: Optional[str] = 'jpg') -> List[Tuple[Path, int]]:
    """
    Reads processed image data from a given directory and returns a list of image paths with corresponding labels.

    This function:
    - Verifies that the specified dataset mode directory (`train`, `val`, or `test`) exists and is not empty.
    - Iterates through subdirectories (class folders) and extracts image file paths.
    - Assigns each image a label based on its parent directory name.
    - Skips any empty class directories but does not raise an error if some are empty.

    Args:
        root_dir (Path): The root directory containing processed data (e.g., "processed").
        mode (str): The mode to read from ("train", "val", or "test").

    Returns:
        List[Tuple[Path, int]]: A list of tuples where each tuple contains:
            - The absolute Path to an image file.
            - The integer label corresponding to its class.

    Raises:
        FileNotFoundError: If the specified dataset mode directory does not exist or is entirely empty.
        Exception: If the specified dataset is empty.
    """

    target_dir = root_dir / mode

    if not (target_dir.exists() and any(target_dir.iterdir())):
        logging.error(f"Directory not found or empty: {target_dir}")
        raise FileNotFoundError(f"Processed dataset directory {target_dir} does not exist or is empty.")

    data = []

    for directory in sorted(target_dir.iterdir()):
        if directory.is_dir():
            images = list(directory.glob(f"*.{extension}"))  # List images

            if not images:
                continue

            label = int(directory.name) - 1  # Use folder name as label
            data.extend([(img.resolve(), label) for img in images])

    if not data:
        logging.error(f"Data is empty: {data}")
        raise Exception(f"Data is empty: {data}. No data found in {target_dir}")

    return data
