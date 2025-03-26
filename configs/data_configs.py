from dataclasses import dataclass, field
from typing import Any, Dict, List
from hydra.core.config_store import ConfigStore


@dataclass
class TransformConfig:
    """
    Configuration for dataset transformations.

    Attributes:
        _target_ (str): The target function/class to be instantiated by Hydra (e.g., `torchvision.transforms.Compose`).
        transforms (List[Dict[str, Any]]): A list of transformations applied to the dataset.
    """
    _target_: str
    transforms: List[Dict[str, Any]]


@dataclass
class DatasetConfig:
    """
    Configuration for dataset properties.

    Attributes:
        dataset_url (str): The URL to download the dataset.
        archive_filename (str): The name of the archive file containing the dataset.
        root_dir (str): The root directory containing dataset files.
        extract_dir (str): The directory where the dataset will be extracted.
        processed_dir (str): The directory for storing processed dataset files.
        metadata_file (str): The path to the metadata file.
        images_per_class (int): The number of images per class.
        image_extension (str): The file extension for images (e.g., '.jpg').
        train_split (float): Proportion of the dataset to use for training.
        val_split (float): Proportion of the dataset to use for validation.
        test_split (float): Proportion of the dataset to use for testing.
        seed (int): Random seed for reproducibility.
    """
    dataset_url: str
    archive_filename: str
    root_dir: str
    extract_dir: str
    processed_dir: str
    metadata_file: str
    images_per_class: int
    image_extension: str
    train_split: float
    val_split: float
    test_split: float
    seed: int


@dataclass
class DataLoaderConfig:
    """
    Configuration for PyTorch DataLoader.

    Attributes:
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to use pinned memory for GPU acceleration.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
    """
    num_workers: int
    pin_memory: bool
    batch_size: int
    shuffle: bool


@dataclass
class DataConfigs:
    """
    Unified configuration schema combining dataset and dataloader configurations.

    Attributes:
        dataset (DatasetConfig): The dataset configuration including root directory and transformations.
        transforms (Dict[str, TransformConfig]): A dictionary mapping dataset splits
            (e.g., 'train', 'val', 'test') to their respective transformation configurations.
        dataloader (Dict[str, DataLoaderConfig]): A dictionary containing separate dataloader configurations
            for 'train', 'val', and 'test' splits.
    """
    dataset: DatasetConfig
    transforms: Dict[str, TransformConfig]
    dataloader: Dict[str, DataLoaderConfig]


# Register the configurations with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="data_configs", node=DataConfigs)