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
        root_dir (str): The root directory containing dataset files.
        transforms (Dict[str, TransformConfig]): A dictionary mapping dataset splits 
            (e.g., 'train', 'val', 'test') to their respective transformation configurations.
    """
    root_dir: str
    transforms: Dict[str, TransformConfig]


@dataclass
class DataLoaderConfig:
    """
    Configuration for PyTorch DataLoader.

    Attributes:
        num_workers (int): Number of subprocesses to use for data loading (default: 1).
        pin_memory (bool): Whether to use pinned memory for GPU acceleration (default: False).
        batch_size (int): Number of samples per batch (default: 32).
        shuffle (bool): Whether to shuffle the dataset (default: True).
    """
    num_workers: int = 1
    pin_memory: bool = False
    batch_size: int = 32
    shuffle: bool = True


@dataclass
class DataConfigs:
    """
    Unified configuration schema combining dataset and dataloader configurations.

    Attributes:
        dataset (DatasetConfig): The dataset configuration including root directory and transformations.
        dataloader (Dict[str, DataLoaderConfig]): A dictionary containing separate dataloader configurations
            for 'train', 'val', and 'test' splits.
    """
    dataset: DatasetConfig
    dataloader: Dict[str, DataLoaderConfig]


"""
Register the configurations with Hydra's ConfigStore
- This registers the `DataConfigs` dataclass with Hydra under the name 'data_configs'.
- It allows Hydra to automatically load and validate configurations following this schema.
"""
cs = ConfigStore.instance()
cs.store(name="data_configs", node=DataConfigs)

