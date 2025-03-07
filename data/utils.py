import logging
from pathlib import Path
from typing import Optional, Tuple
from hydra.utils import instantiate
from data.dataset import FlowerDataset
from utils.logging_configs import setup_logging
from torch.utils.data import DataLoader, Dataset
from utils.config_loader import load_hydra_config


setup_logging()


def get_data_loaders(config_dir: Path, config_name: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads dataset and dataloader configurations via Hydra, instantiates transforms,
    and returns PyTorch DataLoaders for training, validation, and testing.

    This function:
    - Loads dataset and dataloader configurations from Hydra.
    - Instantiates dataset transformations dynamically.
    - Creates `FlowerDataset` instances for train, validation, and test.
    - Constructs `DataLoader` instances using the configurations.

    Args:
        config_dir (Path): The directory containing Hydra configuration files.
        config_name (str): The name of the main configuration file (without `.yaml` extension).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            - `train_dataloader`: DataLoader for training data.
            - `val_dataloader`: DataLoader for validation data.
            - `test_dataloader`: DataLoader for test data.
    """
    cfg = load_hydra_config(config_dir, config_name)
    root_dir = Path(cfg.dataset.root_dir).resolve()

    # Instantiate transforms using Hydra
    train_transforms = instantiate(cfg.dataset.transforms.train)
    val_transforms = instantiate(cfg.dataset.transforms.val)
    test_transforms = instantiate(cfg.dataset.transforms.test)

    # Create dataset instances
    train_dataset = FlowerDataset(root_dir=root_dir, mode="train", transforms=train_transforms)
    val_dataset = FlowerDataset(root_dir=root_dir, mode="val", transforms=val_transforms)
    test_dataset = FlowerDataset(root_dir=root_dir, mode="test", transforms=test_transforms)

    # Extract separate dataloader configurations
    train_dataloader = DataLoader(train_dataset, **cfg.dataloader["train"])
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader["val"])
    test_dataloader = DataLoader(test_dataset, **cfg.dataloader["test"])

    return train_dataloader, val_dataloader, test_dataloader

