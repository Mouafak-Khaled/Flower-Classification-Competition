import random
from pathlib import Path
from typing import Optional, Tuple, Any
from omegaconf import DictConfig
from hydra.utils import instantiate
from data.dataset import FlowerDataset
from torch.utils.data import DataLoader, Dataset, Subset
from utils.constants import DatasetMode


def get_dataset(
        root_dir: Path,
        mode: str,
        transforms: Any,
        target_transforms: Optional[Any] = None,
        num_instances: Optional[int] = None
) -> Dataset:
    """
    Creates a PyTorch Dataset instance for a specified dataset split (train, val, or test).

    This function:
    - Ensures that the `mode` is one of ['train', 'val', 'test'].
    - Instantiates a dataset using the given transformations.
    - Optionally returns a randomly sampled subset of the dataset.

    Args:
        root_dir (Path): The root directory where the dataset is stored.
        mode (str): The dataset split mode (must be one of "train", "val", or "test").
        transforms (Any): The transformation pipeline to be applied to the dataset.
        target_transforms (Optional[Any], optional): Transformation applied to the target labels. Defaults to None.
        num_instances (Optional[int], optional): The number of samples to randomly select from the dataset. Defaults to None.

    Returns:
        Dataset: A PyTorch Dataset instance (either full dataset or randomly sampled subset).

    Raises:
        AssertionError: If `mode` is not one of ["train", "val", "test"].
        ValueError: If `num_instances` is greater than the dataset size.
    """
    assert mode in DatasetMode.list(), f"Invalid mode: {mode}. Must be one of {DatasetMode.list()}"

    dataset = FlowerDataset(root_dir=root_dir, mode=mode, transforms=transforms, target_transforms=target_transforms)

    # If num_instances is specified, return a random subset of the dataset
    if num_instances is not None:
        if num_instances > len(dataset):
            raise ValueError(f"Requested {num_instances} samples, but dataset only has {len(dataset)} samples.")

        subset_indices = random.sample(range(len(dataset)), num_instances)
        dataset = Subset(dataset, subset_indices)

    return dataset


def get_datasets(cfg: DictConfig) -> Dict[str, Dataset]:
    """
    Creates dataset instances for training, validation, and testing using the provided Hydra configuration.

    This function:
    - Resolves the dataset root directory.
    - Instantiates transformations for each dataset split using Hydra.
    - Creates and returns dataset instances for 'train', 'val', and 'test'.

    Args:
        cfg (DictConfig): The loaded Hydra configuration containing dataset settings.

    Returns:
        Dict[str, Dataset]: A dictionary containing dataset instances:
            - "train": Training dataset.
            - "val": Validation dataset.
            - "test": Test dataset.
    """


    root_dir = Path(cfg.dataset.root_dir).resolve()

    # Instantiate dataset transformations
    transforms = {mode: instantiate(cfg.dataset.transforms[mode]) for mode in DatasetMode.list()}

    # Create dataset instances for each mode
    datasets = {
        mode: get_dataset(root_dir=root_dir, mode=mode, transforms=transforms[mode])
        for mode in DatasetMode.list()
    }

    return datasets


def get_dataloader(dataset: Dataset, mode: str, cfg: DictConfig) -> DataLoader:
    """
    Creates a PyTorch DataLoader instance for a specified dataset split (train, val, or test).

    This function:
    - Extracts the corresponding dataloader settings from the provided Hydra configuration.
    - Ensures that the `mode` is one of ['train', 'val', 'test'] before proceeding.
    - Instantiates a DataLoader for the given dataset split using the appropriate configurations.

    Args:
        dataset (Dataset): The dataset instance for the specified mode.
        mode (str): The dataset split mode (must be one of "train", "val", or "test").
        cfg (DictConfig): The Hydra configuration containing dataloader settings.

    Returns:
        DataLoader: A PyTorch DataLoader configured for the specified dataset split.

    Raises:
        AssertionError: If `mode` is not one of ["train", "val", "test"].
    """
    assert mode in DatasetMode.list(), f"Invalid mode: {mode}. Must be one of {DatasetMode.list()}"

    return DataLoader(dataset=dataset, **cfg.dataloader[mode])


def get_dataloaders(datasets: Dict[str, Dataset], cfg: DictConfig) -> Dict[str, DataLoader]:
    """
    Creates DataLoader instances for training, validation, and testing using the provided dataset instances.

    This function:
    - Extracts the dataloader settings for each dataset split.
    - Creates DataLoaders for 'train', 'val', and 'test' datasets using Hydra configurations.

    Args:
        datasets (Dict[str, Dataset]): A dictionary containing dataset instances.
        cfg (DictConfig): The loaded Hydra configuration containing dataloader settings.

    Returns:
        Dict[str, DataLoader]: A dictionary containing DataLoader instances:
            - "train": DataLoader for training data.
            - "val": DataLoader for validation data.
            - "test": DataLoader for test data.
    """

    dataloaders = {
        mode: get_dataloader(datasets[mode], mode=mode, cfg=cfg)
        for mode in DatasetMode.list()
    }

    return dataloaders


