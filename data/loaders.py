from pathlib import Path
from typing import Optional, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
from data.dataset import FlowerDataset
from torch.utils.data import DataLoader, Dataset


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
    transforms = {mode: instantiate(cfg.dataset.transforms[mode]) for mode in ["train", "val", "test"]}

    # Create dataset instances for each mode
    datasets = {
        mode: FlowerDataset(root_dir=root_dir, mode=mode, transform=transforms[mode])
        for mode in ["train", "val", "test"]
    }

    return datasets



def get_data_loaders(datasets: Dict[str, Dataset], cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

    dataloaders = {}

    for mode, dataset in datasets.items():
        dataloader_cfg = cfg.dataloader[mode]
        dataloaders[mode] = DataLoader(dataset=dataset, **dataloader_cfg)

    return dataloaders

