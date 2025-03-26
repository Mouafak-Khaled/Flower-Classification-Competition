import lightning as L
from pathlib import Path
from typing import Any, Optional
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from data.loaders import get_dataset, get_dataloader
from data.setup import split_organize_dataset, process_dataset_pipeline


class FlowerDataModule(L.LightningDataModule):
    """
    A custom PyTorch Lightning DataModule for loading flower classification images.

    This module handles the downloading, processing, and loading of flower classification datasets.
    It organizes the dataset into training, validation, and test sets, and applies specified transformations.

    Attributes:
        configs (DictConfig): Configuration object containing dataset parameters.
        transforms (Optional[Dict[str, Any]]): Optional transformations to apply to images.
        target_transforms (Optional[Dict[str, Any]]): Optional transformations to apply to labels.
        num_instances (Optional[Dict[str, int]]): Optional number of instances to sample per class.
        train_dataset (Optional[Dataset]): The training dataset.
        val_dataset (Optional[Dataset]): The validation dataset.
        test_dataset (Optional[Dataset]): The test dataset.
    """

    def __init__(
            self,
            configs: DictConfig,
            transforms: Optional[Dict[str, Any]] = None,
            target_transforms: Optional[Dict[str, Any]] = None,
            num_instances: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Initializes the FlowerDataModule.

        Args:
            configs (DictConfig): Configuration object containing dataset parameters.
            transforms (Optional[Dict[str, Any]]): Transformations to apply to images. Defaults to None.
            target_transforms (Optional[Dict[str, Any]]): Transformations to apply to labels. Defaults to None.
            num_instances (Optional[Dict[str, int]]): Number of instances to sample per class. Defaults to None.
        """
        super().__init__()
        self.configs = configs
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.num_instances = num_instances
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def prepare_data(self) -> None:
        """
        Prepares the dataset by downloading and processing it.

        This method is called only once and is used to download the dataset and organize it into
        training, validation, and test sets.
        """
        dataset_config = OmegaConf.to_container(self.configs.dataset, resolve=True)
        metadata = process_dataset_pipeline(configurations=dataset_config)
        split_organize_dataset(dataset_metadata=metadata, configurations=dataset_config)


    def setup(self, stage) -> None:
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (Optional[str]): The stage for which to set up the datasets ('fit', 'test', or None).
        """
        root_dir = Path(self.configs.dataset.processed_dir).resolve()

        if stage == 'fit' or stage is None:

            train_transforms = self._get_transforms(mode='train')
            val_transforms = self._get_transforms(mode='val')

            self.train_dataset = get_dataset(
                root_dir=root_dir,
                mode='train',
                transforms=train_transforms,
                target_transforms=self.target_transforms.get('train', None) if self.target_transforms else None,
                num_instances=self.num_instances.get('train', None) if self.num_instances else None
            )

            self.val_dataset = get_dataset(
                root_dir=root_dir,
                mode='val',
                transforms=val_transforms,
                target_transforms=self.target_transforms.get('val', None) if self.target_transforms else None,
                num_instances=self.num_instances.get('val', None) if self.num_instances else None
            )

        if stage == 'test' or stage is None:

            test_transforms = self._get_transforms(mode='test')

            self.test_dataset = get_dataset(
                root_dir=root_dir,
                mode='test',
                transforms=test_transforms,
                target_transforms=self.target_transforms.get('test', None) if self.target_transforms else None,
                num_instances=self.num_instances.get('test', None) if self.num_instances else None
            )


    def train_dataloader(self):
        """
        Creates a PyTorch DataLoader instance for a train dataset.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the train dataset.
        """
        return get_dataloader(self.train_dataset, mode='train', cfg=self.configs.dataloader)


    def val_dataloader(self):
        """
        Creates a PyTorch DataLoader instance for a validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the validation dataset.
        """
        return get_dataloader(self.val_dataset, mode='val', cfg=self.configs.dataloader)


    def test_dataloader(self):
        """
        Creates a PyTorch DataLoader instance for a test dataset.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the test dataset.
        """
        return get_dataloader(self.test_dataset, mode='test', cfg=self.configs.dataloader)


    def _get_transforms(self, mode: str) -> Optional[Any]:
        """
        Retrieves the transformations for the specified mode.

        Args:
            mode (str): The mode for which to retrieve transformations ('train', 'val', or 'test').

        Returns:
            Optional[Any]: The transformations for the specified mode, or None if not defined.
        """
        if self.transforms is None:
            return instantiate(self.configs.dataset.transforms[mode])
        return self.transforms.get(mode, None) if self.transforms else None


