from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Any, Optional
from .dataset_setup import read_processed_data


class FlowerDataset(Dataset):
    """
    A custom PyTorch Dataset for loading flower classification images.

    This dataset reads preprocessed images stored in directories (train, val, test),
    associates them with numerical labels, and applies optional transformations.

    Attributes:
        _root_dir (Path): The root directory containing processed image data.
        _mode (str): The dataset mode ('train', 'val', or 'test').
        _transform (Optional[Callable]): Optional transformation applied to images.
        _target_transform (Optional[Callable]): Optional transformation applied to labels.
        _data (List[Tuple[Path, int]]]: List of (image_path, label) tuples.
    """

    def __init__(
        self,
        root_dir: Path,
        mode: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ) -> None:
        """
        Initializes the FlowerDataset.

        Args:
            root_dir (Path): The root directory where the processed data is stored.
            mode (str): The dataset mode ('train', 'val', or 'test').
            transform (Optional[Callable], optional): A function/transform to apply to images. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform to apply to labels. Defaults to None.
        """
        self._root_dir = root_dir
        self._mode = mode
        self._transform = transform
        self._target_transform = target_transform

        # Load data containing (image_path, label) pairs
        self._data = read_processed_data(root_dir=self._root_dir, mode=self._mode)

        if self._data is None or len(self._data) == 0:
            raise ValueError(f"No data found in {self._root_dir / self._mode}. Ensure data is properly processed.")


    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self._data)


    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Retrieves an image and its corresponding label.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[Any, int]: A tuple containing:
                - The transformed image as a tensor.
                - The integer label corresponding to the image.

        """
        image_path, label = self._data[index]

        # Load image and convert to RGB
        img = Image.open(image_path).convert("RGB")

        # Apply transformations if specified
        if self._transform:
            img = self._transform(img)

        if self._target_transform:
            label = self._target_transform(label)

        return img, label
