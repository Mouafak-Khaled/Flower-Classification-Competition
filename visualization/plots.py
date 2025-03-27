import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from typing import List, Union, Dict, Sequence, Tuple
import numpy as np
import torch
import itertools


def display_sample_images(dataloader: DataLoader, num_images: int = 5) -> None:
    """
    Displays a few sample images from a PyTorch DataLoader.

    This function:
    - Retrieves a batch of images from the DataLoader.
    - Selects the first `num_images` images from the batch.
    - Converts images from PyTorch format (CHW) to matplotlib format (HWC).
    - Displays the images in a row using Matplotlib.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader to sample images from.
        num_images (int, optional): Number of images to display. Defaults to 5.

    Returns:
        None: Displays the images using Matplotlib.

    Raises:
        ValueError: If `num_images` is greater than the batch size in the DataLoader.
    """
    images, labels = next(iter(dataloader))

    if num_images > len(images):
        raise ValueError(f"num_images ({num_images}) cannot be greater than batch size ({len(images)}) in DataLoader.")

    images = images[:num_images]

    fig, axes = plt.subplots(1, num_images, figsize=(13, 5))
    for i, img in enumerate(images):
        img = torch.clamp(img.permute(1, 2, 0), 0, 1)  # Convert from CHW (PyTorch) to HWC (Matplotlib)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()


def plot_batch_class_distribution(dataloader: DataLoader) -> None:
    """
    Plots the class distribution of a a batch.

    This function:
    - Uses a seaborn countplot to visualize the distribution of class labels.
    - Helps in identifying class imbalances.

    Args:
        dataloader (DataLoader): A dataloader instance for obtaining data batch.

    Returns:
        None: Displays the class distribution plot using Matplotlib & seaborn.
    """
    images, labels = next(iter(dataloader))
    plot_class_distribution(labels=labels)


def plot_class_distribution(labels: Union[np.ndarray, List[int]]) -> None:
    """
    Plots the class distribution of a dataset.

    This function:
    - Uses a seaborn countplot to visualize the distribution of class labels.
    - Helps in identifying class imbalances.

    Args:
        labels (List[int]): A list of class labels corresponding to dataset samples.

    Returns:
        None: Displays the class distribution plot using Matplotlib & seaborn.
    """
    plt.figure(figsize=(13, 5))
    sns.countplot(x=labels)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()


def plot_metric(
        results: Dict[str, Sequence],
        title: str,
        xlabel: str = "Epochs",
        ylabel: str = "Metric",
        figsize: Tuple[int, int] = (13, 6),
):
    """
    Plots metrics such as training and validation loss or accuracy over epochs.

    Parameters:
        results (Dict[str, Sequence]):
            A dictionary where keys are metric names (e.g., 'train_loss', 'val_loss') and values are sequences of metric values.
        title (str):
            The title of the plot.
        xlabel (str, optional):
            Label for the x-axis. Default is 'Epochs'.
        ylabel (str, optional):
            Label for the y-axis. Default is 'Metric'.
        figsize (Tuple[int, int], optional):
            Figure size as (width, height). Default is (13, 6).

    Returns:
        None
    """
    if not results:
        print("No data provided to plot.")
        return

    plt.figure(figsize=figsize)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for label, data in results.items():
        plt.plot(data, label=label, marker='o', color=next(color_cycle))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


