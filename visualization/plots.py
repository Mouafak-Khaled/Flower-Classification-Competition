import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


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

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0)  # Convert from CHW (PyTorch) to HWC (Matplotlib)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()


def plot_class_distribution(labels: List[int]) -> None:
    """
    Plots the class distribution of a dataset.

    This function:
    - Uses a seaborn countplot to visualize the distribution of class labels.
    - Helps in identifying class imbalances.

    Args:
        labels (List[int]): A list of class labels corresponding to dataset samples.

    Returns:
        None: Displays the class distribution plot using Matplotlib.
    """
    plt.figure(figsize=(10, 5))
    sns.countplot(x=labels, palette="viridis")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()