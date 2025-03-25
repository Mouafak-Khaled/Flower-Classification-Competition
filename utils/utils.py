import torch.nn as nn
import torchvision.models  as models
from torchvision.transforms import Compose
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of learnable parameters in a PyTorch model.

    Args:
        model (Module): The PyTorch model.

    Returns:
        int: Total number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pretrained_transforms() -> Compose:
    """
    Retrieves the default preprocessing transforms for the MobileNetV3-Small pre-trained model.

    The returned transformation pipeline includes standard preprocessing steps such as resizing,
    normalization, and center cropping, optimized for the model’s expected input format.

    Returns:
        torchvision.transforms.Compose: A composition of preprocessing transformations.

    Raises:
        ValueError: If the specified model does not support preprocessing transforms.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    if not hasattr(weights, "transforms"):
        raise ValueError("Preprocessing transforms are not available for the backbone model.")

    return weights.transforms()  # Returns a torchvision.transforms.Compose object


def init_weights(model: nn.Module, method: str = "kaiming") -> None:
    """
    Initializes weights of the given model using the specified method.

    Args:
        model (nn.Module): The model whose weights need initialization.
        method (str): Initialization method, one of ['kaiming', 'xavier', 'normal']. Default is 'kaiming'.
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if method == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif method == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unsupported initialization method: {method}")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

