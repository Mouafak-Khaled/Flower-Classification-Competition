from torch.nn import Module, Sequential
import torchvision.models  as models
from torchvision.transforms import Compose


def count_parameters(model: Module) -> int:
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

