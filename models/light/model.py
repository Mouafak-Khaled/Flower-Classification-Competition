import torch.nn as nn
from models.base_model import BaseFlowerClassifier
import torchvision.models as models


class FastFlowerClassifier(BaseFlowerClassifier):
    """
    A modular classifier for transfer learning, supporting pre-trained MobileNetV3-Small as a backbone.

    This model:
    - Loads a pre-trained MobileNetV3-Small model as a feature extractor.
    - Freezes the feature extractor layers (only the classifier head is trainable).
    - Uses a custom classifier head, which includes convolutional, activation, pooling, and fully connected layers.
    - Supports handling varying input image sizes via adaptive pooling and convolution operations.

    Args:
        configs (DictConfig): Configuration object containing model parameters like `in_features`, `num_classes`, and other settings for the classifier head.

    Raises:
        ValueError: If an unsupported backbone is provided in the configurations.
    """

    def build_model(self) -> nn.Module:
        """
        Constructs a CNN model using MobileNetV3-Small as a feature extractor and adds a custom classifier head.

        This function:
        - Initializes a pre-trained MobileNetV3-Small model from torchvision.
        - Freezes the feature extractor's parameters to prevent updates during training.
        - Adds a custom classifier head, including convolutional layers, activation functions, pooling, and fully connected layers.

        Returns:
            nn.Module: A neural network model consisting of:
                - A MobileNetV3-Small feature extractor with frozen weights.
                - A custom classifier head with convolutional, activation, pooling, and fully connected layers.

        Raises:
            ValueError: If an unsupported backbone is provided in the configurations.
        """

        # Load pre-trained MobileNetV3-Small model
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feature_extractor = model.features

        # Freeze feature extractor parameters to prevent updates during training
        for param in feature_extractor.parameters():
            param.requires_grad = False

        # Custom classifier with convolutional layers, ReLU activations, pooling, and fully connected layers
        classifier = nn.Sequential(
            feature_extractor,
            nn.Conv2d(self.configs.model.in_features, 64, kernel_size=1, bias=self.configs.model.bias),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, bias=self.configs.model.bias),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, bias=self.configs.model.bias),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.configs.model.num_classes)
        )

        return classifier
