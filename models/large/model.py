import torch.nn as nn
from models.base_model import BaseFlowerClassifier


class ConvBlock(nn.Module):
    """
    A reusable convolutional block consisting of a convolutional layer,
    activation function, batch normalization, and optional max pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Default is 3.
        bias (bool, optional): Whether to use bias in the convolution. Default is True.
        use_maxpool (bool, optional): Whether to apply max pooling. Default is False.
        pool_size (int, optional): Kernel size for max pooling. Default is 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, use_maxpool=False, pool_size=2):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)]

        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.1))

        if use_maxpool:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LargeFlowerClassifier(BaseFlowerClassifier):
    """
    A deep CNN classifier for large-scale flower classification tasks.

    This model:
    - Uses multiple convolutional layers to extract hierarchical features.
    - Applies batch normalization for stable training.
    - Employs adaptive pooling to handle varying input sizes.
    - Uses a fully connected classifier head to predict the number of classes.

    Args:
        configs (DictConfig): Configuration containing model parameters such as:
            - `in_features`: Number of input channels.
            - `num_classes`: Number of output classes.
            - `bias`: Whether to use bias in convolutional layers.
    """

    def build_model(self) -> nn.Module:
        """
        Constructs a deep CNN model with modular convolutional blocks and a fully connected classifier head.

        Returns:
            nn.Module: A neural network model consisting of:
                - Multiple convolutional blocks for feature extraction.
                - Adaptive average pooling to ensure a fixed-size feature representation.
                - Fully connected layers for classification.
        """

        in_features = self.configs.model.in_features
        bias = self.configs.model.bias

        # Feature Extractor
        feature_extractor = nn.Sequential(
            ConvBlock(in_features, 8, bias=bias),
            ConvBlock(8, 16, bias=bias),
            ConvBlock(16, 32, use_maxpool=True, bias=bias),
            ConvBlock(32, 64, bias=bias),
            ConvBlock(64, 128, use_maxpool=True, bias=bias),
            ConvBlock(128, 256, bias=bias),
            ConvBlock(256, 128, use_maxpool=True, bias=bias),
            ConvBlock(128, 256, bias=bias),
            ConvBlock(256, 512, use_maxpool=True, bias=bias),
            ConvBlock(512, 1024, use_maxpool=True, bias=bias),
            nn.AdaptiveAvgPool2d((4, 4))  # Output will be 1024 x 4 x 4 = 16384 features
        )

        # Fully Connected Classifier Head
        classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(16384, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(1024),

            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(128),

            nn.Linear(128, self.configs.model.num_classes)
        )

        return nn.Sequential(feature_extractor, classifier)

