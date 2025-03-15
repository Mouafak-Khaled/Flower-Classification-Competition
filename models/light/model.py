import torch
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig
from typing import Dict, Any, List, Tuple
from utils.metrics import accuracy
from hydra.utils import instantiate
import torchvision.models  as models


class FastFlowerClassifier(L.LightningModule):
    """
        A modular classifier for transfer learning, supporting multiple pre-trained backbones.

        This model:
        - Loads a pre-trained model as a feature extractor.
        - Freezes the feature extractor layers (only the classifier head is trainable).
        - Uses adaptive pooling to handle different feature map sizes.
        - Supports multiple backbones: MobileNetV3, ResNet18, and EfficientNet-B0.

        Args:
            configs (DictConfig): Configuration containing model, optimizer, and scheduler parameters.
        """

    def __init__(self, configs: DictConfig):
        """
        Initializes the classifier, loads the feature extractor, and sets up the classifier head.

        Args:
            configs (DictConfig): A Hydra configuration object containing:
                - model: Backbone type, number of classes, and classifier settings.
                - training: Optimizer and scheduler configurations.
        """
        super(FastFlowerClassifier, self).__init__()
        self.configs = configs

        # Load pre-trained feature extractor and get the number of output features
        self.classifier = self.build_model()


        self.loss_fn = instantiate(self.configs.loss)
        # Track outputs per batch for epoch-level metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def build_model(self) -> nn.Module:
        """
        Constructs a CNN model using MobileNetV3-Small as a feature extractor.

        This function initializes a pre-trained MobileNetV3-Small model, removes its classification
        layers, and builds a custom classifier on top of its feature extractor. The feature extractor's
        parameters are frozen to prevent updates during training.

        Returns:
            nn.Module: A neural network model consisting of:
                - A MobileNetV3-Small feature extractor with frozen weights.
                - A custom classifier with convolutional, activation, pooling, and fully connected layers.

        Raises:
            ValueError: If an unsupported backbone is provided in the configurations.
        """

        # Load pre-trained MobileNetV3-Small model
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feature_extractor = model.features

        # Freeze feature extractor parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False

        # Custom classifier
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor and classifier.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images.

        Returns:
            torch.Tensor: Predicted class scores (logits).
        """
        x = self.classifier(x)  # Classification head
        return x


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Executes a single training step, computes the loss, and stores batch outputs.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss, y_hat, y = self._step(batch)
        self.training_step_outputs.append({'predictions': y_hat, 'labels': y, 'loss': loss})
        return loss


    def on_train_epoch_end(self) -> None:
        """
        Computes and logs the average loss and accuracy at the end of each training epoch.
        """
        self._on_epoch_end(self.training_step_outputs, mode='train')


    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Executes a single validation step, computes the loss, and stores batch outputs.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss, y_hat, y = self._step(batch)
        self.validation_step_outputs.append({'predictions': y_hat, 'labels': y, 'loss': loss})
        return loss


    def on_validation_epoch_end(self):
        """
        Computes and logs the average loss and accuracy at the end of each validation epoch.
        """
        self._on_epoch_end(self.validation_step_outputs, mode='val')


    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Executes a single test step, computes the loss, and logs it.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss, _, _ = self._step(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    def on_test_epoch_end(self):
        """
        Computes and logs the average loss and accuracy at the end of each test epoch.
        """
        self._on_epoch_end(self.test_step_outputs, mode='test')


    def _step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a batch, runs it through the model, and computes the loss.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss, model predictions, and true labels.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y


    def configure_optimizers(self):
        """
        Instantiates the optimizer and scheduler using Hydra.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
            The optimizer and learning rate scheduler.
        """
        optimizer = instantiate(self.configs.optimizer, self.parameters())
        scheduler = instantiate(self.configs.scheduler, optimizer)
        return [optimizer], [scheduler]


    def _on_epoch_end(self, step_outputs: List[Dict[str, Any]], mode: str):
        """
        Aggregates batch outputs, computes epoch-level loss and accuracy, and logs them.

        Args:
            step_outputs (List[Dict[str, Any]]): List of batch outputs containing predictions, labels, and loss.
            mode (str): One of 'train', 'val', or 'test' indicating the current phase.
        """
        if not step_outputs:
            return

        # Aggregate predictions, labels, and losses
        predictions = torch.cat([x['predictions'] for x in step_outputs])
        labels = torch.cat([x['labels'] for x in step_outputs])
        avg_loss = torch.stack([x['loss'] for x in step_outputs]).mean()

        # Compute epoch accuracy
        epoch_acc = accuracy(predictions, labels, device=self.device)

        # Log epoch metrics
        self.log(f"{mode}_loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log(f"{mode}_acc_epoch", epoch_acc, prog_bar=True, on_epoch=True)

        # Clear stored batch outputs
        step_outputs.clear()