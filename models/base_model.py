import torch
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig
from typing import Dict, Any, List, Tuple
from utils.metrics import accuracy
from hydra.utils import instantiate
from abc import ABC, abstractmethod
from utils.utils import init_weights

class BaseFlowerClassifier(L.LightningModule):
    """
    Abstract base classifier for flower classification.

    This model:
    - Supports multiple architectures, including both transfer learning and models trained from scratch.
    - Implements training, validation, and testing logic with logging.

    Args:
        configs (DictConfig): Configuration containing model, optimizer, and scheduler parameters.
    """

    def __init__(self, configs: DictConfig):
        """
        Initializes the classifier, initialize training configuration

        Args:
            configs (DictConfig): A Hydra configuration object containing model and training configurations.
        """
        super(BaseFlowerClassifier, self).__init__()
        self.configs = configs

        # Initialize the model
        self.classifier = self.build_model()

        # Initialize the weights
        init_weights(self.classifier, method="kaiming")  # Apply Kaiming initialization

        # Loss function (instantiated via Hydra)
        self.loss_fn = instantiate(self.configs.loss)

        # Track outputs per batch for epoch-level metrics
        self.epoch_outputs = {"train": [], "val": [], "test": []}


    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Abstract method to build the model.

        This method should:
        - Load a feature extractor (pretrained or not).
        - Optionally freeze feature extractor layers.
        - Add a classification head.
        - Or define a new model architecture design

        Returns:
            nn.Module: The constructed model.
        """
        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images.

        Returns:
            torch.Tensor: Predicted class scores (logits).
        """
        return self.classifier(x)  # Classification head


    def _step(self, batch: Any, mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a batch, runs it through the model, and computes the loss.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).
            mode (str): One of 'train', 'val', or 'test', indicating the current phase.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss, model score predictions, and true labels.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        # Store batch outputs for epoch-level metrics
        self.epoch_outputs[mode].append({'predictions': y_hat, 'labels': y, 'loss': loss})

        # Log loss per batch
        self.log(f"{mode}_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss, y_hat, y


    def _on_epoch_end(self, mode: str):
        """
        Aggregates batch outputs, computes epoch-level loss and accuracy, and logs them.

        Args:
            mode (str): One of 'train', 'val', or 'test' indicating the current phase.
        """
        outputs = self.epoch_outputs[mode]

        if not outputs:
            return

        # Aggregate predictions, labels, and losses
        predictions = torch.cat([x['predictions'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Compute accuracy
        epoch_acc = accuracy(predictions, labels, device=self.device)

        # Log epoch metrics
        self.log(f"{mode}_loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log(f"{mode}_acc_epoch", epoch_acc, prog_bar=True, on_epoch=True)

        # Clear stored batch outputs
        self.epoch_outputs[mode].clear()


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Executes a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        return self._step(batch, mode="train")[0]


    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Executes a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        return self._step(batch, mode="val")[0]


    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Executes a single test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch containing (images, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        return self._step(batch, mode="test")[0]


    def on_train_epoch_end(self) -> None:
        """
        Computes and logs the average loss and accuracy at the end of each training epoch.
        """
        self._on_epoch_end(mode='train')


    def on_validation_epoch_end(self):
        """
        Computes and logs the average loss and accuracy at the end of each validation epoch.
        """
        self._on_epoch_end(mode='val')


    def on_test_epoch_end(self):
        """
        Computes and logs the average loss and accuracy at the end of each test epoch.
        """
        self._on_epoch_end(mode='test')


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

