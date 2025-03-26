from dataclasses import dataclass, field
from typing import Any, Dict, List
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class ModelConfig:
    in_features: int
    num_classes: int
    bias: bool


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.

    Attributes:
        _target_ (str): The target function/class to be instantiated by Hydra (e.g., `torch.optim.AdamW`).
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
    """
    _target_: str
    lr: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    """
    Configuration for the learning rate scheduler.

    Attributes:
        _target_ (str): The target function/class to be instantiated by Hydra (e.g., `torch.optim.lr_scheduler.StepLR`).
        step_size (int): Number of epochs after which to decay the learning rate.
        gamma (float): Multiplicative factor for learning rate decay.
    """
    _target_: str
    step_size: int
    gamma: float


@dataclass
class LossConfig:
    """
    Configuration for the loss function.

    Attributes:
        _target_ (str): The target function/class to be instantiated by Hydra (e.g., `torch.nn.CrossEntropyLoss`).
    """
    _target_: str
    wights: Optional[Any] = None


@dataclass
class TrainingConfig:
    """
    Unified configuration schema for training components.

    Attributes:
        optimizer (OptimizerConfig): Configuration for the optimizer.
        scheduler (SchedulerConfig): Configuration for the learning rate scheduler.
        loss (LossConfig): Configuration for the loss function.
    """
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig


@dataclass
class TrainingConfig:
    optimizer: DictConfig
    scheduler: DictConfig
    epochs: int


@dataclass
class Configs:
    model: ModelConfig
    training: TrainingConfig


# Register the configurations with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="model_configs", node=Configs)

