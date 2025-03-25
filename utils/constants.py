from enum import Enum
from typing import Tuple


class DatasetMode(str, Enum):
    """
    Enumeration representing different dataset modes.

    Attributes:
        TRAIN (str): Mode for training dataset.
        VALIDATION (str): Mode for validation dataset.
        TEST (str): Mode for testing dataset.

    Methods:
        list() -> Tuple[DatasetMode, ...]: Returns all dataset modes as DatasetMode instances.
        values() -> Tuple[str, ...]: Returns all dataset modes as string values.
    """
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


    @classmethod
    def list(cls) -> Tuple["DatasetMode", ...]:
        """Returns all dataset modes as DatasetMode instances."""
        return tuple(cls)  # Dynamically gets all enum members


    @classmethod
    def values(cls) -> Tuple[str, ...]:
        """Returns all dataset modes as string values."""
        return tuple(mode.value for mode in cls)