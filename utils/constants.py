from enum import Enum

class DatasetMode(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def list(cls):
        return [cls.TRAIN, cls.VAL, cls.TEST]