from abc import ABC, abstractmethod
import logging
from typing import TypeVar

import torch
import torch.nn as nn

from core.device import get_device

TrainEpochReturnType = TypeVar("TrainEpochReturnType")
ValidateEpochReturnType = TypeVar("ValidateEpochReturnType")
PredictReturnType = TypeVar("PredictReturnType")


class BaseModel(ABC, nn.Module):
    """Base class for all Deep Learning models. All base models should inherit from this class.

    Usage:
        class TaskBaseModel(BaseModel):
            def __init__(self, ...):
                super(TaskBaseModel, self).__init__("TaskName")
                # Initialize your model here

    Args:
        ABC (_type_): _description_
    """

    device: torch.device
    logger: logging.Logger

    def __init__(self, logger_name: str):
        super(BaseModel, self).__init__()
        self.device = get_device()
        self.logger = logging.getLogger(logger_name)

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def train_epoch(self, *args, **kwargs) -> TrainEpochReturnType:
        """Train the model for one epoch."""
        pass

    @abstractmethod
    def validate_epoch(self, *args, **kwargs) -> ValidateEpochReturnType:
        """Validate the model for one epoch."""
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> PredictReturnType:
        """Make predictions using the model."""
        pass

    @abstractmethod
    def summary(self, *args, **kwargs):
        """Print the model summary."""
        pass

    @abstractmethod
    def plot_history(self, show: bool = True, save: bool = True):
        """Plot the training history."""
        pass
