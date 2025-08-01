import json
import logging
import os
from typing import Any, Literal, Self
from pydantic import BaseModel as PydanticBaseModel

from core.optimizer import OptimizerParams

from .names import (
    DatasetName,
    CNNModelName,
    GANModelName,
    LossFunctionName,
    OptimizerName,
)

logger = logging.getLogger(__name__)


class Config(PydanticBaseModel):
    name: str
    dataset: DatasetName
    batch_size: int
    shuffle: bool = False
    epochs: int
    early_stopping: bool = False
    early_stopping_monitor: Literal["val_loss", "train_loss"] = "val_loss"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    epoch_save: bool = False
    epoch_save_period: int = 1

    @classmethod
    def save_config(cls, config: Self):
        if not os.path.exists("configs"):
            logger.debug("Configs directory does not exist. Creating directory...")
            os.makedirs("configs")

        with open(f"configs/{config.name}.json", "w") as f:
            json.dump(config.model_dump(exclude_none=True), f, indent=4)
            logger.info(f"Config saved to configs/{config.name}.json")

    @classmethod
    def load_config(cls, name: str):
        with open(f"configs/{name}.json", "r") as f:
            config = cls.model_validate_json(f.read())
            logger.info(f"Config loaded from configs/{name}.json")

            return config


class CNNConfig(Config):
    model: CNNModelName
    model_params: dict[str, Any] = {}
    optimizer: OptimizerName
    optimizer_params: OptimizerParams
    loss_function: LossFunctionName


class GANConfig(Config):
    model: GANModelName
    model_params: dict[str, Any] = {}
    g_optimizer: OptimizerName
    g_optimizer_params: OptimizerParams
    d_optimizer: OptimizerName
    d_optimizer_params: OptimizerParams

    g_loss_function: LossFunctionName
    d_loss_function: LossFunctionName

    fake_label: float = 0.0
    real_label: float = 1.0
