import json
import logging
import os
from typing import Self
from pydantic import BaseModel

from .names import (
    dataset_names,
    model_names,
    loss_function_names,
    optimizer_names,
)

logger = logging.getLogger(__name__)


class Config(BaseModel):
    name: str
    model: model_names
    dataset: dataset_names
    batch_size: int
    shuffle: bool = False
    optimizer: optimizer_names
    optimizer_params: dict = {}
    loss_function: loss_function_names
    epochs: int

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
