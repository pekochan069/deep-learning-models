from abc import ABC
from core.config import DiffusionConfig
from ..classification.base_model import History
from ..base_model import BaseModel


class DiffusionBaseModel(BaseModel, ABC):
    config: DiffusionConfig
    history: History

    def __init__(self, config: DiffusionConfig):
        super(DiffusionBaseModel, self).__init__("Diffusion")

        self.config = config
        self.history = History()
