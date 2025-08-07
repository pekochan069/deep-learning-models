from typing import final
from core.config import DiffusionConfig
from models.diffusion.base_model import DiffusionBaseModel


@final
class DDPM(DiffusionBaseModel):
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
