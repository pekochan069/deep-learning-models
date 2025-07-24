from core.config import Config
from core.dataset import get_num_classes
from ..base_model import BaseModel


class UNet(BaseModel):
    def __init__(self, config: Config):
        super(UNet, self).__init__(config)

        self.num_classes = get_num_classes(config.dataset)
