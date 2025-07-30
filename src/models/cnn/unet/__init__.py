from core.config import CNNConfig
from core.dataset import get_num_classes
from ..base_model import BaseCNNModel


class UNet(BaseCNNModel):
    def __init__(self, config: CNNConfig):
        super(UNet, self).__init__(config)

        self.num_classes = get_num_classes(config.dataset)
