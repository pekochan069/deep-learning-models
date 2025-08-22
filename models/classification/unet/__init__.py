from core.config import ClassificationConfig
from core.dataset import get_num_classes
from ..base_model import ClassificationBaseModel


class UNet(ClassificationBaseModel):
    def __init__(self, config: ClassificationConfig):
        super(UNet, self).__init__(config)

        self.num_classes = get_num_classes(config.dataset)
