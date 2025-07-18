from core.config import Config
from models import get_model


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.model = get_model(self.config)

    def train(self):
        pass
