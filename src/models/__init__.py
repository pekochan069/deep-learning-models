from core.config import Config
from .alex_net import AlexNet
from .example_cnn import ExampleCNN
from .le_net import LeNet


def get_model(config: Config):
    if config.model == "example_cnn":
        return ExampleCNN(config)
    elif config.model == "le_net":
        return LeNet(config)
    elif config.model == "alex_net":
        return AlexNet(config)
    raise ValueError(f"Unknown model name: {config.model}")
