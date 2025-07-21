from core.config import Config
from .alex_net import AlexNet
from .example_cnn import ExampleCNN
from .le_net import LeNet
from .vgg_net import VGGNet11


def get_model(config: Config):
    if config.model == "example_cnn":
        return ExampleCNN(config)
    elif config.model == "le_net":
        return LeNet(config)
    elif config.model == "alex_net":
        return AlexNet(config)
    elif config.model == "vgg_net11":
        return VGGNet11(config)
    elif config.model == "vgg_net13":
        return VGGNet11(config)
    elif config.model == "vgg_net16":
        return VGGNet11(config)
    elif config.model == "vgg_net19":
        return VGGNet11(config)
    raise ValueError(f"Unknown model name: {config.model}")
