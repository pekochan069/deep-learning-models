from core.config import Config
from .alex_net import AlexNet
from .example_cnn import ExampleCNN
from .le_net import LeNet
from .vggnet import VGGNet11


def get_model(config: Config):
    if config.model == "example_cnn":
        return ExampleCNN(config)
    elif config.model == "le_net":
        return LeNet(config)
    elif config.model == "alex_net":
        return AlexNet(config)
    elif config.model == "vgg11":
        return VGGNet11(config)
    raise ValueError(f"Unknown model name: {config.model}")
