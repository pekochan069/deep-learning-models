from core.config import Config
from .alex_net import AlexNet
from .example_cnn import ExampleCNN
from .le_net import LeNet
from .vgg_net import VGGNet11, VGGNet13, VGGNet16, VGGNet19
from .res_net import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .inception import InceptionV1, InceptionV2, InceptionV3


def get_model(config: Config):
    match config.model:
        case "example_cnn":
            return ExampleCNN(config)
        case "le_net":
            return LeNet(config)
        case "alex_net":
            return AlexNet(config)
        case "vgg_net11":
            return VGGNet11(config)
        case "vgg_net13":
            return VGGNet13(config)
        case "vgg_net16":
            return VGGNet16(config)
        case "vgg_net19":
            return VGGNet19(config)
        case "res_net18":
            return ResNet18(config)
        case "res_net34":
            return ResNet34(config)
        case "res_net50":
            return ResNet50(config)
        case "res_net101":
            return ResNet101(config)
        case "res_net152":
            return ResNet152(config)
        case "inception_v1":
            return InceptionV1(config)
        case "inception_v2":
            return InceptionV2(config)
        case "inception_v3":
            return InceptionV3(config)
        case _:
            raise ValueError(f"Unknown model name: {config.model}")
