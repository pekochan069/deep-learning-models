from core.config import Config
from .alex_net import AlexNet
from .example_cnn import ExampleCNN
from .le_net import LeNet
from .vgg_net import VGGNet11, VGGNet13, VGGNet16, VGGNet19
from .res_net import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .inception import InceptionV1, InceptionV2
from .dense_net import DenseNetCifar, DenseNet121, DenseNet169, DenseNet201, DenseNet264
from .mobile_net import MobileNet
from .shuffle_net import ShuffleNetV1
from .efficient_net import (
    EfficientNetV1B0,
    EfficientNetV1B1,
    EfficientNetV1B2,
    EfficientNetV1B3,
    EfficientNetV1B4,
    EfficientNetV1B5,
    EfficientNetV1B6,
    EfficientNetV1B7,
    EfficientNetV1B8,
    EfficientNetV1L2,
)


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
        case "dense_net_cifar":
            return DenseNetCifar(config)
        case "dense_net121":
            return DenseNet121(config)
        case "dense_net169":
            return DenseNet169(config)
        case "dense_net201":
            return DenseNet201(config)
        case "dense_net264":
            return DenseNet264(config)
        case "mobile_net":
            return MobileNet(config)
        case "shuffle_net_v1":
            return ShuffleNetV1(config)
        case "efficient_net_v1_b0":
            return EfficientNetV1B0(config)
        case "efficient_net_v1_b1":
            return EfficientNetV1B1(config)
        case "efficient_net_v1_b2":
            return EfficientNetV1B2(config)
        case "efficient_net_v1_b3":
            return EfficientNetV1B3(config)
        case "efficient_net_v1_b4":
            return EfficientNetV1B4(config)
        case "efficient_net_v1_b5":
            return EfficientNetV1B5(config)
        case "efficient_net_v1_b6":
            return EfficientNetV1B6(config)
        case "efficient_net_v1_b7":
            return EfficientNetV1B7(config)
        case "efficient_net_v1_b8":
            return EfficientNetV1B8(config)
        case "efficient_net_v1_l2":
            return EfficientNetV1L2(config)
        case _:
            raise ValueError(f"Unknown model name: {config.model}")
