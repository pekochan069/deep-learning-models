from core.config import GANConfig
from .gan import GAN


def get_gan_model(config: GANConfig):
    match config.model:
        case "gan":
            return GAN(config)
