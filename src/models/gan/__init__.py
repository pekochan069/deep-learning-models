from core.config import GANConfig
from .gan import GAN
from .srgan import SRGAN


def get_gan_model(config: GANConfig):
    match config.model:
        case "gan":
            return GAN(config)
        case "srgan":
            return SRGAN(config)
