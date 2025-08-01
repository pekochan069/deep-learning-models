from core.config import GANConfig
from .gan import GAN
from .srgan import SRGAN
from .esrgan import ESRGAN


def get_gan_model(config: GANConfig):
    match config.model:
        case "gan":
            return GAN(config)
        case "srgan":
            return SRGAN(config)
        case "esrgan":
            return ESRGAN(config)
