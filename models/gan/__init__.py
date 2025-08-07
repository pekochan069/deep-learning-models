from core.config import GANConfig
from .gan import GAN
from .srgan import SRGAN
from .esrgan import ESRGAN
from .esrgan_plus import ESRGANPlus
from .real_esrgan import RealESRGAN


def get_gan_model(config: GANConfig):
    match config.model:
        case "gan":
            return GAN(config)
        case "srgan":
            return SRGAN(config)
        case "esrgan":
            return ESRGAN(config)
        case "esrgan_plus":
            return ESRGANPlus(config)
        case "real_esrgan":
            return RealESRGAN(config)
