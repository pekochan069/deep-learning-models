import logging

from torchvision import transforms

from core.config import GANConfig
from core.logger import init_logger
from core.train_pipeline import GANPipeline

logger = logging.getLogger(__name__)


def main():
    init_logger("INFO")

    # config = CNNConfig(
    #     name="example_cnn",
    #     model="example_cnn",
    #     dataset="mnist",
    #     batch_size=64,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("example_cnn")

    # config = CNNConfig(
    #     name="le_net",
    #     model="le_net",
    #     dataset="mnist",
    #     batch_size=64,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("le_net")

    # config = CNNConfig(
    #     name="alex_net",
    #     model="alex_net",
    #     dataset="mini_imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("alex_net")

    # config = CNNConfig(
    #     name="vgg11",
    #     model="vgg11",
    #     dataset="imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("vgg11")

    # config = CNNConfig(
    #     name="res_net18",
    #     model="res_net18",
    #     dataset="mini_imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=3,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("res_net18")

    # config = CNNConfig(
    #     name="res_net50",
    #     model="res_net50",
    #     dataset="mini_imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=3,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("res_net50")

    # config = CNNConfig(
    #     name="vgg_net_16_cifar10",
    #     model="vgg_net16",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("vgg_net_16_cifar10")

    # config = CNNConfig(
    #     name="res_net50_cifar10",
    #     model="res_net50",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("res_net50_cifar10")
    # 71.44%

    # config = CNNConfig(
    #     name="inception_v1_cifar10",
    #     model="inception_v1",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("inception_v1_cifar10")
    # 70.78%

    # config = CNNConfig(
    #     name="inception_v2_cifar10",
    #     model="inception_v2",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("inception_v2_cifar10")
    # 74.07%

    # config = CNNConfig(
    #     name="dense_net_cifar10",
    #     model="dense_net_cifar",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     epochs=30,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("dense_net_cifar10")
    # 73.09%

    # config = CNNConfig(
    #     name="dense_net121_cifar10",
    #     model="dense_net121",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("dense_net121_cifar10")
    # 74.32%

    # config = CNNConfig(
    #     name="mobile_net_cifar10",
    #     model="mobile_net",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     epochs=30,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("mobile_net_cifar10")
    # 72.96%

    # config = CNNConfig(
    #     name="shuffle_net_v1_cifar10",
    #     model="shuffle_net_v1",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("shuffle_net_v1_cifar10")

    # config = CNNConfig(
    #     name="efficient_net_v1_b0_cifar10",
    #     model="efficient_net_v1_b0",
    #     dataset="cifar10",
    #     batch_size=128,
    #     shuffle=True,
    #     optimizer="rmsprop",
    #     optimizer_params={"lr": 0.016, "weight_decay": 1e-5},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=100,
    # )
    # CNNConfig.save_config(config)
    # config = CNNConfig.load_config("efficient_net_v0_b1_cifar10")
    # 77.92%

    # pipeline = CNNPipeline(config)
    # pipeline.run()

    config = GANConfig(
        name="gan_mnist",
        model="gan",
        dataset="mnist",
        batch_size=128,
        shuffle=True,
        g_optimizer="adam",
        g_optimizer_params={"lr": 0.0002, "betas": (0.5, 0.999)},
        d_optimizer="adam",
        d_optimizer_params={"lr": 0.000075, "betas": (0.5, 0.999)},
        g_loss_function="bce_with_logits",
        d_loss_function="gan_discriminator_loss",
        epochs=30,
        real_label=0.95,
    )
    GANConfig.save_config(config)
    # config = GANConfig.load_config("gan_mnist")

    pipeline = GANPipeline(
        config,
        transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
