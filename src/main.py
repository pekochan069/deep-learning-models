import logging

from core.config import Config
from core.logger import init_logger
from core.train_pipeline import Pipeline

logger = logging.getLogger(__name__)


def main():
    init_logger("INFO")

    # config = Config(
    #     name="example_cnn",
    #     model="example_cnn",
    #     dataset="mnist",
    #     batch_size=64,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # Config.save_config(config)
    # config = Config.load_config("example_cnn")

    # config = Config(
    #     name="le_net",
    #     model="le_net",
    #     dataset="mnist",
    #     batch_size=64,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # Config.save_config(config)
    # config = Config.load_config("le_net")

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("alex_net")

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("vgg11")

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("res_net18")

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("res_net50")

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("vgg_net_16_cifar10")

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("res_net50_cifar10")
    # 71.44%

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("inception_v1_cifar10")
    # 70.78%

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("inception_v2_cifar10")
    # 74.07%

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("dense_net_cifar10")
    # 73.09%

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("dense_net121_cifar10")
    # 74.32%

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("mobile_net_cifar10")
    # 72.96%

    # config = Config(
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
    # Config.save_config(config)
    # config = Config.load_config("shuffle_net_v1_cifar10")

    config = Config(
        name="efficient_net_v1_b0_cifar10",
        model="efficient_net_v1_b0",
        dataset="cifar10",
        batch_size=128,
        shuffle=True,
        optimizer="rmsprop",
        optimizer_params={"lr": 0.016, "weight_decay": 1e-5},
        loss_function="cross_entropy",
        early_stopping=True,
        early_stopping_min_delta=0.05,
        epochs=100,
    )
    Config.save_config(config)
    # config = Config.load_config("efficient_net_v0_b1_cifar10")
    # 77.92%

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
