import logging
import math

import torch

from tqdm import tqdm

from core.config import Config
from core.dataset import get_dataset
from core.logger import init_logger
from models import get_model

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
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # Config.save_config(config)
    # config = Config.load_config("res_net50_cifar10")

    # config = Config(
    #     name="inception_v1_cifar10",
    #     model="inception_v1",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # Config.save_config(config)
    # config = Config.load_config("inception_v1_cifar10")

    # config = Config(
    #     name="inception_v2_cifar10",
    #     model="inception_v2",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # Config.save_config(config)
    # config = Config.load_config("inception_v2_cifar10")

    # config = Config(
    #     name="dense_net_cifar10",
    #     model="dense_net_cifar",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # Config.save_config(config)
    # config = Config.load_config("dense_net_cifar10")

    config = Config(
        name="dense_net121_cifar10",
        model="dense_net121",
        dataset="cifar10",
        batch_size=64,
        shuffle=True,
        optimizer="adam",
        optimizer_params={"lr": 0.001},
        loss_function="cross_entropy",
        epochs=15,
    )
    Config.save_config(config)
    # config = Config.load_config("dense_net121_cifar10")

    model = get_model(config)

    # ImageNet 데이터셋은 기본 transform이 자동으로 적용됩니다
    dataset = get_dataset(config.dataset, config.batch_size, config.shuffle)

    print(f"Dataset: {config.dataset}")
    print(f"Train dataset size: {len(dataset.train)}")
    print(f"Train data shape: {dataset.train.dataset[0][0].shape}")
    print(f"Training Steps: {len(dataset.train) * config.epochs}")

    model.summary((1, 3, 32, 32))
    model.fit(dataset.train)
    model.plot_history()
    model.save(config.name)

    # model.load(config.name)

    model.predict(dataset.test)


if __name__ == "__main__":
    main()
