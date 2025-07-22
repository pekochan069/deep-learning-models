import logging

import torch
from torch.utils.data import DataLoader

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

    config = Config(
        name="alex_net",
        model="alex_net",
        dataset="mini_imagenet",
        batch_size=64,
        shuffle=True,
        optimizer="adam",
        optimizer_params={"lr": 0.001},
        loss_function="cross_entropy",
        epochs=15,
    )
    Config.save_config(config)
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

    model = get_model(config)

    # ImageNet 데이터셋은 기본 transform이 자동으로 적용됩니다
    dataset = get_dataset(config.dataset, config.batch_size, config.shuffle)

    print(f"Dataset: {config.dataset}")
    print(f"Train dataset size: {len(dataset.train)}")
    print(f"Train data shape: {dataset.train.dataset[0][0].shape}")

    model.summary((1, 3, 224, 224))
    model.fit(dataset.train)
    model.plot_history()

    model.save(config.name)

    # model.load(config.name)

    predictions = model.predict(dataset.test)
    predictions = torch.argmax(predictions, dim=1)
    accuracy = (predictions == dataset.test.targets).float().mean().item() * 100  # type: ignore
    logger.info(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
