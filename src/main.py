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
        dataset="imagenet",
        batch_size=64,
        shuffle=True,
        optimizer="adam",
        optimizer_params={"lr": 0.001},
        loss_function="cross_entropy",
        epochs=15,
    )
    Config.save_config(config)
    # config = Config.load_config("alex_net")

    model = get_model(config)

    train, test = get_dataset(config.dataset)
    train_loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=config.shuffle
    )
    test_loader = DataLoader(test)

    model.summary(input_size=(1, 1, 28, 28))
    model.fit(train_loader)
    model.plot_history()

    model.save(config.name)

    # model.load(config.name)

    predictions = model.predict(test_loader)
    predictions = torch.argmax(predictions, dim=1)
    accuracy = (predictions == test.targets).float().mean().item() * 100  # type: ignore
    logger.info(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
