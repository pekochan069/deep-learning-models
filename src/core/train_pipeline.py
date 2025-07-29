import gc
import logging

import torch

from core.config import Config
from core.dataset import get_dataset
from core.logger import init_logger
from models import get_model


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.model = get_model(self.config)
        self.dataset = get_dataset(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self):
        """Execute the complete training pipeline."""
        init_logger("INFO")

        self.logger.info(f"Starting training pipeline for {self.config.name}")
        self.logger.info(f"Dataset: {self.config.dataset}")
        self.logger.info(f"Train dataset size: {len(self.dataset.train)}")
        self.logger.info(f"Train data shape: {self.dataset.train.dataset[0][0].shape}")
        self.logger.info(
            f"Training Steps: {len(self.dataset.train) * self.config.epochs}"
        )

        # Display model summary
        if self.config.dataset in ["mnist", "fashion_mnist"]:
            input_size = (1, 1, 28, 28)
        elif self.config.dataset in ["cifar10", "cifar100"]:
            input_size = (1, 3, 32, 32)
        else:  # imagenet, mini_imagenet
            input_size = (1, 3, 224, 224)

        self.model.summary(input_size)

        # Train the model
        self.model.fit(self.dataset.train, self.dataset.val)

        # Plot training history
        self.model.plot_history()

        # Save the trained model
        self.model.save()

        # Evaluate on test set
        self.model.predict(self.dataset.test)

        self.logger.info(f"Training pipeline completed for {self.config.name}")
        self.clean()

    def evaluate(self):
        """Evaluate a trained model."""
        self.logger.info(f"Loading and evaluating model {self.config.name}")

        # Load the trained model
        self.model.load()

        # Evaluate on test set
        self.model.predict(self.dataset.test)

        self.logger.info(f"Evaluation completed for {self.config.name}")
        self.clean()

    def run(self):
        """Run the complete pipeline: train and evaluate."""
        self.train()
        self.evaluate()
        self.logger.info(f"Full pipeline completed for {self.config.name}")

    def clean(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()
        gc.collect()
        self.logger.info("Cache cleared and garbage collection invoked.")
