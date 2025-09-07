import gc
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, final, override

import torch

from core.config import ClassificationConfig, DiffusionConfig, GANConfig
from core.dataset import TrainableDataset, get_dataset
from core.registry import ModelRegistry
from models.classification import get_classification_model
from models.classification.base_model import ClassificationBaseModel
from models.diffusion import get_diffusion_model
from models.diffusion.base_model import DiffusionBaseModel
from models.gan import get_gan_model
from models.gan.base_model import GANBaseModel


class Pipeline(metaclass=ABCMeta):
    logger: logging.Logger

    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)

    def clean(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()
        _ = gc.collect()
        self.logger.info("Cache cleared and garbage collection invoked.")

    @abstractmethod
    def train(self):
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate the model."""
        pass

    @abstractmethod
    def run(self):
        """Run the complete pipeline: train and evaluate."""
        pass


@final
class ClassificationPipeline(Pipeline):
    config: ClassificationConfig
    model: ClassificationBaseModel
    dataset: TrainableDataset

    def __init__(self, config: ClassificationConfig, load: bool = True):
        super(ClassificationPipeline, self).__init__(
            logger_name="ClassificationPipeline"
        )
        self.config = config
        self.model = get_classification_model(self.config)
        self.dataset = get_dataset(self.config)

        if load:
            self.model.load()

        self.model.to_device()

    @override
    def train(self):
        """Execute the complete training pipeline."""

        self.logger.info(f"Starting training pipeline for {self.config.name}")
        self.logger.info(f"Dataset: {self.config.dataset}")
        self.logger.info(f"Train data shape: {self.dataset.train.dataset[0][0].shape}")
        self.logger.info(f"Training Steps per Epoch: {len(self.dataset.train)}")
        self.logger.info(
            f"Training Steps: {len(self.dataset.train) * self.config.epochs}"
        )

        # Display model summary
        if self.config.dataset in ["mnist", "fashion_mnist"]:
            input_size = (1, 1, 28, 28)
        elif self.config.dataset in ["padded_mnist"]:
            input_size = (1, 1, 32, 32)
        elif self.config.dataset in ["cifar10", "cifar100"]:
            input_size = (1, 3, 32, 32)
        else:  # imagenet, mini_imagenet
            input_size = (1, 3, 224, 224)

        self.model.summary(input_size)

        # Train the model
        self.model.fit(self.dataset.train, self.dataset.val)

        # Plot training history
        self.model.plot_history(self.config.show_plot, self.config.save_plot)

        # Save the trained model
        self.model.save()

        # Evaluate on test set
        self.model.predict(self.dataset.test)

        self.logger.info(f"Training pipeline completed for {self.config.name}")

    @override
    def evaluate(self):
        """Evaluate a trained model."""
        self.logger.info(f"Loading and evaluating model {self.config.name}")

        # Load the trained model
        self.model.load()

        # Evaluate on test set
        self.model.predict(self.dataset.test)

        self.logger.info(f"Evaluation completed for {self.config.name}")

    @override
    def run(self):
        """Run the complete pipeline: train and evaluate."""
        self.train()
        self.evaluate()
        self.logger.info(f"Full pipeline completed for {self.config.name}")


class GANPipeline(Pipeline):
    config: GANConfig
    model: GANBaseModel
    dataset: TrainableDataset

    def __init__(
        self,
        config: GANConfig,
        target_transform: Callable[[Any], torch.Tensor] | None = None,
        input_transform: Callable[[Any], torch.Tensor] | None = None,
        load: bool = True,
    ):
        super(GANPipeline, self).__init__(logger_name="GANPipeline")
        self.config = config
        self.model = get_gan_model(self.config)
        self.dataset = get_dataset(
            self.config, transform=target_transform, input_transform=input_transform
        )

        if load:
            self.model.load()

        self.model.to_device()

    def pretrain(self):
        self.logger.info(f"Starting pre-training for GAN model {self.config.name}")
        self.logger.info(f"Pretrain Dataset: {self.config.dataset}")
        self.logger.info(
            f"Pretrain data shape: {self.dataset.train.dataset[0][0].shape}"
        )
        self.logger.info(f"Pretrain Steps per Epoch: {len(self.dataset.train)}")
        self.logger.info(
            f"Pretrain Steps: {len(self.dataset.train) * self.config.pretrain_epochs}"
        )

        self.model.pretrain(self.dataset.train)

        self.model.plot_pretrain_history(
            self.config.show_pretrained_plot, self.config.save_pretrained_plot
        )

        if self.config.save_pretrained:
            self.model.save_pretrained()

        self.logger.info(f"Pre-training completed for GAN model {self.config.name}")

    @override
    def train(self):
        """Execute the complete GAN training pipeline."""

        self.logger.info(f"Starting GAN training pipeline for {self.config.name}")
        self.logger.info(f"Dataset: {self.config.dataset}")
        self.logger.info(f"Train data shape: {self.dataset.train.dataset[0][0].shape}")
        self.logger.info(f"Training Steps per Epoch: {len(self.dataset.train)}")
        self.logger.info(
            f"Training Steps: {len(self.dataset.train) * self.config.epochs}"
        )

        # Display model summary
        if self.config.dataset in ["mnist", "fashion_mnist"]:
            discriminator_input_size = (1, 1, 28, 28)
            generator_input_size = (1, 1, 28, 28)
        elif self.config.dataset in ["cifar10", "cifar100"]:
            discriminator_input_size = (1, 3, 32, 32)
            generator_input_size = (1, 3, 32, 32)
        elif self.config.dataset in ["imagenet", "mini_imagenet"]:
            discriminator_input_size = (1, 3, 224, 224)
            generator_input_size = (1, 3, 224, 224)
        elif self.config.dataset == "df2k_ost":
            discriminator_input_size = (1, 3, 256, 256)
            generator_input_size = (1, 3, 64, 64)
        elif self.config.dataset == "df2k_ost_small":
            discriminator_input_size = (1, 3, 96, 96)
            generator_input_size = (1, 3, 24, 24)
        else:
            discriminator_input_size = (1, 3, 64, 64)
            generator_input_size = (1, 3, 64, 64)
        self.model.summary(discriminator_input_size, generator_input_size)

        # Train the model
        self.model.fit(self.dataset.train, self.dataset.val)

        # Plot training history
        self.model.plot_history(self.config.show_plot, self.config.save_plot)

        # Save the trained model
        self.model.save()

        # Evaluate on test set
        self.model.predict(self.dataset.test)

        self.logger.info(f"GAN training pipeline completed for {self.config.name}")

    @override
    def evaluate(self):
        """Evaluate a trained GAN model."""
        self.logger.info(f"Loading and evaluating GAN model {self.config.name}")

        # Load the trained model
        self.model.load()

        # Evaluate on test set
        self.model.predict(self.dataset.test)

        self.logger.info(f"GAN evaluation completed for {self.config.name}")

    @override
    def run(self):
        """Run the complete GAN pipeline: train and evaluate."""
        if self.config.load_pretrained:
            pretrained_loaded = self.model.load_pretrained()

            if not pretrained_loaded and self.config.pretrain:
                self.logger.info("Pretrained model not found, starting pre-training...")
                self.pretrain()
        else:
            if self.config.pretrain:
                self.pretrain()
        self.train()
        self.evaluate()
        self.logger.info(f"Full GAN pipeline completed for {self.config.name}")


@final
class DiffusionPipeline(Pipeline):
    config: DiffusionConfig
    model: DiffusionBaseModel
    dataset: TrainableDataset

    def __init__(self, config: DiffusionConfig, load: bool = True):
        super(DiffusionPipeline, self).__init__("DiffusionPipeline")

        self.config = config
        self.model = get_diffusion_model(self.config)
        self.dataset = get_dataset(self.config)

        if load:
            self.model.load()

        self.model.to_device()

    @override
    def train(self):
        """Execute the complete training pipeline."""
        # Display model summary
        if self.config.dataset in ["mnist", "fashion_mnist"]:
            input_size = (1, 1, 28, 28)
        elif self.config.dataset in ["padded_mnist"]:
            input_size = (1, 1, 32, 32)
        elif self.config.dataset in ["cifar10", "cifar100"]:
            input_size = (1, 3, 32, 32)
        else:  # imagenet, mini_imagenet
            input_size = (1, 3, 224, 224)
        self.model.summary(input_size)

        self.logger.info(f"Starting training pipeline for {self.config.name}")
        self.logger.info(f"Dataset: {self.config.dataset}")
        self.logger.info(f"Train data shape: {self.dataset.train.dataset[0][0].shape}")
        self.logger.info(f"Training Steps per Epoch: {len(self.dataset.train)}")
        self.logger.info(
            f"Training Steps: {len(self.dataset.train) * self.config.epochs}"
        )

        # Train the model
        self.model.fit(self.dataset.train, self.dataset.val)

        # Plot training history
        self.model.plot_history(self.config.show_plot, self.config.save_plot)

        # Save the trained model
        self.model.save()

        self.logger.info(f"Training pipeline completed for {self.config.name}")

    @override
    def evaluate(self, *args, **kwargs):
        """Evaluate a trained model."""
        self.logger.info(f"Loading and evaluating model {self.config.name}")

        # Evaluate on test set
        self.model.predict(*args, **kwargs)

        self.logger.info(f"Evaluation completed for {self.config.name}")

    @override
    def run(self, *args, **kwargs):
        """Run the complete pipeline: train and evaluate."""
        self.train()
        self.evaluate(*args, **kwargs)
        self.logger.info(f"Full pipeline completed for {self.config.name}")
