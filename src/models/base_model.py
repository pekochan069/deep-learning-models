import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from core.device import get_device
from core.loss import get_loss_function
from core.optimizer import get_optimizer
from core.weights import load_model, save_model


class History:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []


class BaseModel(nn.Module):
    def __init__(self, config: Config):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.device = get_device()

        self.history = History()

    @abstractmethod
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        pass

    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader, loss_function: nn.Module) -> float:
        """Evaluate the model."""
        pass

    def save(self, name: str):
        """Save the model."""
        save_model(self, name)

    def load(self, name: str):
        """Load the model and update weights."""
        loaded_model = load_model(self, name)
        if loaded_model is None:
            self.logger.error(f"Model {name} not found.")
            return
        self.load_state_dict(loaded_model.state_dict())
        self.to(self.device)
        self.logger.info(f"Model {name} loaded successfully.")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None):
        self.logger.info(
            f"Training {self.config.model} on {self.config.dataset} dataset"
        )
        self.to(self.device)

        optimizer = get_optimizer(self.config.optimizer)(
            self.parameters(),
            **self.config.optimizer_params,
        )
        loss_function = get_loss_function(self.config.loss_function).to(self.device)

        for epoch in range(self.config.epochs):
            self.logger.info(f"Training epoch {epoch + 1}/{self.config.epochs}")

            epoch_loss = self.train_epoch(train_loader, optimizer, loss_function)

            self.history.train_loss.append(epoch_loss)

            if val_loader:
                epoch_val_loss = self.validate_epoch(val_loader, loss_function)
                self.history.val_loss.append(epoch_val_loss)
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} finished, "
                    f"Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} finished, Loss: {epoch_loss:.4f}"
                )

        self.logger.info("Training complete.")

    @abstractmethod
    def predict(self, data_loader: DataLoader):
        """Evaluate the model on the provided data loader."""
        pass
        # self.logger.info("Evaluating model...")
        # self.eval()

        # predictions = []
        # with torch.no_grad():
        #     for inputs, targets in tqdm(data_loader, desc="Evaluating"):
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)

        #         outputs = self(inputs)
        #         predictions.append(self.process_output(outputs))

        # self.logger.info("Evaluation complete.")
        # return torch.cat(predictions, dim=0)

    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        """Process the model output."""
        return output.cpu()

    def summary(self, input_size: tuple):
        summary(self, input_size=input_size)

    def plot_history(self):
        plt.plot(range(1, self.config.epochs + 1), self.history.train_loss, marker="o")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(f"images/{self.config.name}_training_loss.png")
        plt.show()
