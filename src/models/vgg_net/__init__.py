import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from core.dataset import get_num_classes
from models.base_model import BaseModel


class VGGNet11(BaseModel):
    def __init__(self, config: Config):
        super(VGGNet11, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.LocalResponseNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        self.train(False)

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader: DataLoader, loss_function: nn.Module):
        """validate the model."""
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                loss = loss_function(outputs, targets)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    def predict(self, data_loader: DataLoader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                predictions = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += (predictions == targets).sum().item()

        accuracy = (correct / total) * 100
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")


class VGGNet13(BaseModel):
    def __init__(self, config: Config):
        super(VGGNet13, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        self.train(False)

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader: DataLoader, loss_function: nn.Module):
        """validate the model."""
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                loss = loss_function(outputs, targets)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    def predict(self, data_loader: DataLoader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                predictions = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += (predictions == targets).sum().item()

        accuracy = (correct / total) * 100
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")


class VGGNet16(BaseModel):
    def __init__(self, config: Config):
        super(VGGNet16, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        self.train(False)

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader: DataLoader, loss_function: nn.Module):
        """validate the model."""
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                loss = loss_function(outputs, targets)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    def predict(self, data_loader: DataLoader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                predictions = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += (predictions == targets).sum().item()

        accuracy = (correct / total) * 100
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")


class VGGNet19(BaseModel):
    def __init__(self, config: Config):
        super(VGGNet19, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        self.train(False)

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader: DataLoader, loss_function: nn.Module):
        """validate the model."""
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                loss = loss_function(outputs, targets)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    def predict(self, data_loader: DataLoader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                predictions = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += (predictions == targets).sum().item()

        accuracy = (correct / total) * 100
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")
