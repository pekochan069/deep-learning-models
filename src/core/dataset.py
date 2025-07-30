from dataclasses import dataclass
import logging
from typing import Callable, final, override
import torch
from torchvision import datasets, transforms  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import random_split, Dataset as TorchDataset
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset  # pyright: ignore[reportMissingTypeStubs]

from core.config import Config

from .names import DatasetName

logger = logging.getLogger(__name__)


def get_num_classes(dataset_name: DatasetName) -> int:
    """데이터셋별 클래스 수를 반환합니다."""
    match dataset_name:
        case "mnist" | "fashion_mnist" | "cifar10":
            return 10
        case "cifar100" | "mini_imagenet":
            return 100
        case "imagenet":
            return 1000


@dataclass
class TrainableDataset:
    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None


@final
class HFDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, hf_dataset: Dataset, transform: Callable | None = None):  # pyright: ignore[reportMissingTypeArgument]
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset(
    config: Config,
    transform: Callable | None = None,  # pyright: ignore[reportMissingTypeArgument]
) -> TrainableDataset:
    name = config.dataset
    batch_size = config.batch_size
    shuffle = config.shuffle
    match name:
        case "mnist":
            return mnist(batch_size, shuffle, transform)
        case "cifar10":
            return cifar10(batch_size, shuffle, transform)
        case "cifar100":
            return cifar100(batch_size, shuffle, transform)
        case "fashion_mnist":
            return fashion_mnist(batch_size, shuffle, transform)
        case "imagenet":
            return imagenet(batch_size, shuffle, transform)
        case "mini_imagenet":
            return mini_imagenet(batch_size, shuffle, transform)


def mnist(batch_size: int, shuffle: bool, transform: Callable | None = None):  # pyright: ignore[reportMissingTypeArgument]
    if transform is None:
        logger.info("Using default transform for MNIST")
        transform = transforms.ToTensor()
    else:
        logger.info("Using custom transform for MNIST")

    train = datasets.MNIST("data", download=True, transform=transform, train=True)
    test = datasets.MNIST("data", download=True, transform=transform, train=False)

    return TrainableDataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def cifar10(batch_size: int, shuffle: bool, transform: Callable | None = None):  # pyright: ignore[reportMissingTypeArgument]
    if transform is None:
        logger.info("Using default transform for CIFAR-10")
        transform = transforms.ToTensor()
    else:
        logger.info("Using custom transform for CIFAR-10")

    train = datasets.CIFAR10("data", download=True, transform=transform, train=True)
    test = datasets.CIFAR10("data", download=True, transform=transform, train=False)

    validation = int(0.1 * len(train))
    train_set, val_set = random_split(train, [len(train) - validation, validation])

    return TrainableDataset(
        train=DataLoader(train_set, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
        val=DataLoader(val_set, batch_size=batch_size, shuffle=False),
    )


def cifar100(batch_size: int, shuffle: bool, transform: Callable | None = None):  # pyright: ignore[reportMissingTypeArgument]
    if transform is None:
        logger.info("Using default transform for CIFAR-100")
        transform = transforms.ToTensor()
    else:
        logger.info("Using custom transform for CIFAR-100")

    train = datasets.CIFAR100("data", download=True, transform=transform, train=True)
    test = datasets.CIFAR100("data", download=True, transform=transform, train=False)

    return TrainableDataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def fashion_mnist(batch_size: int, shuffle: bool, transform: Callable | None = None):  # pyright: ignore[reportMissingTypeArgument]
    if transform is None:
        logger.info("Using default transform for Fashion MNIST")
        transform = transforms.ToTensor()
    else:
        logger.info("Using custom transform for Fashion MNIST")

    train = datasets.FashionMNIST(
        "data", download=True, transform=transform, train=True
    )
    test = datasets.FashionMNIST(
        "data", download=True, transform=transform, train=False
    )

    return TrainableDataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def imagenet(batch_size: int, shuffle: bool, transform: Callable | None = None):  # pyright: ignore[reportMissingTypeArgument]
    # ImageNet에 적합한 기본 transform 설정
    if transform is None:
        logger.info("Using default transform for ImageNet")
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # 짧은 변을 256으로 resize
                transforms.CenterCrop(224),  # 224x224로 center crop
                transforms.Lambda(
                    lambda x: x.convert("RGB") if x.mode != "RGB" else x  # pyright: ignore[reportUnknownLambdaType]
                ),  # Grayscale을 RGB로 변환
                transforms.ToTensor(),  # 먼저 텐서로 변환
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet 정규화
            ]
        )
    else:
        logger.info("Using custom transform for ImageNet")

    # train = datasets.ImageNet("data", transform=transform, split="train")
    val = datasets.ImageNet("data/IMAGENET", transform=transform, split="val")

    # split val dataset to train and test 80/20
    val_size = len(val)
    train_size = int(0.8 * val_size)
    test_size = val_size - train_size
    split = random_split(val, [train_size, test_size])

    train = split[0]
    test = split[1]

    return TrainableDataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def mini_imagenet(batch_size: int, shuffle: bool, transform: Callable | None = None):  # pyright: ignore[reportMissingTypeArgument]
    # Mini-ImageNet에 적합한 기본 transform 설정
    if transform is None:
        logger.info("Using default transform for Mini-ImageNet")
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # 짧은 변을 256으로 resize
                transforms.CenterCrop(224),  # 224x224로 center crop
                transforms.Lambda(
                    lambda x: x.convert("RGB") if x.mode != "RGB" else x  # pyright: ignore[reportUnknownLambdaType]
                ),  # Grayscale을 RGB로 변환
                transforms.ToTensor(),  # 먼저 텐서로 변환
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet 정규화
            ]
        )
    else:
        logger.info("Using custom transform for Mini-ImageNet")

    dataset: DatasetDict = load_dataset("timm/mini-imagenet")  # pyright: ignore[reportAssignmentType]

    # 각 split을 PyTorch 데이터셋으로 변환
    train_dataset = HFDataset(dataset["train"], transform)
    test_dataset = HFDataset(dataset["test"], transform)
    val_dataset = HFDataset(dataset["validation"], transform)

    return TrainableDataset(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        val=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )
