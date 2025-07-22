from dataclasses import dataclass
from typing import Callable
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset as TorchDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from .names import dataset_names


def get_num_classes(dataset_name: dataset_names) -> int:
    """데이터셋별 클래스 수를 반환합니다."""
    num_classes = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "fashion_mnist": 10,
        "imagenet": 1000,
        "mini_imagenet": 100,
    }
    return num_classes.get(dataset_name, 1000)


@dataclass
class Dataset:
    train: DataLoader
    test: DataLoader
    val: DataLoader | None = None


class HFDataset(TorchDataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset(
    name: dataset_names,
    batch_size: int,
    shuffle: bool,
    transform: Callable | None = None,
) -> Dataset:
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
        case _:
            raise ValueError(f"Unknown dataset: {name}")


def mnist(batch_size, shuffle, transform: Callable | None = None):
    if transform is None:
        transform = transforms.ToTensor()

    train = datasets.MNIST("data", download=True, transform=transform, train=True)
    test = datasets.MNIST("data", download=True, transform=transform, train=False)

    return Dataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def cifar10(batch_size, shuffle, transform: Callable | None = None):
    if transform is None:
        transform = transforms.ToTensor()

    train = datasets.CIFAR10("data", download=True, transform=transform, train=True)
    test = datasets.CIFAR10("data", download=True, transform=transform, train=False)

    return Dataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def cifar100(batch_size, shuffle, transform: Callable | None = None):
    if transform is None:
        transform = transforms.ToTensor()

    train = datasets.CIFAR100("data", download=True, transform=transform, train=True)
    test = datasets.CIFAR100("data", download=True, transform=transform, train=False)

    return Dataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def fashion_mnist(batch_size, shuffle, transform: Callable | None = None):
    if transform is None:
        transform = transforms.ToTensor()

    train = datasets.FashionMNIST(
        "data", download=True, transform=transform, train=True
    )
    test = datasets.FashionMNIST(
        "data", download=True, transform=transform, train=False
    )

    return Dataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def imagenet(batch_size, shuffle, transform: Callable | None = None):
    # ImageNet에 적합한 기본 transform 설정
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # 짧은 변을 256으로 resize
                transforms.CenterCrop(224),  # 224x224로 center crop
                transforms.Lambda(
                    lambda x: x.convert("RGB") if x.mode != "RGB" else x
                ),  # Grayscale을 RGB로 변환
                transforms.ToTensor(),  # 먼저 텐서로 변환
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet 정규화
            ]
        )

    # train = datasets.ImageNet("data", transform=transform, split="train")
    val = datasets.ImageNet("data/IMAGENET", transform=transform, split="val")

    # split val dataset to train and test 80/20
    val_size = len(val)
    train_size = int(0.8 * val_size)
    test_size = val_size - train_size
    split = random_split(val, [train_size, test_size])

    train = split[0]
    test = split[1]

    return Dataset(
        train=DataLoader(train, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def mini_imagenet(batch_size, shuffle, transform: Callable | None = None):
    # Mini-ImageNet에 적합한 기본 transform 설정
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # 짧은 변을 256으로 resize
                transforms.CenterCrop(224),  # 224x224로 center crop
                transforms.Lambda(
                    lambda x: x.convert("RGB") if x.mode != "RGB" else x
                ),  # Grayscale을 RGB로 변환
                transforms.ToTensor(),  # 먼저 텐서로 변환
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet 정규화
            ]
        )

    dataset = load_dataset("timm/mini-imagenet")

    # 각 split을 PyTorch 데이터셋으로 변환
    train_dataset = HFDataset(dataset["train"], transform)
    test_dataset = HFDataset(dataset["test"], transform)
    val_dataset = HFDataset(dataset["validation"], transform)

    return Dataset(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        val=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )
