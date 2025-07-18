from typing import Callable, Literal
from torchvision import datasets, transforms


dataset_names = Literal["mnist", "cifar10", "cifar100", "fashion_mnist", "imagenet"]


def get_dataset(
    name: dataset_names, transform: Callable | None = transforms.ToTensor()
) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    match name:
        case "mnist":
            return mnist(transform)
        case "cifar10":
            return cifar10(transform)
        case "cifar100":
            return cifar100(transform)
        case "fashion_mnist":
            return fashion_mnist(transform)
        case "imagenet":
            return imagenet(transform)
        case _:
            raise ValueError(f"Unknown dataset: {name}")


def mnist(transform: Callable | None = transforms.ToTensor()):
    train = datasets.MNIST("data", download=True, transform=transform, train=True)
    test = datasets.MNIST("data", download=True, transform=transform, train=False)

    return train, test


def cifar10(transform: Callable | None = transforms.ToTensor()):
    train = datasets.CIFAR10("data", download=True, transform=transform, train=True)
    test = datasets.CIFAR10("data", download=True, transform=transform, train=False)

    return train, test


def cifar100(transform: Callable | None = transforms.ToTensor()):
    train = datasets.CIFAR100("data", download=True, transform=transform, train=True)
    test = datasets.CIFAR100("data", download=True, transform=transform, train=False)

    return train, test


def fashion_mnist(transform: Callable | None = transforms.ToTensor()):
    train = datasets.FashionMNIST(
        "data", download=True, transform=transform, train=True
    )
    test = datasets.FashionMNIST(
        "data", download=True, transform=transform, train=False
    )

    return train, test


def imagenet(transform: Callable | None = transforms.ToTensor()):
    train = datasets.ImageNet("data", download=True, transform=transform, split="train")
    test = datasets.ImageNet("data", download=True, transform=transform, split="val")

    return train, test
