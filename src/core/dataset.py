import logging
import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, final, override

import torch
import cv2
from huggingface_hub import hf_hub_download
from torchvision import datasets, transforms  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import random_split, Dataset as TorchDataset
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset  # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

from core.preprocess import bucket_crop_image, quad_crop_image

from .names import DatasetName
from core.config import Config
from core.fs import scan_files

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
        case _:
            return 0


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


@final
class UpscaleDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor]]):
    target_transform: Callable[[Any], torch.Tensor]
    z_transform: Callable[[Any], torch.Tensor]

    def __init__(
        self,
        path: str,
        target_transform: Callable[[Any], torch.Tensor] | None = None,
        z_transform: Callable[[Any], torch.Tensor] | None = None,
    ):
        if target_transform is None:
            target_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        if z_transform is None:
            z_transform = transforms.ToTensor()
        self.path = path
        self.target_transform = target_transform
        self.z_transform = z_transform
        self.data = list(scan_files(path))
        self.length = len(self.data)

    def __len__(self) -> int:
        return self.length

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
        input = cv2.resize(
            target,
            (target.shape[1] // 4, target.shape[0] // 4),
            interpolation=cv2.INTER_CUBIC,
        )

        input = self.z_transform(input)
        target = self.target_transform(target)

        return input, target


def get_dataset(
    config: Config, transform: Callable[[Any], torch.Tensor] | None = None
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
        case "df2k_ost":
            return get_df2k_ost(batch_size, shuffle, transform)
        case "df2k_ost_small":
            return get_df2k_ost_small(batch_size, shuffle, transform)


def mnist(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
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


def cifar10(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
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


def cifar100(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
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


def fashion_mnist(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
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


def imagenet(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
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


def mini_imagenet(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
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


def get_div2k(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
    """DIV2K 데이터셋을 로드합니다."""
    if transform is None:
        logger.info("Using default transform for DIV2K")
        transform = transforms.ToTensor()
    else:
        logger.info("Using custom transform for DIV2K")

    raw_train_dataset: Dataset = load_dataset(
        "eugenesiow/Div2k", "bicubic_x2", split="train"
    )  # pyright: ignore[reportAssignmentType]
    raw_val_dataset: Dataset = load_dataset(
        "eugenesiow/Div2k", "bicubic_x2", split="validation"
    )  # pyright: ignore[reportAssignmentType]

    train_dataset = HFDataset(raw_train_dataset, transform)
    val_dataset = HFDataset(raw_val_dataset, transform)

    return TrainableDataset(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )


def get_df2k_ost(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
    if transform is None:
        logger.info("Using default transform for DF2K OST")
        transform = transforms.ToTensor()
    else:
        logger.info("Using custom transform for DF2K OST")

    if not os.path.exists("data/df2k/train"):
        logger.info("DF2K OST Dataset not found.")

        if not os.path.exists("data/df2k.zip"):
            logger.info("df2k.zip not found. Downloading...")
            filename = hf_hub_download(
                repo_id="Iceclear/DF2K-OST",
                filename="df2k.zip",
                repo_type="dataset",
                local_dir="data",
            )
            logger.info(f"Downloaded df2k.zip to {filename}")
        else:
            filename = "data/df2k.zip"
            logger.info("Found existing df2k.zip file. Skipping download.")

        if not os.path.exists("data/df2k/share"):
            logger.info("Extracting df2k.zip...")
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall("data/df2k")
            logger.info("df2k.zip extracted.")
        else:
            logger.info("df2k.zip already extracted. Skipping extraction.")

        # preprocess
        raw_image_dir = "data/df2k/share/jywang/dataset/df2k_ost/GT"
        processed_dir = "data/df2k"
        train_dir = f"{processed_dir}/train"
        test_dir = f"{processed_dir}/test"

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        files = list(scan_files(raw_image_dir))
        total_files = len(files)
        train_size = int(0.8 * total_files)

        current = 0

        train_index = 1
        test_index = 1

        logger.info("Preprocessing DF2K OST dataset...")
        for file in tqdm(files):
            raw_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            image_list = bucket_crop_image(raw_image, 256, 256)

            if current < train_size:
                for i, image in enumerate(image_list):
                    result = cv2.imwrite(f"{train_dir}/image_{train_index}.png", image)

                    if not result:
                        logger.error(
                            f"Failed to write image {i + 1} of {file} to train directory."
                        )
                    else:
                        train_index += 1

                current += 1
            else:
                for i, image in enumerate(image_list):
                    result = cv2.imwrite(f"{test_dir}/image_{test_index}.png", image)

                    if not result:
                        logger.error(
                            f"Failed to write image {i + 1} of {file} to test directory."
                        )
                    else:
                        test_index += 1
        logger.info("DF2K OST dataset preprocessed and saved.")
        logger.info(f"Total images: Train - {train_index - 1}, Test - {test_index - 1}")

        shutil.rmtree("data/df2k/share", ignore_errors=True)

    train_dataset = UpscaleDataset("data/df2k/train", target_transform=transform)
    test_dataset = UpscaleDataset("data/df2k/test", target_transform=transform)

    return TrainableDataset(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def get_df2k_ost_small(
    batch_size: int,
    shuffle: bool,
    transform: Callable[[Any], torch.Tensor] | None = None,
):
    if transform is None:
        logger.info("Using default transform for DF2K OST Small")
        transform = transforms.ToTensor()
    else:
        logger.info("Using custom transform for DF2K OST Small")

    zip_file = "data/df2k.zip"
    raw_image_dir = "data/df2k/share/jywang/dataset/df2k_ost/GT"
    processed_dir = "data/df2k_small"
    train_dir = f"{processed_dir}/train"
    test_dir = f"{processed_dir}/test"

    if not os.path.exists(train_dir):
        logger.info("DF2K OST Dataset not found.")

        if not os.path.exists(zip_file):
            logger.info("df2k.zip not found. Downloading...")
            filename = hf_hub_download(
                repo_id="Iceclear/DF2K-OST",
                filename="df2k.zip",
                repo_type="dataset",
                local_dir="data",
            )
            logger.info(f"Downloaded df2k.zip to {filename}")
        else:
            logger.info("Found existing df2k.zip file. Skipping download.")

        if not os.path.exists("data/df2k/share"):
            logger.info("Extracting df2k.zip...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall("data/df2k")
            logger.info("df2k.zip extracted.")
        else:
            logger.info("df2k.zip already extracted. Skipping extraction.")

        # preprocess

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        files = list(scan_files(raw_image_dir))
        total_files = len(files)
        train_size = int(0.8 * total_files)

        current = 0

        train_index = 1
        test_index = 1

        logger.info("Preprocessing DF2K OST Small dataset...")
        for file in tqdm(files):
            raw_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            image_list = quad_crop_image(raw_image, 96, 96)

            if current < train_size:
                for i, image in enumerate(image_list):
                    result = cv2.imwrite(f"{train_dir}/image_{train_index}.png", image)

                    if not result:
                        logger.error(
                            f"Failed to write image {i + 1} of {file} to train directory."
                        )
                    else:
                        train_index += 1

                current += 1
            else:
                for i, image in enumerate(image_list):
                    result = cv2.imwrite(f"{test_dir}/image_{test_index}.png", image)

                    if not result:
                        logger.error(
                            f"Failed to write image {i + 1} of {file} to test directory."
                        )
                    else:
                        test_index += 1
        logger.info("DF2K OST Small dataset preprocessed and saved.")
        logger.info(f"Total images: Train - {train_index - 1}, Test - {test_index - 1}")

        shutil.rmtree("data/df2k/share", ignore_errors=True)

    train_dataset = UpscaleDataset(train_dir, target_transform=transform)
    test_dataset = UpscaleDataset(test_dir, target_transform=transform)

    return TrainableDataset(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        test=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
