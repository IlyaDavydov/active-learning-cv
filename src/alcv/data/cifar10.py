from pathlib import Path

from torchvision.datasets import CIFAR10
from torchvision import transforms


def load_cifar10(data_dir: str | Path) -> tuple[CIFAR10, CIFAR10]:
    """
    Загружает train и test части CIFAR-10.

    Args:
        data_dir: Папка, в которую нужно скачать или из которой нужно загрузить CIFAR-10.

    Returns:
        Кортеж из двух датасетов:
            - train_dataset: тренировочная часть CIFAR-10;
            - test_dataset: тестовая часть CIFAR-10.
    """
    transform = transforms.ToTensor()

    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset