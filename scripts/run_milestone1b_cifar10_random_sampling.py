"""
Запуск Milestone 1B: random sampling pipeline на CIFAR-10.

Этот скрипт загружает настоящий датасет CIFAR-10, выбирает начальный labeled set
из тренировочной части датасета с помощью random sampling, считает распределение
классов среди выбранных объектов и сохраняет результаты на диск.

Milestone 1B заменяет synthetic labels из Milestone 1A на реальные метки CIFAR-10,
но использует тот же базовый active learning pipeline.
"""

from pathlib import Path

from alcv.active_learning.random_sampling import random_sample_indices
from alcv.analysis.class_distribution import compute_class_distribution
from alcv.data.cifar10 import load_cifar10
from alcv.utils.csv import save_class_distribution_csv
from alcv.utils.io import save_json


def main() -> None:
    """
    Запускает random sampling pipeline для CIFAR-10.

    Pipeline выполняет следующие шаги:
        1. Находит корневую папку проекта.
        2. Загружает train и test части CIFAR-10.
        3. Берёт реальные labels из train_dataset.
        4. Случайно выбирает initial_labeled_size индексов как labeled set.
        5. Остальные индексы сохраняет как unlabeled pool.
        6. Считает распределение классов среди labeled indices.
        7. Сохраняет labeled/unlabeled indices в JSON.
        8. Сохраняет class distribution в CSV.
        9. Печатает краткий отчёт в терминал.
    """
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data/raw"
    initial_labeled_size = 1000
    seed = 42

    train_dataset, test_dataset = load_cifar10(data_dir=data_dir)

    temp_image, temp_label = train_dataset[0]
    print(f'test image: {temp_image}, shape: {temp_image.shape} \n')
    print(f'test label: {temp_label}\n')

    dataset_size = len(train_dataset)
    labels = train_dataset.targets
    class_names = train_dataset.classes
    num_classes = len(class_names)

    labeled_indices, unlabeled_indices = random_sample_indices(
        dataset_size=dataset_size,
        sample_size=initial_labeled_size,
        seed=seed,
    )

    distribution_dict = compute_class_distribution(
        labels=labels,
        indices=labeled_indices,
        num_classes=num_classes,
    )

    labeled_indices_path = project_root / "data/splits/milestone1b/random_labeled_indices.json"
    unlabeled_indices_path = project_root / "data/splits/milestone1b/random_unlabeled_indices.json"
    class_distribution_path = project_root / "reports/tables/milestone1b/class_distribution.csv"

    save_json(labeled_indices, labeled_indices_path)
    save_json(unlabeled_indices, unlabeled_indices_path)
    save_class_distribution_csv(distribution_dict, class_distribution_path)

    print("Milestone 1B: CIFAR-10 Random Sampling")
    print()
    print("Dataset: CIFAR-10")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Initial labeled size: {len(labeled_indices)}")
    print(f"Unlabeled size: {len(unlabeled_indices)}")
    print(f"Seed: {seed}")
    print()
    print("Class distribution:")

    for class_id, count in distribution_dict.items():
        class_name = class_names[class_id]
        print(f"{class_id} {class_name}: {count}")

    print()
    print("Saved files:")
    print(labeled_indices_path.relative_to(project_root))
    print(unlabeled_indices_path.relative_to(project_root))
    print(class_distribution_path.relative_to(project_root))


if __name__ == "__main__":
    main()