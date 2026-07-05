from pathlib import Path

import numpy as np

from alcv.active_learning.random_sampling import random_sample_indices
from alcv.analysis.class_distribution import compute_class_distribution
from alcv.utils.io import save_json
from alcv.utils.csv import save_class_distribution_csv


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    dataset_size = 50000
    num_classes = 10
    initial_labeled_size = 1000
    seed = 42

    labels = np.arange(dataset_size) % num_classes

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

    labeled_indices_path = project_root / "data/splits/milestone1/random_labeled_indices.json"
    unlabeled_indices_path = project_root / "data/splits/milestone1/random_unlabeled_indices.json"
    class_distribution_path = project_root / "reports/tables/milestone1/class_distribution.csv"

    save_json(labeled_indices, labeled_indices_path)
    save_json(unlabeled_indices, unlabeled_indices_path)
    save_class_distribution_csv(distribution_dict, class_distribution_path)

    print("Milestone 1A: Random Sampling")
    print()
    print(f"Dataset size: {dataset_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Initial labeled size: {len(labeled_indices)}")
    print(f"Unlabeled size: {len(unlabeled_indices)}")
    print(f"Seed: {seed}")
    print()
    print("Class distribution:")

    for class_id, count in distribution_dict.items():
        print(f"{class_id}: {count}")

    print()
    print("Saved files:")
    print(labeled_indices_path.relative_to(project_root))
    print(unlabeled_indices_path.relative_to(project_root))
    print(class_distribution_path.relative_to(project_root))


if __name__ == "__main__":
    main()