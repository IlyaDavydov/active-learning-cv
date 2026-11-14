from operator import index as integer_index

import numpy as np

from alcv.utils.update_indices import (update_labeled_and_unlabeled_indices,)


def random_sample_indices(dataset_size: int, sample_size: int, seed: int = 42,) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible initial labeled/unlabeled split."""
    dataset_size = integer_index(dataset_size)
    sample_size = integer_index(sample_size)

    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    if sample_size > dataset_size:
        raise ValueError(
            "sample_size cannot be larger than dataset_size"
        )

    rng = np.random.default_rng(seed)

    labeled_indices = rng.choice(dataset_size, size=sample_size, replace=False,)

    mask = np.ones(dataset_size, dtype=bool)
    mask[labeled_indices] = False

    unlabeled_indices = np.arange(dataset_size)[mask]

    return labeled_indices, unlabeled_indices


def expand_labeled_pool(labeled_indices, unlabeled_indices, sample_size: int = 1000, seed: int = 42,) -> tuple[list[int], list[int], list[int]]:
    """Randomly acquire samples and update both pools."""
    sample_size = integer_index(sample_size)

    labeled_indices = [integer_index(i) for i in labeled_indices]
    unlabeled_indices = [integer_index(i) for i in unlabeled_indices]

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    if sample_size > len(unlabeled_indices):
        raise ValueError("sample_size cannot be larger than the unlabeled pool")

    rng = np.random.default_rng(seed)

    selected_positions = rng.choice(len(unlabeled_indices), size=sample_size, replace=False,)

    selected_indices = [unlabeled_indices[position] for position in selected_positions]

    updated_labeled, updated_unlabeled = (update_labeled_and_unlabeled_indices(labeled_indices, unlabeled_indices, selected_indices,))

    return (updated_labeled, updated_unlabeled, selected_indices,)