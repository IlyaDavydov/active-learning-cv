import numpy as np
import pytest

from alcv.active_learning.random_sampling import random_sample_indices


def test_random_sample_indices():
    """
    Проверяет базовые инварианты функции random_sample_indices.

    Тест убеждается, что функция:
        - возвращает labeled_indices нужного размера;
        - возвращает unlabeled_indices нужного размера;
        - не допускает пересечения между labeled и unlabeled индексами;
        - покрывает все индексы датасета от 0 до dataset_size - 1;
        - не создаёт дубликаты внутри labeled и unlabeled;
        - даёт одинаковый результат при одинаковом seed;
        - выбрасывает ValueError, если sample_size больше dataset_size.
    """
    dataset_size = 1000
    sample_size = 100
    seed = 42

    labeled_indices, unlabeled_indices = random_sample_indices(
        dataset_size,
        sample_size,
        seed,
    )

    assert len(labeled_indices) == sample_size
    assert len(unlabeled_indices) == dataset_size - sample_size

    assert len(set(labeled_indices) & set(unlabeled_indices)) == 0

    assert np.array_equal(
        np.union1d(labeled_indices, unlabeled_indices),
        np.arange(dataset_size),
    )

    assert len(labeled_indices) == len(set(labeled_indices))
    assert len(unlabeled_indices) == len(set(unlabeled_indices))

    labeled_1, unlabeled_1 = random_sample_indices(1000, 100, 33)
    labeled_2, unlabeled_2 = random_sample_indices(1000, 100, 33)

    assert np.array_equal(labeled_1, labeled_2)
    assert np.array_equal(unlabeled_1, unlabeled_2)

    with pytest.raises(ValueError):
        random_sample_indices(1000, 1100, 42)