import numpy as np


def random_sample_indices(dataset_size: int, sample_size: int, seed: int = 42):
    """
    Случайно выбирает подмножество индексов датасета как начальный labeled set.

    Функция используется как baseline-стратегия для Active Learning.
    Она выбирает sample_size уникальных индексов из полного набора данных,
    а все невыбранные индексы относит к unlabeled pool.

    Args:
        dataset_size: Общее количество объектов в датасете.
        sample_size: Количество объектов, которые нужно выбрать в labeled set.
        seed: Seed генератора случайных чисел для воспроизводимости результата.

    Returns:
        Кортеж из двух массивов:
            - labeled_indices: случайно выбранные индексы labeled set.
            - unlabeled_indices: оставшиеся индексы unlabeled pool.

    Raises:
        ValueError: Если sample_size больше dataset_size.
    """
    if sample_size > dataset_size:
        raise ValueError("sample_size cannot be larger than dataset_size")

    rng = np.random.default_rng(seed)

    all_indices = np.arange(dataset_size)
    labeled_indices = rng.choice(dataset_size, sample_size, replace=False)

    mask = np.ones(dataset_size, dtype=bool)
    mask[labeled_indices] = False

    unlabeled_indices = all_indices[mask]

    return labeled_indices, unlabeled_indices

import numpy as np


def expand_labeled_pool(
    labeled_indices,
    unlabeled_indices,
    sample_size: int = 1000,
    seed: int = 42,
):
    """
    Добавляет sample_size новых уникальных индексов в labeled pool
    и удаляет их из unlabeled pool.

    Args:
        labeled_indices: Уже размеченные индексы.
        unlabeled_indices: Еще не размеченные индексы.
        sample_size: Сколько новых индексов добавить.
        seed: Seed для воспроизводимости.

    Returns:
        updated_labeled_indices: Обновленный labeled pool.
        updated_unlabeled_indices: Обновленный unlabeled pool.
        new_labeled_indices: Только новые индексы, выбранные на этом раунде.
    """
    labeled_indices = np.array(labeled_indices)
    unlabeled_indices = np.array(unlabeled_indices)

    if sample_size > len(unlabeled_indices):
        raise ValueError("sample_size cannot be larger than unlabeled pool size")

    rng = np.random.default_rng(seed)

    selected_positions = rng.choice(
        len(unlabeled_indices),
        size=sample_size,
        replace=False
    )

    new_labeled_indices = unlabeled_indices[selected_positions]

    mask = np.ones(len(unlabeled_indices), dtype=bool)
    mask[selected_positions] = False

    updated_labeled_indices = np.concatenate([
        labeled_indices,
        new_labeled_indices
    ])

    updated_unlabeled_indices = unlabeled_indices[mask]

    return updated_labeled_indices, updated_unlabeled_indices, new_labeled_indices