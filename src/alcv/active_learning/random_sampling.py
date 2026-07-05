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