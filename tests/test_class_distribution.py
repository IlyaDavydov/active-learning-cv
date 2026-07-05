import numpy as np

from alcv.analysis.class_distribution import compute_class_distribution


def test_compute_class_distribution_basic_case():
    """
    Проверяет обычный случай подсчёта распределения классов.

    В выбранных индексах присутствуют все классы, и функция должна
    корректно посчитать количество объектов каждого класса.
    """
    labels = np.array([0, 1, 1, 2, 0, 2, 2, 1])
    indices = np.array([0, 2, 3, 6])
    num_classes = 3

    distribution = compute_class_distribution(
        labels=labels,
        indices=indices,
        num_classes=num_classes,
    )

    expected_distribution = {
        0: 1,
        1: 1,
        2: 2,
    }

    assert distribution == expected_distribution


def test_compute_class_distribution_missing_class():
    """
    Проверяет случай, когда один из классов отсутствует среди выбранных индексов.

    Функция всё равно должна вернуть этот класс в словаре,
    но с количеством 0.
    """
    labels = np.array([0, 0, 1, 1])
    indices = np.array([0, 1, 2])
    num_classes = 3

    distribution = compute_class_distribution(
        labels=labels,
        indices=indices,
        num_classes=num_classes,
    )

    expected_distribution = {
        0: 2,
        1: 1,
        2: 0,
    }

    assert distribution == expected_distribution