import numpy as np

def compute_class_distribution(
    labels: np.ndarray,
    indices: np.ndarray,
    num_classes: int,
) -> dict[int, int]:
    """
    Считает распределение классов среди выбранных индексов.

    Args:
        labels: Массив меток классов для всего датасета.
        indices: Индексы объектов, для которых нужно посчитать распределение классов.
        num_classes: Общее количество классов.

    Returns:
        Словарь формата {class_id: count}, где:
            - class_id: номер класса;
            - count: количество выбранных объектов этого класса.
    """
    distribution_dict = {class_id: 0 for class_id in range(num_classes)}

    for index in indices:
        class_id = int(labels[index])
        distribution_dict[class_id] += 1

    return distribution_dict