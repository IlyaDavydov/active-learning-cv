import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(data: Any, filepath: str | Path) -> None:
    """
    Сохраняет данные в JSON-файл.

    Если родительских папок ещё нет, функция создаёт их автоматически.
    Если данные являются NumPy-массивом, они предварительно конвертируются
    в обычный Python list, потому что JSON не умеет сохранять np.ndarray напрямую.

    Args:
        data: Данные для сохранения.
        filepath: Путь к JSON-файлу.

    Returns:
        None.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, np.ndarray):
        data = data.tolist()

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(filepath: str | Path) -> Any:
    """
    Загружает данные из JSON-файла.

    Args:
        filepath: Путь к JSON-файлу.

    Returns:
        Данные, прочитанные из JSON-файла.
    """
    path = Path(filepath)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data