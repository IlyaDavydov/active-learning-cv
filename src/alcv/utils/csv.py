from pathlib import Path
import csv


def save_class_distribution_csv(
    distribution: dict[int, int],
    path: str | Path,
) -> None:
    """
    Сохраняет распределение классов в CSV-файл.

    Функция принимает словарь формата {class_id: count} и записывает его
    в CSV-таблицу с двумя колонками: class_id и count.

    Args:
        distribution: Словарь, где ключ — id класса, а значение — количество объектов этого класса.
        path: Путь к CSV-файлу, в который нужно сохранить распределение.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["class_id", "count"]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        for class_id in sorted(distribution):
            writer.writerow(
                {
                    "class_id": class_id,
                    "count": distribution[class_id],
                }
            )