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

def load_column_from_csv(
    file_path: Path,
    column_name: str
) -> list[float]:
    """
    Загружает значения одного столбца из CSV-файла.

    Args:
        file_path:
            Путь к CSV-файлу.

        column_name:
            Название столбца, значения которого нужно загрузить.

    Returns:
        Список значений выбранного столбца.
    """
    values = []

    with file_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("CSV-файл не содержит заголовка.")

        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Столбец '{column_name}' не найден. "
                f"Доступные столбцы: {reader.fieldnames}"
            )

        for row in reader:
            values.append(float(row[column_name]))

    return values