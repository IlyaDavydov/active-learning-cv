import csv
from pathlib import Path


def save_csv_from_columns(save_path, header, columns):
    """
    Saves column-based data to a CSV file.

    Args:
        save_path:
            Path where the CSV file should be saved.

        header:
            List of column names.

        columns:
            List of columns. Each column is a list of values.

    Example:
        header = ["epoch", "accuracy", "loss"]
        columns = [
            [0, 1, 2],
            [0.31, 0.38, 0.42],
            [1.9, 1.7, 1.5],
        ]
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    number_of_columns = len(header)

    if len(columns) < number_of_columns:
        raise ValueError(
            f"Not enough columns: header has {number_of_columns} columns, "
            f"but only {len(columns)} columns were provided."
        )

    selected_columns = columns[:number_of_columns]

    column_lengths = [len(column) for column in selected_columns]

    if len(set(column_lengths)) != 1:
        raise ValueError(
            f"All columns must have the same length. Got lengths: {column_lengths}"
        )

    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(zip(*selected_columns))