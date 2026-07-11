from pathlib import Path

import matplotlib.pyplot as plt


def save_line_plot(
    x_values: list[float],
    y_values: list[float],
    save_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    Сохраняет линейный график по указанному пути.

    Args:
        x_values:
            Значения по оси X.

        y_values:
            Значения по оси Y.

        save_path:
            Путь для сохранения графика.

        title:
            Название графика.

        x_label:
            Подпись оси X.

        y_label:
            Подпись оси Y.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_values, y_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

from pathlib import Path

import matplotlib.pyplot as plt


def save_two_lines_plot(
    x_values: list[float],
    first_y_values: list[float],
    second_y_values: list[float],
    first_label: str,
    second_label: str,
    save_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    Сохраняет график с двумя линиями по указанному пути.

    Args:
        x_values:
            Значения по оси X для обеих линий.

        first_y_values:
            Значения по оси Y для первой линии.

        second_y_values:
            Значения по оси Y для второй линии.

        first_label:
            Название первой линии.

        second_label:
            Название второй линии.

        save_path:
            Путь для сохранения графика.

        title:
            Название графика.

        x_label:
            Подпись оси X.

        y_label:
            Подпись оси Y.
    """
    if len(x_values) != len(first_y_values):
        raise ValueError(
            "Количество x_values должно совпадать "
            "с количеством first_y_values."
        )

    if len(x_values) != len(second_y_values):
        raise ValueError(
            "Количество x_values должно совпадать "
            "с количеством second_y_values."
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x_values,
        first_y_values,
        marker="o",
        label=first_label
    )

    ax.plot(
        x_values,
        second_y_values,
        marker="o",
        label=second_label
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def save_two_lines_plot(
    x_values: list[float],
    first_y_values: list[float],
    second_y_values: list[float],
    first_label: str,
    second_label: str,
    save_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x_values,
        first_y_values,
        marker="o",
        label=first_label
    )

    ax.plot(
        x_values,
        second_y_values,
        marker="o",
        label=second_label
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)