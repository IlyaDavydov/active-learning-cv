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