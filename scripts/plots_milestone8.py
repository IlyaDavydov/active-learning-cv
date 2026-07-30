"""
This script rebuilds the milestone 8 plots from saved CSV files.

It was added because the original comparison used mismatched labeled pool sizes:
BALD started with 500 samples, while Random Sampling started with 1000.
As a result, the Random results were plotted against incorrect x-axis values.
"""


from pathlib import Path

from alcv.utils.csv import load_column_from_csv
from alcv.utils.save_plots import save_line_plot, save_two_lines_plot


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BALD_CSV_PATH = (
    PROJECT_ROOT
    / "reports/tables/milestone8/bald_sampling_experiment.csv"
)

RANDOM_CSV_PATH = (
    PROJECT_ROOT
    / "reports/tables/milestone3/random_sampling_experiment_with_dropout.csv"
)

SAVE_DIR_PLOTS = PROJECT_ROOT / "reports/figures/milestone8"


def main() -> None:
    # BALD
    bald_labeled_size = load_column_from_csv(
        BALD_CSV_PATH,
        "labeled_size"
    )

    bald_train_accuracy = load_column_from_csv(
        BALD_CSV_PATH,
        "train_accuracy"
    )

    bald_train_loss = load_column_from_csv(
        BALD_CSV_PATH,
        "train_loss"
    )

    bald_test_accuracy = load_column_from_csv(
        BALD_CSV_PATH,
        "test_accuracy"
    )

    bald_test_loss = load_column_from_csv(
        BALD_CSV_PATH,
        "test_loss"
    )

    # Random Sampling with Dropout
    random_labeled_size = load_column_from_csv(
        RANDOM_CSV_PATH,
        "labeled_size"
    )

    random_train_accuracy = load_column_from_csv(
        RANDOM_CSV_PATH,
        "train_accuracy"
    )

    random_train_loss = load_column_from_csv(
        RANDOM_CSV_PATH,
        "train_loss"
    )

    random_test_accuracy = load_column_from_csv(
        RANDOM_CSV_PATH,
        "test_accuracy"
    )

    random_test_loss = load_column_from_csv(
        RANDOM_CSV_PATH,
        "test_loss"
    )

    # save_two_lines_plot использует одну общую ось X,
    # поэтому размеры labeled pool должны совпадать.
    if bald_labeled_size != random_labeled_size:
        raise ValueError(
            "BALD and Random have different labeled_size values:\n"
            f"BALD:   {bald_labeled_size}\n"
            f"Random: {random_labeled_size}"
        )

    # Отдельные графики BALD
    save_line_plot(
        x_values=bald_labeled_size,
        y_values=bald_train_accuracy,
        save_path=SAVE_DIR_PLOTS / "train_accuracy.png",
        title="Train Accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_line_plot(
        x_values=bald_labeled_size,
        y_values=bald_train_loss,
        save_path=SAVE_DIR_PLOTS / "train_loss.png",
        title="Train Loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

    save_line_plot(
        x_values=bald_labeled_size,
        y_values=bald_test_accuracy,
        save_path=SAVE_DIR_PLOTS / "test_accuracy.png",
        title="Test Accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_line_plot(
        x_values=bald_labeled_size,
        y_values=bald_test_loss,
        save_path=SAVE_DIR_PLOTS / "test_loss.png",
        title="Test Loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

    # Сравнительные графики
    save_two_lines_plot(
        bald_labeled_size,
        bald_train_accuracy,
        random_train_accuracy,
        "BALD Sampling",
        "Random Sampling with Dropout",
        save_path=SAVE_DIR_PLOTS / "train_accuracy_comparison.png",
        title="Train Accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_two_lines_plot(
        bald_labeled_size,
        bald_train_loss,
        random_train_loss,
        "BALD Sampling",
        "Random Sampling with Dropout",
        save_path=SAVE_DIR_PLOTS / "train_loss_comparison.png",
        title="Train Loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

    save_two_lines_plot(
        bald_labeled_size,
        bald_test_accuracy,
        random_test_accuracy,
        "BALD Sampling",
        "Random Sampling with Dropout",
        save_path=SAVE_DIR_PLOTS / "test_accuracy_comparison.png",
        title="Test Accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_two_lines_plot(
        bald_labeled_size,
        bald_test_loss,
        random_test_loss,
        "BALD Sampling",
        "Random Sampling with Dropout",
        save_path=SAVE_DIR_PLOTS / "test_loss_comparison.png",
        title="Test Loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

    print(f"Plots saved to: {SAVE_DIR_PLOTS}")


if __name__ == "__main__":
    main()