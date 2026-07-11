from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from alcv.data.cifar10 import load_cifar10
from alcv.active_learning.random_sampling import random_sample_indices
from alcv.evaluation.evaluation import compute_accuracy_and_cross_entropy_loss
from alcv.utils.save_plots import save_line_plot, save_two_lines_plot
from alcv.utils.save_csv_from_columns import save_csv_from_columns
from alcv.models.simple_cnn import SimpleCNN
from alcv.utils.update_indices import update_labeled_and_unlabeled_indices
from alcv.active_learning.least_confidence import least_confidence, DatasetWithIndices
from alcv.utils.csv import load_column_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/raw"

SAVE_PATH_CSV = PROJECT_ROOT / "reports/tables/milestone4/least_confidence_experiment.csv"
SAVE_DIR_PLOTS = PROJECT_ROOT / "reports/figures/milestone4"
PATH_TO_CSV_FROM_MILESTONE3 = PROJECT_ROOT / "reports/tables/milestone3/random_sampling_experiment.csv"

NUM_ROUNDS = 25
SAMPLE_SIZE = 1000
SEED = 42
BATCH_SIZE = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    train_dataset, test_dataset = load_cifar10(DATA_DIR)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    header = [
        "strategy",
        "seed",
        "round",
        "labeled_size",
        "unlabeled_size",
        "selected_batch_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "momentum",
        "train_accuracy",
        "train_loss",
        "test_accuracy",
        "test_loss",
    ]

    strategy_column = []
    seed_column = []
    round_column = []
    labeled_size_column = []
    unlabeled_size_column = []
    selected_batch_size_column = []
    epochs_column = []
    batch_size_column = []
    learning_rate_column = []
    momentum_column = []
    train_accuracy_column = []
    train_loss_column = []
    test_accuracy_column = []
    test_loss_column = []

    criterion = nn.CrossEntropyLoss()

    labeled_indices = []
    unlabeled_indices = []

    train_dataset_with_indices = DatasetWithIndices(train_dataset)

    for round_idx in range(NUM_ROUNDS):
        if round_idx == 0:
            labeled_indices, unlabeled_indices = random_sample_indices(
                len(train_dataset),
                SAMPLE_SIZE,
                SEED
            )
            selected_batch_size = 0
        else:
            unlabeled_subset = Subset(train_dataset_with_indices, unlabeled_indices)

            unlabeled_loader = DataLoader(
                unlabeled_subset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            selected_indices = least_confidence(model, unlabeled_loader, sample_size=SAMPLE_SIZE)
            labeled_indices, unlabeled_indices = update_labeled_and_unlabeled_indices(labeled_indices, unlabeled_indices, selected_indices)
            selected_batch_size = len(selected_indices)

        torch.manual_seed(SEED + round_idx)

        model = SimpleCNN().to(device)

        labeled_subset = Subset(train_dataset, labeled_indices)
        train_loader = DataLoader(labeled_subset, BATCH_SIZE, shuffle=True)

        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM
        )

        for epoch in range(EPOCHS):
            model.train()

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

        train_accuracy, train_loss = compute_accuracy_and_cross_entropy_loss(
            model,
            train_loader,
            device
        )

        test_accuracy, test_loss = compute_accuracy_and_cross_entropy_loss(
            model,
            test_loader,
            device
        )

        strategy_column.append("least_confidence")
        seed_column.append(SEED)
        round_column.append(round_idx)
        labeled_size_column.append(len(labeled_indices))
        unlabeled_size_column.append(len(unlabeled_indices))
        selected_batch_size_column.append(selected_batch_size)
        epochs_column.append(EPOCHS)
        batch_size_column.append(BATCH_SIZE)
        learning_rate_column.append(LEARNING_RATE)
        momentum_column.append(MOMENTUM)
        train_accuracy_column.append(train_accuracy)
        train_loss_column.append(train_loss)
        test_accuracy_column.append(test_accuracy)
        test_loss_column.append(test_loss)

        print(
            f"after round {round_idx}:\n"
            f"labeled size: {len(labeled_indices)}\n"
            f"train accuracy: {train_accuracy:.3f}\n"
            f"train loss: {train_loss:.3f}\n"
            f"test accuracy: {test_accuracy:.3f}\n"
            f"test loss: {test_loss:.3f}\n"
        )

    columns = [
        strategy_column,
        seed_column,
        round_column,
        labeled_size_column,
        unlabeled_size_column,
        selected_batch_size_column,
        epochs_column,
        batch_size_column,
        learning_rate_column,
        momentum_column,
        train_accuracy_column,
        train_loss_column,
        test_accuracy_column,
        test_loss_column,
    ]

    save_csv_from_columns(
        save_path=SAVE_PATH_CSV,
        header=header,
        columns=columns
    )

    save_line_plot(
        x_values=labeled_size_column,
        y_values=train_accuracy_column,
        save_path=SAVE_DIR_PLOTS / "train_accuracy.png",
        title="Train accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_line_plot(
        x_values=labeled_size_column,
        y_values=train_loss_column,
        save_path=SAVE_DIR_PLOTS / "train_loss.png",
        title="Train loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

    save_line_plot(
        x_values=labeled_size_column,
        y_values=test_accuracy_column,
        save_path=SAVE_DIR_PLOTS / "test_accuracy.png",
        title="Test accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_line_plot(
        x_values=labeled_size_column,
        y_values=test_loss_column,
        save_path=SAVE_DIR_PLOTS / "test_loss.png",
        title="Test loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

    train_loss_of_milestone3 = load_column_from_csv(
        PATH_TO_CSV_FROM_MILESTONE3,
        "train_loss"
    )

    train_accuracy_of_milestone3 = load_column_from_csv(
        PATH_TO_CSV_FROM_MILESTONE3,
        "train_accuracy"
    )

    test_loss_of_milestone3 = load_column_from_csv(
        PATH_TO_CSV_FROM_MILESTONE3,
        "test_loss"
    )

    test_accuracy_of_milestone3 = load_column_from_csv(
        PATH_TO_CSV_FROM_MILESTONE3,
        "test_accuracy"
    )

    save_two_lines_plot(
        labeled_size_column,
        train_accuracy_column,
        train_accuracy_of_milestone3,
        "Least Confidence",
        "Random Sampling",
        save_path=SAVE_DIR_PLOTS / "train_accuracy_comparison.png",
        title="Train Accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_two_lines_plot(
        labeled_size_column,
        train_loss_column,
        train_loss_of_milestone3,
        "Least Confidence",
        "Random Sampling",
        save_path=SAVE_DIR_PLOTS / "train_loss_comparison.png",
        title="Train Loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

    save_two_lines_plot(
        labeled_size_column,
        test_accuracy_column,
        test_accuracy_of_milestone3,
        "Least Confidence",
        "Random Sampling",
        save_path=SAVE_DIR_PLOTS / "test_accuracy_comparison.png",
        title="Test Accuracy",
        x_label="Labeled samples",
        y_label="Accuracy"
    )

    save_two_lines_plot(
        labeled_size_column,
        test_loss_column,
        test_loss_of_milestone3,
        "Least Confidence",
        "Random Sampling",
        save_path=SAVE_DIR_PLOTS / "test_loss_comparison.png",
        title="Test Loss",
        x_label="Labeled samples",
        y_label="Loss"
    )

if __name__ == "__main__":
    main()