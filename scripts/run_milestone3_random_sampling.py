from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from alcv.data.cifar10 import load_cifar10
from alcv.active_learning.random_sampling import random_sample_indices, expand_labeled_pool
from alcv.evaluation.evaluation import compute_accuracy_and_cross_entropy_loss
from alcv.utils.save_plots import save_line_plot
from alcv.utils.save_csv_from_columns import save_csv_from_columns
from alcv.models.simple_cnn import SimpleCNN
from alcv.models.simple_cnn_with_dropout import SimpleCNNWithDropout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/raw"

SAVE_PATH_CSV = PROJECT_ROOT / "reports/tables/milestone3/random_sampling_experiment_with_dropout.csv"
SAVE_DIR_PLOTS = PROJECT_ROOT / "reports/figures/milestone3/with_dropout"

NUM_ROUNDS = 25
INITIAL_LABELED_SIZE = 1000
SAMPLE_SIZE = 1000
SEED = 42
BATCH_SIZE = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 8

# Dropout params
DROPOUT = 0.3

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
        "Dropout"
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
    dropout_column = []


    criterion = nn.CrossEntropyLoss()

    labeled_indices = []
    unlabeled_indices = []

    for round_idx in range(NUM_ROUNDS):
        if round_idx == 0:
            labeled_indices, unlabeled_indices = random_sample_indices(
                len(train_dataset),
                INITIAL_LABELED_SIZE,
                SEED
            )
            selected_batch_size = 0
        else:
            labeled_indices, unlabeled_indices, selected_indices = expand_labeled_pool(
                labeled_indices,
                unlabeled_indices,
                SAMPLE_SIZE,
                SEED + round_idx
            )
            selected_batch_size = len(selected_indices)

        labeled_subset = Subset(train_dataset, labeled_indices)

        train_loader = DataLoader(
            labeled_subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        torch.manual_seed(SEED + round_idx)

        model = SimpleCNNWithDropout(DROPOUT).to(device)

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

        strategy_column.append("random_sampling_with_dropout")
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
        dropout_column.append(DROPOUT)

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
        dropout_column
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


if __name__ == "__main__":
    main()



