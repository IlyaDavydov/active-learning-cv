'''

1. Load CIFAR-10
2. Load random_labeled_indices.json from Milestone 1B
3. Create train Subset with only those 1000 indices
4. Create DataLoaders
5. Create model
6. Train model for several epochs
7. Evaluate on test set
8. Save metrics
9. Save trained model

'''

from pathlib import Path
from alcv.data.cifar10 import load_cifar10
from alcv.utils.io import load_json
from torch.utils.data import Subset, DataLoader
from alcv.models.simple_cnn import SimpleCNN
import torch.optim as optim
import torch.nn as nn
import torch
from alcv.evaluation.evaluation import compute_accuracy_and_cross_entropy_loss
from alcv.utils.save_plots import save_line_plot
from alcv.utils.save_csv_from_columns import save_csv_from_columns

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data/raw"
    trainset, testset = load_cifar10(data_dir)
    filepath_json = project_root / "data/splits/milestone1b"
    labeled_indices = load_json(filepath_json / "random_labeled_indices.json")

    labeled_subset = Subset(trainset, labeled_indices)
    loader = DataLoader(labeled_subset, batch_size=10, shuffle=True)
    test_loader = DataLoader(testset, batch_size=10)
    model = SimpleCNN()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = [[], [], [], [], []] # num, train_acc, train_loss, test_acc, test_loss

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_accuracy, train_loss = compute_accuracy_and_cross_entropy_loss(model, loader, device)
        test_accuracy, test_loss = compute_accuracy_and_cross_entropy_loss(model, test_loader, device)

        epochs[0].append(epoch)
        epochs[1].append(train_accuracy)
        epochs[2].append(train_loss)
        epochs[3].append(test_accuracy)
        epochs[4].append(test_loss)

        print(f'after epoch {epoch}: \n'
              f'train accuracy: {round(train_accuracy, 3)} \n'
              f'train_loss: {round(train_loss, 3)}  \n' 
              f'test_accuracy: {round(test_accuracy, 3)} \n'
              f'test_loss: {round(test_loss, 3)} \n')

    save_path_csv = project_root / "reports/tables/milestone2/metrics.csv"
    header = ["epoch", "train_accuracy", "train_loss", "test_accuracy", "test_loss"]
    save_csv_from_columns(save_path=save_path_csv, header=header, columns=epochs)

    save_line_plot(
        x_values=epochs[0],
        y_values=epochs[1],
        save_path=project_root / "reports/figures/milestone2/train_accuracy.png",
        title="Train accuracy",
        x_label="Epoch",
        y_label="Accuracy",
    )

    save_line_plot(
        x_values=epochs[0],
        y_values=epochs[2],
        save_path=project_root / "reports/figures/milestone2/train_loss.png",
        title="Train loss",
        x_label="Epoch",
        y_label="Loss",
    )

    save_line_plot(
        x_values=epochs[0],
        y_values=epochs[3],
        save_path=project_root / "reports/figures/milestone2/test_accuracy.png",
        title="Test accuracy",
        x_label="Epoch",
        y_label="Accuracy",
    )

    save_line_plot(
        x_values=epochs[0],
        y_values=epochs[4],
        save_path=project_root / "reports/figures/milestone2/test_loss.png",
        title="Test loss",
        x_label="Epoch",
        y_label="Loss",
    )

    save_path_model_weights = project_root / "models/milestone2" / "simple_cnn.pth"
    save_path_model_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path_model_weights)

if __name__ == "__main__":
    main()