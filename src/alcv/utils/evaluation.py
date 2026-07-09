import torch.nn as nn
import torch


def compute_accuracy_and_cross_entropy_loss(model, loader):
    """
    Считает accuracy и CrossEntropyLoss для модели на переданном DataLoader.

    Функция используется для оценки качества модели после обучения или между
    эпохами обучения.

    Args:
        model:
            PyTorch-модель, которую нужно оценить.

        loader:
            DataLoader с данными для evaluation.

    Returns:
        tuple[float, float]:
            accuracy:
                Доля правильно классифицированных изображений.

            loss:
                Значение CrossEntropyLoss, округлённое до двух знаков.
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)

            predictions = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    loss = total_loss / len(loader)

    return round(accuracy, 3), round(loss, 2)