import torch
import torch.nn as nn


def compute_accuracy_and_cross_entropy_loss(model, loader, device):
    """
    Computes accuracy and average CrossEntropyLoss for a model on a given DataLoader.

    Args:
        model:
            PyTorch model to evaluate.

        loader:
            DataLoader with evaluation data.

        device:
            Device where the model and tensors should be located.

    Returns:
        tuple[float, float]:
            accuracy:
                Fraction of correctly classified samples.

            loss:
                Average CrossEntropyLoss per sample.
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty data loader")

    accuracy = total_correct / total_samples
    average_loss = total_loss / total_samples

    return accuracy, average_loss
