import torch.nn as nn


def train_model(model, train_loader, optimizer, epochs, device):
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def train_learning_loss(model, loss_model, train_loader, optimizer, loss_optimizer, epochs, device):
    """Keep the milestone's simplified loss-prediction objective (MSE)."""
    criterion = nn.CrossEntropyLoss(reduction="none")
    prediction_criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        loss_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            features1, features2, features3, outputs = model.get_learning_loss_features(images)
            real_losses = criterion(outputs, labels)
            real_losses.mean().backward()
            optimizer.step()

            loss_optimizer.zero_grad(set_to_none=True)
            predicted_losses = loss_model(features1.detach(), features2.detach(), features3.detach()).flatten()
            prediction_loss = prediction_criterion(predicted_losses, real_losses.detach())
            prediction_loss.backward()
            loss_optimizer.step()
