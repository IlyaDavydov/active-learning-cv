import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def learning_loss(classification_model, loss_model, loader, sample_size):
    device = next(classification_model.parameters()).device

    if next(loss_model.parameters()).device != device:
        raise ValueError(
            "Classification model and loss model must be on the same device"
        )

    classification_model.eval()
    loss_model.eval()

    losses = []
    indices = []

    with torch.no_grad():
        for images, _, batch_indices in loader:
            images = images.to(device)

            features1, features2, features3, _ = classification_model.get_learning_loss_features(images)
            predicted_loss = loss_model(features1, features2, features3)

            losses.append(predicted_loss)
            indices.append(batch_indices)

    losses = torch.cat(losses, dim=0).flatten()
    indices = torch.cat(indices, dim=0).flatten()

    loss_indices = torch.topk(losses, k=sample_size).indices
    indices = indices[loss_indices]

    return indices.detach().cpu().tolist()


