from operator import index as integer_index

import torch

from alcv.models.simple_cnn import SimpleCNN


def core_set_sampling(model: SimpleCNN, labeled_loader, unlabeled_loader, sample_size: int, distance_chunk_size: int = 2048,) -> list[int]:
    sample_size = integer_index(sample_size)
    distance_chunk_size = integer_index(distance_chunk_size)

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    if distance_chunk_size <= 0:
        raise ValueError("distance_chunk_size must be greater than zero")

    device = next(model.parameters()).device
    model.eval()

    labeled_batches = []
    unlabeled_batches = []
    unlabeled_indices = []

    with torch.no_grad():
        for images, _, _ in labeled_loader:
            images = images.to(device)
            embeddings = model.get_embeddings(images)
            labeled_batches.append(embeddings)

        for images, _, indices in unlabeled_loader:
            images = images.to(device)
            embeddings = model.get_embeddings(images)

            unlabeled_batches.append(embeddings)
            unlabeled_indices.extend(indices.tolist())

    if not labeled_batches:
        raise ValueError("Labeled loader must provide at least one sample")

    if not unlabeled_batches:
        raise ValueError("Unlabeled loader must provide at least one sample")

    labeled_embeddings = torch.cat(labeled_batches, dim=0)
    unlabeled_embeddings = torch.cat(unlabeled_batches, dim=0)

    if labeled_embeddings.shape[0] == 0:
        raise ValueError("Labeled pool must not be empty")

    if unlabeled_embeddings.shape[0] == 0:
        raise ValueError("Unlabeled pool must not be empty")

    if sample_size > len(unlabeled_indices):
        raise ValueError("sample_size cannot be greater than the unlabeled pool size")

    if not torch.isfinite(labeled_embeddings).all().item():
        raise ValueError("Labeled embeddings contain NaN or infinite values")

    if not torch.isfinite(unlabeled_embeddings).all().item():
        raise ValueError("Unlabeled embeddings contain NaN or infinite values")

    # Distance from each unlabeled point to its nearest labeled point.
    min_distances_parts = []

    for start in range(0, len(unlabeled_embeddings), distance_chunk_size,):
        end = start + distance_chunk_size
        unlabeled_chunk = unlabeled_embeddings[start:end]

        distances = torch.cdist(unlabeled_chunk, labeled_embeddings, p=2,)

        min_distances_parts.append(distances.min(dim=1).values)

    min_distances = torch.cat(min_distances_parts, dim=0)
    selected_indices = []

    # k-Center Greedy selection.
    for _ in range(sample_size):
        selected_position = torch.argmax(min_distances).item()

        selected_indices.append(int(unlabeled_indices[selected_position]))

        new_center = unlabeled_embeddings[selected_position].unsqueeze(0)

        distances_to_new_center = torch.cdist(unlabeled_embeddings, new_center, p=2,).squeeze(1)

        min_distances = torch.minimum(min_distances, distances_to_new_center,)

        # Previously selected positions remain excluded.
        min_distances[selected_position] = -torch.inf

    return selected_indices