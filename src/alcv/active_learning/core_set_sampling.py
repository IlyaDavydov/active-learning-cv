import torch

from alcv.models.simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def core_set_sampling(model: SimpleCNN, labeled_loader, unlabeled_loader, sample_size: int, distance_chunk_size: int = 2048) -> list[int]:
    model = model.to(device)
    model.eval()

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    labeled_embeddings = []
    unlabeled_embeddings = []
    unlabeled_indices = []

    with torch.no_grad():
        # Labeled embeddings
        for images, _, _ in labeled_loader:
            images = images.to(device)
            embeddings = model.get_embeddings(images)
            labeled_embeddings.append(embeddings)

        # Unlabeled embeddings and their original dataset indices
        for images, _, indices in unlabeled_loader:
            images = images.to(device)
            embeddings = model.get_embeddings(images)

            unlabeled_embeddings.append(embeddings)
            unlabeled_indices.extend(indices.tolist())

    labeled_embeddings = torch.cat(labeled_embeddings, dim=0)
    unlabeled_embeddings = torch.cat(unlabeled_embeddings, dim=0)

    if sample_size > len(unlabeled_indices):
        raise ValueError(
            "sample_size cannot be greater than the unlabeled pool size"
        )

    # For every unlabeled embedding:
    # distance to the nearest labeled embedding.
    min_distances_parts = []

    for start in range(0, len(unlabeled_embeddings), distance_chunk_size):
        end = start + distance_chunk_size
        unlabeled_chunk = unlabeled_embeddings[start:end]

        distances = torch.cdist(unlabeled_chunk,labeled_embeddings,p=2) # [unlabeled_chunk's length x labeled_embeddings' length]

        minimum_distances = distances.min(dim=1).values
        min_distances_parts.append(minimum_distances) #nums of chunks x chunk_size

    min_distances = torch.cat(min_distances_parts) # num_of_chunks * chunk_size

    selected_indices = []

    # k-Center Greedy selection
    for _ in range(sample_size):
        selected_position = torch.argmax(min_distances).item() #max index position tensor -> position index number

        selected_indices.append(unlabeled_indices[selected_position])

        new_center = unlabeled_embeddings[selected_position].unsqueeze(0)

        # Distance from every unlabeled point to the new center
        distances_to_new_center = torch.cdist(
            unlabeled_embeddings,
            new_center,
            p=2,
        ).squeeze(1)

        # The nearest center may now be the newly selected point
        min_distances = torch.minimum(
            min_distances,
            distances_to_new_center,
        )

        # Prevent selecting the same point again
        min_distances[selected_position] = -torch.inf

    return selected_indices