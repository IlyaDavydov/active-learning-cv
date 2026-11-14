from operator import index as integer_index

import torch


def kmeans_plus_plus(embeddings, sample_size, *, seed=42):
    sample_size = integer_index(sample_size)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    if not 1 <= sample_size <= len(embeddings):
        raise ValueError("sample_size must be between 1 and the pool size")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("embeddings must contain only finite values")

    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(seed)
    indices = []

    first_index = torch.argmax(torch.sum(embeddings ** 2, dim=1)).item()
    actual_centroid = embeddings[first_index]
    indices.append(first_index)

    min_distances = torch.cdist(embeddings, actual_centroid.unsqueeze(0), p=2).squeeze(1) ** 2

    for i in range(1, sample_size):
        min_distances[indices] = 0

        if torch.sum(min_distances) == 0:
            available_indices = [index for index in range(len(embeddings)) if index not in indices]
            position = torch.randint(len(available_indices), (1,), device=embeddings.device, generator=generator).item()
            actual_index = available_indices[position]
        else:
            probabilities = min_distances / torch.sum(min_distances)
            actual_index = torch.multinomial(probabilities, 1, generator=generator).item()

        actual_centroid = embeddings[actual_index]
        indices.append(actual_index)

        actual_distances = torch.cdist(embeddings, actual_centroid.unsqueeze(0), p=2).squeeze(1) ** 2
        min_distances = torch.minimum(min_distances, actual_distances)

    return torch.tensor(indices, device=embeddings.device)





