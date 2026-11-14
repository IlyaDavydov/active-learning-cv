import random
import torch


def kmeans_plus_plus(embeddings, sample_size, *, seed=42):
    indices = []

    first_index = torch.argmax(torch.sum(embeddings ** 2, dim=1)).item()
    actual_centroid = embeddings[first_index]
    indices.append(first_index)

    min_distances = torch.cdist(embeddings, actual_centroid.unsqueeze(0), p=2).squeeze(1) ** 2

    for i in range(1, sample_size):
        min_distances[indices] = 0

        if torch.sum(min_distances) == 0:
            available_indices = [index for index in range(len(embeddings)) if index not in indices]
            actual_index = random.choice(available_indices)
        else:
            probabilities = min_distances / torch.sum(min_distances)
            actual_index = torch.multinomial(probabilities, 1).item()

        actual_centroid = embeddings[actual_index]
        indices.append(actual_index)

        actual_distances = torch.cdist(embeddings, actual_centroid.unsqueeze(0), p=2).squeeze(1) ** 2
        min_distances = torch.minimum(min_distances, actual_distances)

    return torch.tensor(indices, device=embeddings.device)


