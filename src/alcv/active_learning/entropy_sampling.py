import torch

def entropy_calculate(probabilities_vector: torch.Tensor,) -> list[torch.Tensor]:
    tiny = torch.finfo(probabilities_vector.dtype).tiny

    entropies = -(probabilities_vector * probabilities_vector.clamp_min(tiny).log2()).sum(dim=1)

    return list(entropies.unbind(0))

def entropy_sampling(model, unlabeled_loader, sample_size):
    device = next(model.parameters()).device
    model.eval()

    indices_and_scores = []

    with torch.no_grad():
        for images, _, indices in unlabeled_loader:
            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            entropy_vector = entropy_calculate(probabilities)

            for index, score in zip(indices, entropy_vector):
                indices_and_scores.append(
                    (index.item(), score.item())
                )

    indices_and_scores = sorted(indices_and_scores, key = lambda x: x[1], reverse = True)

    selected_indices = [
        index
        for index, score
        in indices_and_scores[:sample_size]
    ]

    return selected_indices

