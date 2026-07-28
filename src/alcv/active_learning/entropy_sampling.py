import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entropy_calculate(probabilities_vector: torch.Tensor) -> list[torch.Tensor]:
    entropy_vector = []

    for probability_vector in probabilities_vector:
        entropy = 0

        for probability in probability_vector:
            if probability > 0:
                entropy -= probability * torch.log2(probability)

        entropy_vector.append(entropy)

    return entropy_vector

def entropy_sampling(model, unlabeled_loader, sample_size):
    model = model.to(device)
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

