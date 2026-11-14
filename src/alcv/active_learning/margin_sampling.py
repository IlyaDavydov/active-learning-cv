import torch

def margin_sampling(model, unlabeled_loader, sample_size):
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

            best_probabilities = torch.topk(probabilities, 2, dim = 1).values
            scores = best_probabilities[:, 0] - best_probabilities[:, 1]

            for index, score in zip(indices, scores):
                indices_and_scores.append(
                    (index.item(), score.item())
                )

    indices_and_scores.sort(
        key=lambda item: item[1],
        reverse=False
    )

    selected_indices = [
        index
        for index, score
        in indices_and_scores[:sample_size]
    ]

    return selected_indices