import torch
import torch.nn as nn

from alcv.active_learning.entropy_sampling import entropy_calculate

def BALD_sampling(model, unlabeled_loader, sample_size, T: int):
    device = next(model.parameters()).device
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    indices_and_scores = []

    with torch.no_grad():
        for images, _, indices in unlabeled_loader:
            images = images.to(device)

            probabilities_sum = None
            entropies = []

            for _ in range(T):
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)

                if probabilities_sum is None:
                    probabilities_sum = probabilities.clone()
                else:
                    probabilities_sum += probabilities

                entropy = torch.stack(
                    entropy_calculate(probabilities)
                )

                entropies.append(entropy)

            # Среднее распределение вероятностей
            mean_probabilities = probabilities_sum / T

            # H(E[p(y|x, w)])
            predictive_entropy = torch.stack(entropy_calculate(mean_probabilities))

            # E[H(p(y|x, w))]
            expected_entropy = torch.stack(entropies, dim=0).mean(dim=0)

            # BALD = predictive entropy - expected entropy
            bald_scores = predictive_entropy - expected_entropy

            for index, score in zip(indices, bald_scores):
                indices_and_scores.append((int(index.item()), float(score.item())))

    indices_and_scores.sort(key=lambda x: x[1], reverse=True)

    selected_indices = [index for index, _ in indices_and_scores[:sample_size]]

    model.eval()

    return selected_indices
