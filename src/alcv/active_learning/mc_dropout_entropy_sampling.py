import torch
import torch.nn as nn
from alcv.active_learning.entropy_sampling import entropy_calculate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mc_dropout_entropy_sampling(model, unlabeled_loader, sample_size, T : int):
    model = model.to(device)
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    indices_and_scores = []

    with torch.no_grad():
        for images, _, indices in unlabeled_loader:
            images = images.to(device)

            for i in range(T):
                if i == 0:
                    outputs = model(images) # [[0.1 ... 0.9] , [0.2 ... 0.8] ... [0.9 ... 0.1]]
                    probabilities_batch = torch.softmax(outputs, dim=1)
                else:
                    new_outputs = model(images)
                    new_probabilities_batch = torch.softmax(new_outputs, dim=1)
                    probabilities_batch += new_probabilities_batch

            probabilities_batch /= T

            entropy_vector = entropy_calculate(probabilities_batch)

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

    model.eval()
    return selected_indices