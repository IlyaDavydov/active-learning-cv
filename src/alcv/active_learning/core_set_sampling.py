import torch
import math
from alcv.models.simple_cnn import SimpleCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def core_set_sampling(model : SimpleCNN, labeled_loader, unlabeled_loader, sample_size):
    model.to(device)
    model.eval()

    labeled_embeddings, unlabeled_embeddings = [], []

    for images, _, indices in labeled_loader:
        images = images.to(device)
        outputs = model.get_embeddings(images)
        labeled_embeddings.append(zip(outputs, indices))

    for images, _, indices in unlabeled_loader:
        images = images.to(device)
        outputs = model.get_embeddings(images)
        unlabeled_embeddings.append(zip(outputs, indices))

    #labeled_embeddings, unlabeled_embeddings = torch.stack(labeled_embeddings), torch.stack(unlabeled_embeddings)

    indices_and_scores = []
    temp = []

    for u_embedding, index in unlabeled_embeddings:
        temp = []
        for l_embedding, index in labeled_embeddings:
            temp.append((euclidean_distance(u_embedding, l_embedding), index))

    temp = sorted(temp, key=lambda x: x[0], reverse=True)
    indices_and_scores.append(temp[0], temp[1])


def euclidean_distance(x, y):
    sum = 0
    for (c1, c2) in zip(x, y):
        sum += math.pow((c1 - c2), 2)
    return math.sqrt(sum)