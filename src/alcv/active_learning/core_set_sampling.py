import torch
import math
from alcv.models.simple_cnn import SimpleCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def core_set_sampling(model : SimpleCNN, labeled_loader, unlabeled_loader, sample_size):
    model.to(device)
    model.eval()

    labeled_embeddings, unlabeled_embeddings = [], []
    indices_and_scores = []

    with torch.no_grad():
        for images, _, indices in labeled_loader:
            images = images.to(device)
            outputs = model.get_embeddings(images)
            pairs = get_pairs(images, outputs)
            labeled_embeddings.append(pairs)

        for images, _, indices in unlabeled_loader:
            images = images.to(device)
            outputs = model.get_embeddings(images)
            pairs = get_pairs(images, outputs)
            unlabeled_embeddings.append(pairs)

        labeled_embeddings = [[pair for pair in vector] for vector in labeled_embeddings]
        unlabeled_embeddings = [[pair for pair in vector] for vector in unlabeled_embeddings]


        for unlabeled_embedding, unlabeled_index in labeled_embeddings:
            distances_for_embedding = []
            for labeled_embedding, unlabeled_index in unlabeled_embeddings:
                distances_for_embedding.append([euclidean_distance(unlabeled_embedding, labeled_embedding), unlabeled_index])
            best_distance_and_index = sorted(distances_for_embedding, key=lambda x: x[0], reverse=False)[0]
            indices_and_scores.append(best_distance_and_index)

def euclidean_distance(x, y):
    sum = 0
    for (c1, c2) in zip(x, y):
        sum += math.pow((c1 - c2), 2)
    return math.sqrt(sum)

def get_pairs(arr1, arr2):
    pairs = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            pair = [arr1[i], arr2[i]]
            pairs.append(pair)
    else:
        raise ValueError("arr1 and arr2 must be the same length")
    return pairs