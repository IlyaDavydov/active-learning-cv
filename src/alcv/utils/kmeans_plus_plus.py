import random
import torch

def kmeans_plus_plus(embeddings, sample_size):
    indices = []
    centroids = []
    rand_number = random.randint(0, len(embeddings)-1)
    random_embedding = embeddings[rand_number] # tensor [100]

    for i in range(sample_size):
        if i == 0:
            centroids.append(random_embedding)
            indices.append(rand_number)
            centroids = torch.tensor(centroids).unsqueeze(0) # tensor [N, 100]
            indices = torch.tensor(indices).unsqueeze(0) # tensor [N, 1]
        else:
            euclidian_distance = torch.cdist(embeddings, centroids, p=2) # tensor [embeddings_size, centroids_length]
            euclidian_distance_squared = euclidian_distance ** 2 # tensor [embeddings_size, centroids_length]
            euclidian_distance_squared_sum = torch.sum(euclidian_distance_squared, dim=1) # embeddings_size
            probabilities = euclidian_distance_squared / euclidian_distance_squared_sum.unsqueeze(1) # [embeddings_size, centroids_length]

