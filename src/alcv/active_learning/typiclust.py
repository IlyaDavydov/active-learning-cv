import torch
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def typiclust_sampling(model, unlabeled_loader, labeled_loader, sample_size):
    model.to(device)
    model.eval()

    unlabeled_embeddings = []
    labeled_embeddings = []

    L = len(labeled_loader.dataset)
    N = sample_size

    with torch.no_grad():
        for images, labels, indexes in labeled_loader:
            images = images.to(device)
            labels = labels.to(device)
            indexes = indexes.to(device)

            batch_embeddings = model.get_embeddings(images)
            labeled_embeddings.append(batch_embeddings)

        for images, labels, indexes in unlabeled_loader:
            images = images.to(device)
            labels = labels.to(device)
            indexes = indexes.to(device)

            batch_embeddings = model.get_embeddings(images)
            unlabeled_embeddings.append(batch_embeddings)

        labeled_embeddings = torch.cat(labeled_embeddings, dim=0)
        unlabeled_embeddings = torch.cat(unlabeled_embeddings, dim=0)

        all_embeddings = torch.cat([labeled_embeddings, unlabeled_embeddings], dim=0)

        cluster_size = L + N

        kmeans = KMeans(n_clusters=cluster_size)
        results = kmeans.fit_predict(all_embeddings)

        d = [{}, {}]  # labeled vs unlabeled

        for i in range(0, len(labeled_embeddings)):
            cluster_id = results[i]
            d[0][cluster_id] = d[0].get(cluster_id, 0) + 1

        for i in range(len(labeled_embeddings), len(results)):
            cluster_id = results[i]
            d[1][cluster_id] = d[1].get(cluster_id, 0) + 1


