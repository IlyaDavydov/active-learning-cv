import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def typiclust_sampling(model, unlabeled_loader, labeled_loader):
    model.to(device)
    model.eval()

    unlabeled_embeddings = []
    labeled_embeddings = []

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

