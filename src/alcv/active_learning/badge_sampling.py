import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def badge_sampling(model, unlabeled_loader, sample_size):
    model.to(device)
    model.eval()

    unlabeled_embeddings = []
    unlabeled_logits = []
    unlabeled_pseudo_labels = []
    unlabeled_indices = []

    with torch.no_grad():
        for images, _, indices in unlabeled_loader:
            images = images.to(device)
            unlabeled_logits.append(model(images)) # list [batch_size, 10]
            unlabeled_embeddings.append(model.get_embeddings(images)) # list [batch_size, 1000]
            unlabeled_indices.append(indices) # list [batch_size]

        unlabeled_logits = torch.cat(unlabeled_logits, dim=0) # tensor [labeled_size, 10]
        unlabeled_embeddings = torch.cat(unlabeled_embeddings, dim=0) # tensor [labeled_size, 100]

        unlabeled_probabilities = torch.softmax(unlabeled_logits, dim=1) # tensor [labeled_size, 10]
        



