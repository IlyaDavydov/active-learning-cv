import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def badge_sampling(model, unlabeled_loader, sample_size):
    model.to(device)
    model.eval()

    unlabeled_embeddings = []
    unlabeled_logits = []
    unlabeled_pseudo_labels_vectors = []
    unlabeled_indices = []
    gradient_embeddings = []

    with torch.no_grad():
        for images, _, indices in unlabeled_loader:
            images = images.to(device)
            outputs, embeddings = model.get_outputs_and_embeddings(images)
            unlabeled_logits.append(outputs) # list [batch_size, 10]
            unlabeled_embeddings.append(embeddings) # list [batch_size, 100]
            unlabeled_indices.append(indices) # list [batch_size]

        unlabeled_logits = torch.cat(unlabeled_logits, dim=0) # tensor [unlabeled_size, 10]
        unlabeled_embeddings = torch.cat(unlabeled_embeddings, dim=0) # tensor [unlabeled_size, 100]
        unlabeled_probabilities = torch.softmax(unlabeled_logits, dim=1) # tensor [unlabeled_size, 10]
        pseudo_labels = torch.argmax(unlabeled_probabilities, dim=1)  # tensor [unlabeled_size]
        unlabeled_pseudo_labels_vectors = F.one_hot(pseudo_labels, num_classes=unlabeled_probabilities.shape[1]) # tensor [unlabeled_size, 10] one_hot_vectors with pseudolabels
        unlabeled_indices = torch.cat(unlabeled_indices, dim=0) # tensor [unlabeled_size]
        unlabeled_indices = unlabeled_indices.to(device)
        # N = unlabeled_size
        # We need: [N, 10, 100] : N x class x embedding
        # We have: [N, 10] : N X class AND [N, 100] N x embedding
        # We need: [N, 10, 1] x [N, 1, 100] = [N, 10, 100]
        temp = torch.unsqueeze(unlabeled_probabilities - unlabeled_pseudo_labels_vectors, 2)
        unlabeled_embeddings = torch.unsqueeze(unlabeled_embeddings, 1)
        gradient_embeddings = temp * unlabeled_embeddings # [unlabeled_size, 10, 100]

        gradient_embeddings = torch.flatten(gradient_embeddings, start_dim=1) # unlabeled_size, 1000





