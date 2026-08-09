from torch import nn
import torch

def discriminator_loss(discriminator, labeled_mu, unlabeled_mu):
    labeled_predictions = discriminator(labeled_mu)
    unlabeled_predictions = discriminator(unlabeled_mu)

    labeled = torch.ones_like(labeled_predictions)
    unlabeled = torch.zeros_like(unlabeled_predictions)

    loss = nn.BCELoss()

    return loss(labeled_predictions, labeled) + loss(unlabeled_predictions, unlabeled)

