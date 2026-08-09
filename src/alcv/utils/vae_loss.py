import torch.nn.functional as F
import torch


def reconstruction_loss(recon_x, x):
    loss = F.mse_loss(recon_x, x)
    return loss


def kl_loss(mu, logvar):
    loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return loss


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    reconstruction_loss_ = reconstruction_loss(recon_x, x)
    kl_loss_ = kl_loss(mu, logvar)
    total_loss = reconstruction_loss_ + beta * kl_loss_

    return total_loss