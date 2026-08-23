import torch

from alcv.utils.vae_loss import vae_loss
from alcv.utils.discriminator_loss import discriminator_loss
from alcv.utils.adversarial_loss import adversarial_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vaal_train(vae, discriminator, labeled_loader, unlabeled_loader, epochs, lr=0.0005, beta=1.0, adversary_param=1.0):
    vae = vae.to(device)
    discriminator = discriminator.to(device)

    vae.train()
    discriminator.train()

    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        labeled_iterator = iter(labeled_loader)

        for unlabeled_images, _, _ in unlabeled_loader:
            try:
                labeled_images, _, _ = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(labeled_loader)
                labeled_images, _, _ = next(labeled_iterator)

            labeled_images = labeled_images.to(device)
            unlabeled_images = unlabeled_images.to(device)

            # VAE step

            optimizer_vae.zero_grad()

            for parameter in discriminator.parameters():
                parameter.requires_grad = False

            labeled_recons, _, labeled_mu, labeled_logvar = vae(labeled_images)
            unlabeled_recons, _, unlabeled_mu, unlabeled_logvar = vae(unlabeled_images)

            labeled_vae_loss = vae_loss(labeled_recons, labeled_images, labeled_mu, labeled_logvar, beta)
            unlabeled_vae_loss = vae_loss(unlabeled_recons, unlabeled_images, unlabeled_mu, unlabeled_logvar, beta)
            adversarial_loss_ = adversarial_loss(discriminator, labeled_mu, unlabeled_mu)

            total_vae_loss = labeled_vae_loss + unlabeled_vae_loss + adversary_param * adversarial_loss_

            total_vae_loss.backward()
            optimizer_vae.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad = True

            # Discriminator step

            optimizer_discriminator.zero_grad()

            with torch.no_grad():
                _, _, labeled_mu, _ = vae(labeled_images)
                _, _, unlabeled_mu, _ = vae(unlabeled_images)

            discriminator_loss_ = discriminator_loss(discriminator, labeled_mu, unlabeled_mu)

            discriminator_loss_.backward()
            optimizer_discriminator.step()

    return vae, discriminator

def vaal_sampling(vae, discriminator, unlabeled_loader, sample_size):
    vae = vae.to(device)
    discriminator = discriminator.to(device)

    vae.eval()
    discriminator.eval()

    indices_and_scores = []

    with torch.no_grad():
        for images, _, indices in unlabeled_loader:
            images = images.to(device)

            _, _, mu, _ = vae(images)
            scores = discriminator(mu).squeeze(1)

            for index, score in zip(indices, scores):
                indices_and_scores.append((index.item(), score.item()))

    indices_and_scores.sort(key=lambda pair: pair[1])

    selected_indices = [index for index, _ in indices_and_scores[:sample_size]]

    return selected_indices