import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 16, num_classes: int = 6):
        super().__init__()
        # input channels = image channels + num_classes (one-hot)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, latent_channels * 2, kernel_size = 3, stride = 2, padding = 1),
            # *2 because we output mean and log_var
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> tuple:
        # convert label to one-hot
        one_hot = torch.zeros(x.shape[0], self.num_classes,
                              x.shape[2], x.shape[3]).to(x.device)
        one_hot.scatter_(1, label.view(-1, 1, 1, 1).expand(
            -1, -1, x.shape[2], x.shape[3]), 1)

        # concatenate image and label
        x_conditioned = torch.cat([x, one_hot], dim=1)

        encoded = self.encoder(x_conditioned)
        mean, log_var = encoded.chunk(2, dim=1)
        return mean, log_var


class CVAEDecoder(nn.Module):
    def __init__(self, latent_channels: int = 16,
                 out_channels: int = 1,
                 num_classes: int = 6):
        super().__init__()
        # latent + num_classes channels as input
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels + num_classes, 128,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.num_classes = num_classes

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # add label to latent space
        one_hot = torch.zeros(z.shape[0], self.num_classes,
                              z.shape[2], z.shape[3]).to(z.device)
        one_hot.scatter_(1, label.view(-1, 1, 1, 1).expand(
            -1, -1, z.shape[2], z.shape[3]), 1)

        z_conditioned = torch.cat([z, one_hot], dim=1)
        return self.decoder(z_conditioned)


class CVAE(nn.Module):
    def __init__(self, in_channels: int = 1,
                 latent_channels: int = 16,
                 num_classes: int = 6):
        super().__init__()
        self.encoder = CVAEEncoder(in_channels, latent_channels, num_classes)
        self.decoder = CVAEDecoder(latent_channels, in_channels, num_classes)

    def reparameterise(self, mean: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x: torch.Tensor,
                label: torch.Tensor) -> tuple:
        mean, log_var = self.encoder(x, label)
        z = self.reparameterise(mean, log_var)
        reconstruction = self.decoder(z, label)
        return reconstruction, mean, log_var

    def reconstruct(self, x: torch.Tensor,
                    label: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, log_var = self.encoder(x, label)
            z = self.reparameterise(mean, log_var)
            return self.decoder(z, label)

    def generate(self, label: int, n_samples: int = 1,
                 device: str = 'cpu') -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n_samples, 16, 8, 8).to(device)
            labels = torch.full((n_samples,), label,
                                dtype=torch.long).to(device)
            return self.decoder(z, labels)


def vae_loss(reconstruction, original, mean, log_var, beta = 1.0):
    """
    VAE loss = reconstruction loss + beta * KL divergence

    beta=1.0  → standard VAE
    beta>1.0  → beta-VAE, stronger disentanglement
    """
    # reconstruction loss - how well did we reconstruct?
    recon_loss = F.binary_cross_entropy(reconstruction, original, reduction='sum')

    # KL divergence - how close is the latent to N(0, 1)?
    kl_loss = -0.5 * torch.sum(
        1 + log_var - mean ** 2 - torch.exp(log_var)
    )

    return recon_loss + beta * kl_loss, recon_loss, kl_loss