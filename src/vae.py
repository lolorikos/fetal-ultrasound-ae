import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size = 3, stride = 2, padding = 1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=1)
        return mean, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_channels: int = 16, out_channels: int = 1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 16):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_channels)
        self.decoder = VAEDecoder(latent_channels, in_channels)

    def reparameterise(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)

        return mean + std * epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encoder(x)
        z = self.reparameterise(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var

    def reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, log_var = self.encoder(x)
            z = self.reparameterise(mean, log_var)
            return self.decoder(z)

    def generate(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n_samples, 16, 8, 8).to(device)
            return self.decoder(z)


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