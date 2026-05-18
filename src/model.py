import torch
import torch.nn as nn

class Encoder(nn.Module):\

    def __init__(self, in_channels: int = 1, latent_channels: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # block 2
            nn.Conv2d(32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # block 3
            nn.Conv2d(64, out_channels = 128, kernel_size =3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # bottleneck
            nn.Conv2d(128, latent_channels, kernel_size = 3, stride = 2, padding = 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, latent_channels: int = 16, out_channels: int = 1):
        super().__init__()

        self.decoder = nn.Sequential(
            # block 1
            nn.ConvTranspose2d(latent_channels, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # block 2
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride =2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # block 3
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride =2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # output
            nn.ConvTranspose2d(32, out_channels, kernel_size = 3, stride =2, padding = 1, output_padding = 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class AutoEncoder(nn.Module):

    def __init__(self, in_channels: int = 1, latent_channels: int = 16):
        super().__init__()

        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(latent_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encoder(x)
        return self.decoder(code)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)