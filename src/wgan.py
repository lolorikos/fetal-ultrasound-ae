import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # project noise to spatial feature map
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU()
        )

        # upsample to 128x128
        self.conv = nn.Sequential(
            # 8x8 → 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 16x16 → 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 32x32 → 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 64x64 → 128x128
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # project and reshape
        x = self.project(z)
        x = x.view(-1, 256, 8, 8)  # reshape to spatial
        return self.conv(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            # 128x128 → 64x64
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 64x64 → 32x32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 32x32 → 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 16x16 → 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # flatten and score
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1)
            # NO Sigmoid!
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

def gradient_penalty(critic: Critic,
                     real: torch.Tensor,
                     fake: torch.Tensor,
                     device: str) -> torch.Tensor:
    batch_size = real.size(0)

    # random interpolation between real and fake
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    # critic score on interpolated
    critic_score = critic(interpolated)

    # compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    # gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # penalty = (norm - 1)²
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

def train_wgan_gp(generator: Generator,
                  critic: Critic,
                  train_loader,
                  num_epochs: int = 50,
                  latent_dim: int = 128,
                  lr: float = 0.0001,
                  lambda_gp: float = 10,
                  critic_iterations: int = 5,
                  device: str = "cpu"):

    generator = generator.to(device)
    critic = critic.to(device)

    # Adam optimizer - works with WGAN-GP unlike WGAN
    opt_G = torch.optim.Adam(generator.parameters(),
                             lr=lr, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(critic.parameters(),
                             lr=lr, betas=(0.0, 0.9))

    G_losses = []
    C_losses = []

    for epoch in range(num_epochs):
        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ======= TRAIN CRITIC =======
            for _ in range(critic_iterations):
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_images = generator(noise).detach()

                # Wasserstein loss
                c_loss = (torch.mean(critic(fake_images))
                          - torch.mean(critic(real_images)))

                # gradient penalty
                gp = gradient_penalty(critic, real_images,
                                      fake_images, device)

                c_loss_total = c_loss + lambda_gp * gp

                opt_C.zero_grad()
                c_loss_total.backward()
                opt_C.step()

            # ======= TRAIN GENERATOR =======
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            g_loss = -torch.mean(critic(fake_images))

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        G_losses.append(g_loss.item())
        C_losses.append(c_loss_total.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} | "
                  f"C_loss: {c_loss_total.item():.4f} | "
                  f"G_loss: {g_loss.item():.4f}")

    return G_losses, C_losses