import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import FetalPlaneDataset
from src.model import AutoEncoder

def train(model: AutoEncoder,
          train_loader: DataLoader,
          num_epochs: int = 20,
          learning_rate: float = 1e-3,
          device: str = "cpu") -> list:

    # move model to device
    model = model.to(device)

    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #track losses
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # tqdm wraps the loader and shows a progress bar
        loop = tqdm(train_loader, desc = f"Epoch {epoch + 1} / {num_epochs}")

        for images, _ in loop:
            # move images to device
            images = images.to(device)

            # forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track loss
            epoch_loss += loss.item()
            loop.set_postfix(loss = loss.item())

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1} / {num_epochs} - avg loss: {avg_loss: .6f}")

    return losses


def visualise_reconstructions(model: AutoEncoder,
                             dataset: FetalPlaneDataset,
                             device: str = "cpu"):
    model.eval()
    classes = ["Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"]

    fig, axes = plt.subplots(2, 6, figsize = (18, 6))

    for i, plane in enumerate(classes):
        # find first image of this class
        for j in range(len(dataset)):
            image, label = dataset[j]
            if label == plane:
                image_tensor = image.unsqueeze(0).to(device)
                reconstructed = model.reconstruct(image_tensor)

                axes[0, i].imshow(image.squeeze().numpy(), cmap='gray')
                axes[0, i].set_title(plane.replace("Fetal ", "")[:10], fontsize=8)
                axes[0, i].axis('off')

                axes[1, i].imshow(reconstructed.squeeze().cpu().numpy(), cmap='gray')
                axes[1, i].set_title("recon", fontsize=8)
                axes[1, i].axis('off')
                break

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    plt.tight_layout()
    plt.show()


