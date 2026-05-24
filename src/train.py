import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import FetalPlaneDataset
from src.model import AutoEncoder, FetalPlaneClassifier
from src.vae import CVAE, vae_loss


def train_cvae(model: CVAE,
               train_loader: DataLoader,
               num_epochs: int = 20,
               learning_rate: float = 0.001,
               beta: float = 1.0,
               device: str = "cpu") -> list:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass - pass labels too
            reconstruction, mean, log_var = model(images, labels)

            # compute loss
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, images, mean, log_var, beta=beta
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"loss: {avg_loss:.2f} "
              f"recon: {avg_recon:.2f} "
              f"kl: {avg_kl:.2f}")

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

def save_model(model: AutoEncoder, path: str):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path: str, latent_channels: int = 16, device: str = "cpu") -> AutoEncoder:
    model = AutoEncoder(in_channels = 1, latent_channels = latent_channels)
    model.load_state_dict(torch.load(path, map_location = device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model


def train_classifier(model: FetalPlaneClassifier,
                     train_loader: DataLoader,
                     num_epochs: int = 20,
                     learning_rate: float = 1e-3,
                     device: str = "cpu",
                     class_weights: torch.Tensor = None) -> list:

    model = model.to(device)

    # loss function for classification
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    # only optimize classifier parameters, not frozen encoder
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = learning_rate
    )

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc = f"Epoch {epoch + 1} / {num_epochs}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            epoch_loss += loss.item()

            loop.set_postfix(loss = loss.item(),
                             acc = f"{100. * correct / total:.2f}")

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f} - accuracy: {accuracy:.1f}%")

    return losses


def train_vae(model: CVAE,
              train_loader: DataLoader,
              num_epochs: int = 20,
              learning_rate: float = 1e-3,
              beta: float = 1.0,
              device: str = "cpu") -> list:

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        loop = tqdm(train_loader, desc = f"Epoch {epoch + 1} / {num_epochs}")

        for images, _ in loop:
            images = images.to(device)

            # forward pass
            reconstruction, mean, log_var = model(images)

            # compute loss
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, images, mean, log_var, beta=beta
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"loss: {avg_loss:.2f} "
              f"recon: {avg_recon:.2f} "
              f"kl: {avg_kl:.2f}")

    return losses