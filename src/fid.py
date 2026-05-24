import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_features(model: nn.Module,
                     loader: DataLoader,
                     device: str) -> np.ndarray:
    """Extract features from encoder for FID computation."""
    model.eval()
    features = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            # use encoder to extract features
            feat = model(images)
            # flatten spatial dimensions
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fid(features_real: np.ndarray,
                features_fake: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance between two feature sets.

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
    """
    # compute means
    mu_real = np.mean(features_real, axis=0)
    mu_fake = np.mean(features_fake, axis=0)

    # compute covariances
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)

    # mean difference term
    diff = mu_real - mu_fake
    term1 = diff.dot(diff)

    # matrix square root term
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    # handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    term2 = np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(term1 + term2)


def compute_fid_for_model(encoder: nn.Module,
                          real_loader: DataLoader,
                          synthetic_images: torch.Tensor,
                          batch_size: int = 32,
                          device: str = "cpu") -> float:
    """Compute FID between real images and synthetic images."""

    # extract real features
    print("Extracting real image features...")
    real_features = extract_features(encoder, real_loader, device)

    # extract synthetic features
    print("Extracting synthetic image features...")
    synthetic_loader = DataLoader(
        torch.utils.data.TensorDataset(
            synthetic_images,
            torch.zeros(len(synthetic_images), dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=False
    )
    fake_features = extract_features(encoder, synthetic_loader, device)

    print(f"Real features shape: {real_features.shape}")
    print(f"Fake features shape: {fake_features.shape}")

    # compute FID
    fid = compute_fid(real_features, fake_features)
    return fid