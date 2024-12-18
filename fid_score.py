import os
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import models


# Function to extract features (InceptionV3)
def extract_features(image_paths, model, batch_size=32, device="cpu"):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features = []
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            # Forward pass through InceptionV3
            output = model(img)
            features.append(output.cpu().numpy())

    return np.concatenate(features, axis=0)


# Function to calculate the FID score
def calculate_fid(real_features, generated_features):
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)

    # Handle the potential complex part of the covariance mean
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


# Function to get the device (CPU or GPU)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Main function to calculate the FID score
def main():
    real_images_dir = "enhanced_image"  # Directory with real images
    generated_images_dir = "generated_image"  # Directory with generated images

    device = get_device()
    print(f"Using device: {device}")

    # Load the InceptionV3 model for feature extraction
    model = models.inception_v3(pretrained=True)
    model = model.to(device)
    model.eval()

    # Get all PNG image paths in the enhanced_images directory
    real_image_paths = [
        os.path.join(real_images_dir, fname)
        for fname in os.listdir(real_images_dir)
        if fname.endswith(".png")
    ]
    generated_image_paths = [
        os.path.join(generated_images_dir, fname)
        for fname in os.listdir(generated_images_dir)
        if fname.endswith(".png")
    ]

    print(f"Validating images in directory: {real_images_dir}")
    real_features = extract_features(real_image_paths, model, device=device)
    print(f"Extracted features from real images. Shape: {real_features.shape}")

    print(f"Validating images in directory: {generated_images_dir}")
    generated_features = extract_features(generated_image_paths, model, device=device)
    print(
        f"Extracted features from generated images. Shape: {generated_features.shape}"
    )

    # Calculate FID score
    fid_score = calculate_fid(real_features, generated_features)
    print(f"FID Score: {fid_score}")


if __name__ == "__main__":
    main()
