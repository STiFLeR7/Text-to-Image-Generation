import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm

# Custom dataset to load images from a directory without subdirectories
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Function to get dataloader
def get_dataloader(image_dir, batch_size=16):  # Reduced batch size
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = CustomImageDataset(image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Function to calculate FID score
def calculate_fid_score(real_images, generated_images):
    # Flatten images
    real_images = real_images.view(real_images.size(0), -1)
    generated_images = generated_images.view(generated_images.size(0), -1)
    
    # Compute mean and covariance of real and generated images
    mu_real = torch.mean(real_images, dim=0)
    sigma_real = torch.cov(real_images.T)

    mu_gen = torch.mean(generated_images, dim=0)
    sigma_gen = torch.cov(generated_images.T)

    # Compute the FID score
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)

    fid = torch.norm(diff) + torch.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid.item()

# Main function
if __name__ == "__main__":
    image_dir = 'enhanced_image'  # Path to enhanced images
    dataloader = get_dataloader(image_dir)

    # Initialize list to store features
    real_features = []

    # Extract features from real images in batches
    for real_images in tqdm(dataloader):
        real_images = real_images.cuda()  # Move images to GPU
        real_features.append(real_images)
    
    real_features = torch.cat(real_features, dim=0)

    # Now for the generated images, repeat the process similarly
    generated_image_dir = 'generated_image'
    generated_dataloader = get_dataloader(generated_image_dir)

    generated_features = []

    # Extract features from generated images in batches
    for generated_images in tqdm(generated_dataloader):
        generated_images = generated_images.cuda()  # Move images to GPU
        generated_features.append(generated_images)

    generated_features = torch.cat(generated_features, dim=0)

    # Calculate FID score between real and generated images
    fid = calculate_fid_score(real_features, generated_features)
    print(f"FID Score: {fid}")
