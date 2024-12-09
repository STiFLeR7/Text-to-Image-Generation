import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_fid.inception import InceptionV3
from scipy.stats import entropy
from PIL import Image

# Custom Dataset class for loading images from a flat directory
class FlatImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label (0)

# Function to validate image paths and formats
def validate_images(directory):
    print(f"Validating images in directory: {directory}")
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        try:
            img = Image.open(img_path)
            img.verify()  # Ensure the image is valid
        except Exception as e:
            print(f"Invalid image detected and skipped: {img_path}, Error: {e}")
            os.remove(img_path)

# Load images using DataLoader
def get_dataloader(image_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize images to Inception model input size
        transforms.ToTensor(),         # Convert image to tensor
    ])
    
    dataset = FlatImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Use single-threaded loading
    return dataloader

# Compute the Inception Score
def calculate_inception_score(dataloader, splits=10):
    print("Calculating Inception Score...")
    
    # Load InceptionV3 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = InceptionV3([3]).to(device)  # Use the pool3 layer
    inception_model.eval()

    # Extract predictions
    preds = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            # Pass through the model's classification head (fc layer) for class probabilities
            output = inception_model(images)
            
            # Get the logits and apply softmax
            output = output[0] if isinstance(output, list) else output  # Extract tensor from list if necessary
            pred = torch.nn.functional.softmax(output, dim=1).cpu().numpy()

            if pred.size == 0:
                print(f"Warning: Empty prediction for batch, skipping...")
            else:
                preds.append(pred)

    if not preds:
        print("No predictions were generated. Check your image loading pipeline.")
        return None

    preds = np.concatenate(preds, axis=0)
    print(f"Predictions shape: {preds.shape}")

    # Split the predictions into subsets
    scores = []
    for i in range(splits):
        part = preds[i * (preds.shape[0] // splits):(i + 1) * (preds.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        scores.append(entropy(part.T, py).mean())

    # Compute the final IS
    inception_score = np.exp(np.mean(scores))
    print(f"Inception Score: {inception_score}")
    return inception_score

# Main execution function
if __name__ == "__main__":
    # Define the directory for images
    image_dir = "enhanced_images"  # Update this path if needed

    # Step 1: Validate Images
    validate_images(image_dir)

    # Step 2: Prepare DataLoader
    dataloader = get_dataloader(image_dir)

    # Step 3: Calculate Inception Score
    calculate_inception_score(dataloader)
