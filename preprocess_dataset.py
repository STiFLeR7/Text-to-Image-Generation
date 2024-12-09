import os
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch
from torchvision import transforms

# Paths
DATASET_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"  # Path to images
ANNOTATIONS_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
TRAIN_IMAGES_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr_8k.trainImages.txt"

# Custom Dataset Class
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, annotations_file, train_images_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load captions from Flickr8k.token
        with open(annotations_file, "r") as file:
            raw_captions = file.readlines()
        
        # Create a dictionary mapping image names to captions
        self.captions = {}
        for line in raw_captions:
            img_caption_pair = line.strip().split("\t")
            img_name = img_caption_pair[0].split("#")[0]  # Remove "#caption_index"
            caption = img_caption_pair[1]
            if img_name not in self.captions:
                self.captions[img_name] = []
            self.captions[img_name].append(caption)
        
        # Load training image names
        with open(train_images_file, "r") as file:
            train_images = file.readlines()
        self.image_files = [img.strip() for img in train_images if img.strip() in self.captions]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        caption = self.captions[img_name][0]  # Use the first caption
        
        if self.transform:
            image = self.transform(image)
        return image, caption

# Transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Save Dataset
dataset = Flickr8kDataset(
    img_dir=DATASET_DIR, 
    annotations_file=ANNOTATIONS_FILE, 
    train_images_file=TRAIN_IMAGES_FILE, 
    transform=transform
)

# Save preprocessed dataset
os.makedirs("processed_data", exist_ok=True)
torch.save(dataset, "processed_data/train_dataset.pt")
print(f"Preprocessed {len(dataset)} images and saved to processed_data/train_dataset.pt")
