import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from transformers import AdamW
from torchvision import transforms
from PIL import Image
import os

# Define paths
DATASET_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"  # Path to your image folder
ANNOTATIONS_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"  # Path to the annotations file
TRAIN_IMAGES_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr_8k.trainImages.txt"  # Path to the train images list

CHECKPOINT_DIR = "D:/Text-to-Image-Generation/model_checkpoints"  # Path to save checkpoints

# Dataset class
class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotations_file, train_images_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = self.load_annotations(annotations_file)
        self.train_images = self.load_train_images(train_images_file)

    def load_annotations(self, file_path):
        annotations = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_id = parts[0]
                caption = parts[1]
                if img_id not in annotations:
                    annotations[img_id] = []
                annotations[img_id].append(caption)
        return annotations

    def load_train_images(self, file_path):
        with open(file_path, "r") as f:
            train_images = [line.strip() for line in f.readlines()]
        return train_images

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        img_name = self.train_images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        captions = self.annotations.get(img_name, [])

        if self.transform:
            image = self.transform(image)
        
        return image, captions


# Define the transformations to apply to images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor(),
])

# Load Dataset
dataset = Flickr8kDataset(
    img_dir=DATASET_DIR, 
    annotations_file=ANNOTATIONS_FILE, 
    train_images_file=TRAIN_IMAGES_FILE, 
    transform=transform
)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, captions) in enumerate(data_loader):
        images = images.to(device)
        
        # Tokenize captions
        captions_input_ids = []
        captions_attention_mask = []
        
        for caption_list in captions:
            # Check if there are no captions for the image (skip empty captions)
            if len(caption_list) == 0:
                continue
            
            # Tokenize each caption
            tokenized = tokenizer(caption_list, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
            captions_input_ids.append(tokenized["input_ids"])
            captions_attention_mask.append(tokenized["attention_mask"])
        
        # Skip the batch if no valid captions are available
        if len(captions_input_ids) == 0:
            continue
        
        captions_input_ids = torch.cat(captions_input_ids).to(device)
        captions_attention_mask = torch.cat(captions_attention_mask).to(device)
        
        # Forward pass
        outputs = model(
            pixel_values=images,  # Pass image to pixel_values
            labels=captions_input_ids,  # Pass tokenized caption as labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item()}")
    
    # Save model checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

print("Training complete!")
