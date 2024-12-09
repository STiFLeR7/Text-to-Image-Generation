import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from transformers import get_scheduler

# Paths
DATASET_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
CAPTIONS_FILE = "D:/Text-to-Image-Generation/processed_captions.json"
CHECKPOINT_DIR = "D:/Text-to-Image-Generation/model_checkpoints"

# Dataset class
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions, transform=None):
        self.image_dir = image_dir
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id = list(self.captions.keys())[idx]
        caption = self.captions[image_id][0]  # Use the first caption for simplicity

        # Load image
        img_path = os.path.join(self.image_dir, image_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")

        # Apply transformation
        if self.transform:
            img = self.transform(img)

        return img, caption

# Training function
def train_model(dataset_dir, captions_file, checkpoint_dir, num_epochs=5, batch_size=1):
    # Load captions
    with open(captions_file, "r") as f:
        captions = json.load(f)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prepare dataset and dataloader
    dataset = Flickr8kDataset(dataset_dir, captions, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Add a padding token to the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * num_epochs)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, captions) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item()}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train_model(DATASET_DIR, CAPTIONS_FILE, CHECKPOINT_DIR)
