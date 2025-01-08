from diffusers import StableDiffusionPipeline
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from PIL import Image
from tqdm import tqdm

# Dataset Preparation
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, image_size=(512, 512)):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.data = []

        # Read captions file
        with open(dataset_file, "r") as f:
            for line in f:
                # Split into image filename and caption
                parts = line.strip().split(" ", 1)  # Split at the first space
                if len(parts) != 2:
                    print(f"Skipping invalid line: {line.strip()}")  # Debugging
                continue
            image_filename, caption = parts
            print(f"Image: {image_filename}, Caption: {caption}")  # Debugging


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": torch.tensor(image).permute(2, 0, 1) / 255.0,
            "input_ids": tokens.input_ids.squeeze(),
            "attention_mask": tokens.attention_mask.squeeze()
        }

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Dataset and Dataloader
image_dir = "D:/Flickr8k-Dataset/Images"
captions_file = r"D:/Flickr8k-Dataset/captions.txt"
dataset = Flickr8kDataset(image_dir, captions_file, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load Stable Diffusion Model
model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
model.to("cuda")

# Training Loop
optimizer = torch.optim.AdamW(model.unet.parameters(), lr=5e-5)
epochs = 5

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to("cuda")
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")

        # Forward pass
        loss = model(pixel_values, input_ids, attention_mask).loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")

# Save Fine-Tuned Model
model.save_pretrained("fine_tuned_stable_diffusion")

# Testing the Fine-Tuned Model

fine_tuned_model = StableDiffusionPipeline.from_pretrained("fine_tuned_stable_diffusion")
fine_tuned_model.to("cuda")

prompt = "A beautiful day in the park."
image = fine_tuned_model(prompt).images[0]
image.show()
