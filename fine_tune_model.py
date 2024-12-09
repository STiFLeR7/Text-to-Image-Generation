import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, StableDiffusionTrainer
import os
from PIL import Image
from tqdm import tqdm

# Configuration
DATASET_DIR = "E:/preprocessed_data"  # Path to preprocessed images
ANNOTATIONS_FILE = "D:/COCO-Dataset/annotations_trainval2014/annotations/captions_train2014.json"
MODEL_DIR = "CompVis/stable-diffusion-v1-4"  # Pretrained model directory
OUTPUT_DIR = "./fine_tuned_model"  # Directory to save the fine-tuned model
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset Definition
class CustomCocoDataset(Dataset):
    def __init__(self, image_dir, annotations_file, tokenizer, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.annotations = self.load_annotations(annotations_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def load_annotations(self, file_path):
        import json
        with open(file_path, "r") as f:
            data = json.load(f)
        return data["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        image_path = os.path.join(self.image_dir, f"processed_COCO_train2014_{image_id:012d}.jpg")

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise ValueError(f"Image not found: {image_path}")

        if self.transform:
            image = self.transform(image)

        tokens = self.tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return {"image": image, "input_ids": tokens["input_ids"].squeeze(), "attention_mask": tokens["attention_mask"].squeeze()}


# Load Dataset
tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR)
dataset = CustomCocoDataset(DATASET_DIR, ANNOTATIONS_FILE, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load Pretrained Model
pipeline = StableDiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
vae = pipeline.vae
unet = pipeline.unet
text_encoder = pipeline.text_encoder
vae.to(DEVICE)
unet.to(DEVICE)
text_encoder.to(DEVICE)

# Fine-Tuning Setup
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

# Fine-Tune Model
print("Starting fine-tuning...")
unet.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        images = batch["image"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # Encode text
        encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Get VAE latent space
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor

        # Forward pass through UNet
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.size(0),), device=DEVICE).long()
        noisy_latents = latents + noise
        predictions = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Loss computation
        loss = torch.nn.functional.mse_loss(predictions, noise)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(dataloader)}")

# Save Fine-Tuned Model
pipeline.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuned model saved to {OUTPUT_DIR}")
