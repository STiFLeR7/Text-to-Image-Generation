import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from torchvision import transforms
from PIL import Image
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
DATASET_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"  # Update this path to your dataset's image folder
ANNOTATIONS_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"  # File containing captions
CHECKPOINT_PATH = "D:/Text-to-Image-Generation/model_checkpoints"  # Path for model checkpoints

# Read captions
with open(ANNOTATIONS_FILE, 'r') as file:
    captions = file.readlines()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define custom Dataset class for loading images and captions
class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, captions, transform=None):
        self.image_dir = image_dir
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Get image filename and caption
        img_name = self.captions[idx].split()[0]
        caption = " ".join(self.captions[idx].split()[1:]).strip()

        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # Apply transformation
        if self.transform:
            img = self.transform(img)

        return img, caption

# Initialize dataset and dataloader
train_dataset = Flickr8kDataset(DATASET_DIR, captions, transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Load pre-trained Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16).to(device)

# Set up optimizer (use AdamW for fine-tuning)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-6)

# Initialize learning rate scheduler (optional, if you want to use learning rate scheduler)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(5):  # Number of epochs
    print(f"Epoch {epoch + 1}/5")
    for batch_idx, (images, captions) in enumerate(train_dataloader):
        images = images.to(device)
        captions = list(captions)

        # Tokenize captions
        text_inputs = pipe.tokenizer(
            captions,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(device)

        # Forward pass
        latents = pipe.vae.encode(images).latent_dist.sample().detach()
        latents = latents * 0.18215  # Scaling factor for latents

        noise = torch.randn_like(latents).to(device)
        timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,), device=device).long()

        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Use `last_hidden_state` from text encoder
        text_embeddings = pipe.text_encoder(input_ids).last_hidden_state

        # Pass embeddings to the UNet
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

        # Loss calculation
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update scheduler
        scheduler.step()

        # Log progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item()}")

    # Save model checkpoint after each epoch
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f"model_epoch_{epoch+1}.pth")
    torch.save(pipe.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

print("Fine-tuning complete!")
