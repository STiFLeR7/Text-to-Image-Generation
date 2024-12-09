import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import os
from PIL import Image
from torchvision import transforms

# Constants
BATCH_SIZE = 1  # Small batch size for memory efficiency
NUM_EPOCHS = 5
LR = 1e-5  # Learning rate
ACCUMULATE_GRADIENTS = 4  # Gradient accumulation steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load components of Stable Diffusion
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
vae = pipeline.vae.to(DEVICE)
text_encoder = pipeline.text_encoder.to(DEVICE)
tokenizer = pipeline.tokenizer
unet = pipeline.unet.to(DEVICE)

# Optimizer for UNet
optimizer = AdamW(unet.parameters(), lr=LR)

# Gradient scaler for mixed precision
scaler = GradScaler()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to save memory
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load dataset
def load_dataset(dataset_dir):
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "text": "sample caption"}  # Replace with actual captions

# Dataset directory
dataset_dir = "D:/COCO-Dataset/train2014/train2014"
image_paths = load_dataset(dataset_dir)
dataset = CustomDataset(image_paths, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Training loop
for epoch in range(NUM_EPOCHS):
    unet.train()
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(DEVICE).half()  # Convert to float16
        texts = batch["text"]

        # Tokenize text
        text_inputs = tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        input_ids = text_inputs.input_ids.to(DEVICE)

        # Encode text
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Encode images into latents
        latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.size(0),), device=DEVICE).long()
        noisy_latents = latents + noise

        # Predict noise using UNet
        with autocast():
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backpropagation with gradient accumulation
        scaler.scale(loss).backward()
        if (batch_idx + 1) % ACCUMULATE_GRADIENTS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item()}")

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Average Loss: {epoch_loss / len(dataloader)}")

    # Save model checkpoint after each epoch
    save_path = f"fine_tuned_model/epoch_{epoch + 1}"
    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(save_path)

    # Clear GPU memory
    torch.cuda.empty_cache()

print("Fine-tuning complete.")
