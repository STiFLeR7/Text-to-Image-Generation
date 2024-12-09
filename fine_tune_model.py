import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import json

# Hyperparameters
BATCH_SIZE = 1  # Reduced batch size to 1
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
ACCUMULATE_GRADIENTS = 4  # Accumulate gradients over multiple steps if needed

# Directories for saving models and checkpoints
CHECKPOINT_DIR = "./checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to(DEVICE)

# Dataset class to load COCO images and captions
class MyDataset(Dataset):
    def __init__(self, image_folder, captions_file):
        self.image_folder = image_folder
        self.captions = self.load_captions(captions_file)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def load_captions(self, captions_file):
        with open(captions_file, 'r') as file:
            data = json.load(file)
        annotations = data['annotations']
        captions = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            caption = annotation['caption']
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Get the image file name and corresponding caption
        image_id = list(self.captions.keys())[idx]
        image_path = os.path.join(self.image_folder, f"COCO_train2014_{image_id:012d}.jpg")
        caption = self.captions[image_id][0]  # You can change this to sample different captions

        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return {'image': image, 'text': caption}

# Path to the dataset
IMAGE_FOLDER = "D:/COCO-Dataset/train2014/train2014"  # Your image folder path
CAPTIONS_FILE = "D:/COCO-Dataset/annotations_trainval2014/annotations/captions_train2014.json"  # Your captions file path

# Initialize dataset and dataloader
dataset = MyDataset(IMAGE_FOLDER, CAPTIONS_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer for the model components (UNet, VAE, text encoder)
optimizer = AdamW(
    list(pipe.unet.parameters()) +
    list(pipe.vae.parameters()) +
    list(pipe.text_encoder.parameters()),
    lr=LEARNING_RATE
)

# Training loop
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    # Set the components to training mode
    pipe.unet.train()
    pipe.vae.train()
    pipe.text_encoder.train()

    # Loop over the dataloader with batch_idx to accumulate gradients
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
        optimizer.zero_grad()
        
        images = batch["image"].to(DEVICE)
        text = batch["text"]

        # Convert text to tokens
        inputs = pipe.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = inputs.input_ids.to(DEVICE)

        # Forward pass
        latents = pipe.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.size(0),), device=DEVICE).long()
        noisy_latents = latents + noise
        encoder_hidden_states = pipe.text_encoder(input_ids=input_ids).last_hidden_state
        predictions = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute loss
        loss = torch.nn.functional.mse_loss(predictions, noise)
        loss = loss / ACCUMULATE_GRADIENTS  # Gradient accumulation

        # Backward pass
        loss.backward()

        # Update model after gradient accumulation
        if (batch_idx + 1) % ACCUMULATE_GRADIENTS == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

    # Clear CUDA memory after each epoch
    torch.cuda.empty_cache()  # Clears the GPU memory

    # Print loss for the epoch
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(dataloader)}")

    # Save the model checkpoint after each epoch
    pipe.save_pretrained(os.path.join(CHECKPOINT_DIR, f"stable-diffusion-epoch-{epoch+1}"))

# Optionally save the final model after training is complete
pipe.save_pretrained(os.path.join(CHECKPOINT_DIR, "stable-diffusion-final"))
