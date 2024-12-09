
import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline

from datasets import load_dataset

# Constants
COCO_IMAGE_DIR = "D:/COCO-Dataset/train2014/train2014"
COCO_CAPTIONS_FILE = "D:/COCO-Dataset/annotations_trainval2014/annotations/captions_train2014.json"
OUTPUT_DIR = "fine_tuned_model"
BATCH_SIZE = 1
EPOCHS = 1
LR = 5e-6
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"

# Dataset
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        from pycocotools.coco import COCO
        self.coco = COCO(captions_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        caption_data = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        captions = [ann['caption'] for ann in caption_data]

        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        return image, captions[0] if captions else ""

# Initialize Dataset and DataLoader
dataset = COCODataset(COCO_IMAGE_DIR, COCO_CAPTIONS_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model setup
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16).to(DEVICE)

optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

# Training loop
for epoch in range(EPOCHS):
    for batch_idx, (images, captions) in enumerate(dataloader):
        if images is None: continue  # Skip invalid data

        images = images.to(DEVICE)
        inputs = tokenizer(captions, padding="max_length", return_tensors="pt").to(DEVICE)
        text_embeddings = text_encoder(**inputs).last_hidden_state

        with torch.autocast(device_type=DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32):
            latents = pipeline.vae.encode(images).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

            noise = torch.randn_like(latents).to(DEVICE)
            noisy_latents = latents + noise

            pred_noise = pipeline.unet(noisy_latents, text_embeddings).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")

# Save fine-tuned model
pipeline.save_pretrained(OUTPUT_DIR)
