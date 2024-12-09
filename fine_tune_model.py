import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Paths
DATASET_DIR = "D:/Text-to-Image-Generation/preprocessed_data"
ANNOTATIONS_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
MODEL_SAVE_PATH = "D:/Text-to-Image-Generation/model_checkpoints"

# Dataset Class
class Flickr8kDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.captions = []
        self.image_names = []
        
        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                image_name, caption = parts[0].split("#")[0], parts[1]
                self.image_names.append(image_name)
                self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(image_path).convert("RGB")
        caption = self.captions[idx]
        return image, caption

# Fine-Tuning Function
def fine_tune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = Flickr8kDataset(ANNOTATIONS_FILE, DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load Pretrained Model
    model_name = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=5e-6)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Training Loop
    for epoch in range(3):  # Adjust as needed
        print(f"Epoch {epoch + 1}")
        for step, (image, caption) in enumerate(dataloader):
            optimizer.zero_grad()

            # Preprocess inputs
            image = image[0].resize((256, 256))
            image_tensor = pipeline.feature_extractor(image, return_tensors="pt").pixel_values.to(device)

            # Generate Latents
            latents = pipeline.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * 0.18215

            # Tokenize captions
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
            text_input = tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (1,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item()}")

    # Save Model
    pipeline.save_pretrained(MODEL_SAVE_PATH)
    print("Model fine-tuned and saved successfully.")

if __name__ == "__main__":
    fine_tune_model()
