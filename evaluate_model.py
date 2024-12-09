import os
import random
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch

# Load the trained model and tokenizer
model_path = "D:/Text-to-Image-Generation/model_checkpoints/model_epoch_3.pt"  # Replace with your model checkpoint path
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load checkpoint
model.load_state_dict(torch.load(model_path))
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Path to your dataset images
dataset_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"  # Update with the correct path to your dataset images

# Get a list of image filenames from the dataset directory
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]  # Assuming images are in .jpg format

# Select a random image from the dataset
image_path = os.path.join(dataset_dir, random.choice(image_files))

# Check if the image exists
if os.path.exists(image_path):
    print(f"Image found at {image_path}")
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(inputs["pixel_values"], max_length=50, num_beams=4, early_stopping=True)

    # Decode the generated caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated Caption: ", caption)
else:
    print(f"Image not found at {image_path}")
