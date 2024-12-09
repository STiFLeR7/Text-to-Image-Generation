import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Ensure old images are deleted before new ones are generated
def clear_old_images():
    # Delete old images from generated_image, enhanced_image, and comparison_image folders
    generated_folder = "generated_image"
    enhanced_folder = "enhanced_image"
    comparison_folder = "comparison_image"
    
    for folder in [generated_folder, enhanced_folder, comparison_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)  # Create folder if it doesn't exist
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Load pretrained model and tokenizer from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU for faster processing

# Function to generate an image from a text prompt
def generate_image_from_text(prompt: str):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

# Function to save and display generated image
def save_and_display_image(image: Image.Image, filename: str):
    image.save(filename)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Function to enhance image resolution (optional post-processing)
def enhance_image(image_path: str):
    img = cv2.imread(image_path)
    enhanced_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    enhanced_pil_img = Image.fromarray(enhanced_img)
    return enhanced_pil_img

# Function to compare multiple outputs by generating 5 variations of the same prompt
def compare_multiple_outputs(prompt: str, num_samples=5):
    comparison_images = []
    for i in range(num_samples):
        comparison_image = generate_image_from_text(prompt)
        comparison_image.save(f"comparison_image/image_{i+1}.png")
        comparison_images.append(comparison_image)
    return comparison_images

# Clear old images from previous runs
clear_old_images()

# Generate 15 images for a given prompt
prompt = "A futuristic city at sunset with flying cars."
for i in range(15):
    generated_image = generate_image_from_text(prompt)
    generated_image.save(f"generated_image/generated_image_{i+1}.png")
    save_and_display_image(generated_image, f"generated_image/generated_image_{i+1}.png")

    # Enhance and save the image
    enhanced_image = enhance_image(f"generated_image/generated_image_{i+1}.png")
    enhanced_image.save(f"enhanced_image/enhanced_image_{i+1}.png")
    save_and_display_image(enhanced_image, f"enhanced_image/enhanced_image_{i+1}.png")

# Compare multiple outputs (5 variations)
compare_multiple_outputs("A beautiful forest landscape at dawn", num_samples=5)
