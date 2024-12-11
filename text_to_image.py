import os
import torch
from diffusers import StableDiffusionPipeline
from realesrgan import RealESRGANer
from PIL import Image
from torchvision.transforms import functional as F


# Directories for storing images
generated_images_dir = './generated_images'
enhanced_images_dir = './enhanced_images'
comparison_images_dir = './comparison_images'

# Create directories if they don't exist
os.makedirs(generated_images_dir, exist_ok=True)
os.makedirs(enhanced_images_dir, exist_ok=True)
os.makedirs(comparison_images_dir, exist_ok=True)

# Load the StableDiffusionv2.1 model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cuda")  # Move the model to GPU for faster processing if available

# Generate an image from a text prompt
def generate_image(prompt, output_path):
    image = pipe(prompt).images[0]
    image.save(output_path)
    return image

# Load the REALESRGAN model
def enhance_image(input_path, output_path):
    model = RealESRGANer.from_pretrained('RealESRGAN_x4')
    model = model.to("cuda")
    image = Image.open(input_path).convert("RGB")

    # Enhance the image
    enhanced_image = model.predict(image)
    enhanced_image.save(output_path)
    return enhanced_image

# Function to generate and enhance images
def generate_and_enhance_image(prompt):
    # Generate image
    generated_image_path = os.path.join(generated_images_dir, f"{prompt[:10]}.png")
    image = generate_image(prompt, generated_image_path)
    
    # Enhance image
    enhanced_image_path = os.path.join(enhanced_images_dir, f"{prompt[:10]}_enhanced.png")
    enhance_image(generated_image_path, enhanced_image_path)
    
    # Create a comparison image (before and after enhancement)
    comparison_image_path = os.path.join(comparison_images_dir, f"{prompt[:10]}_comparison.png")
    comparison_image = Image.new("RGB", (image.width * 2, image.height))
    comparison_image.paste(image, (0, 0))
    comparison_image.paste(Image.open(enhanced_image_path), (image.width, 0))
    comparison_image.save(comparison_image_path)
    
    return generated_image_path, enhanced_image_path, comparison_image_path

# Example usage
prompt = "A futuristic city at sunset"
generated_image, enhanced_image, comparison_image = generate_and_enhance_image(prompt)
print(f"Generated: {generated_image}, Enhanced: {enhanced_image}, Comparison: {comparison_image}")
