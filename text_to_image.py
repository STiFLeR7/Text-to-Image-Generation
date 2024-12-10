import os
from diffusers import StableDiffusionPipeline
import torch
from realesrgan import RealESRGAN
from PIL import Image
import shutil

# Set up directories
generated_dir = "generated_images"
enhanced_dir = "enhanced_images"
comparison_dir = "comparison_images"

# Create directories if they don't exist
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(enhanced_dir, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)

# Load the Stable Diffusion model (Stable Diffusion 2.1)
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Load Real-ESRGAN for enhancing images
esrgan = RealESRGAN.from_pretrained('RealESRGAN_x4', device='cuda')
esrgan.to('cuda')

# Function to generate image from a text prompt
def generate_image(prompt, output_path="generated_images/image.png"):
    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return image

# Function to enhance an image using Real-ESRGAN
def enhance_image(input_image_path, output_image_path="enhanced_images/image.png"):
    print(f"Enhancing image: {input_image_path}")
    image = Image.open(input_image_path).convert("RGB")
    enhanced_image = esrgan.enhance(image)
    enhanced_image.save(output_image_path)
    print(f"Enhanced image saved to {output_image_path}")
    return enhanced_image

# Function to generate a comparison between the original and enhanced images
def create_comparison(original_image_path, enhanced_image_path, comparison_image_path="comparison_images/comparison.png"):
    print(f"Creating comparison image: {comparison_image_path}")
    original = Image.open(original_image_path)
    enhanced = Image.open(enhanced_image_path)

    # Resize images for comparison (if necessary)
    enhanced = enhanced.resize(original.size)

    # Combine images side by side
    comparison = Image.new('RGB', (original.width + enhanced.width, original.height))
    comparison.paste(original, (0, 0))
    comparison.paste(enhanced, (original.width, 0))
    
    comparison.save(comparison_image_path)
    print(f"Comparison image saved to {comparison_image_path}")
    return comparison

# Example usage
if __name__ == "__main__":
    prompt = "A futuristic city with flying cars and neon lights"

    # Generate image
    generated_image = generate_image(prompt, os.path.join(generated_dir, "futuristic_city.png"))

    # Enhance the generated image using Real-ESRGAN
    enhanced_image = enhance_image(os.path.join(generated_dir, "futuristic_city.png"), os.path.join(enhanced_dir, "futuristic_city_enhanced.png"))

    # Create a comparison image between original and enhanced
    create_comparison(os.path.join(generated_dir, "futuristic_city.png"), os.path.join(enhanced_dir, "futuristic_city_enhanced.png"), os.path.join(comparison_dir, "futuristic_city_comparison.png"))
