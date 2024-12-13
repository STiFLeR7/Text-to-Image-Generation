import torch
from transformers import pipeline
from realesrgan import RealESRGANer
from PIL import Image
import os

# Paths to save images
GENERATED_IMAGES_PATH = 'generated_images'
ENHANCED_IMAGES_PATH = 'enhanced_images'
COMPARISON_IMAGES_PATH = 'comparison_images'

# Ensure the output directories exist
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
os.makedirs(ENHANCED_IMAGES_PATH, exist_ok=True)
os.makedirs(COMPARISON_IMAGES_PATH, exist_ok=True)

# Authenticate Hugging Face if needed (uncomment and replace YOUR_TOKEN if required)
# from huggingface_hub import login
# login(token="YOUR_HUGGINGFACE_TOKEN")

# Load Stable Diffusion v2.1 model
generator = pipeline("text-to-image", model="stabilityai/stable-diffusion-2-1")

# Define function to generate image
def generate_image(prompt, output_path):
    image = generator(prompt, height=512, width=512)[0]["image"]
    image.save(output_path)
    print(f"Generated image saved at {output_path}")
    return image

# Load RealESRGAN model (ensure it's pre-trained)
real_esrgan_model = RealESRGANer(
    scale=4, 
    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x4plus.pth", 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Function to enhance image using RealESRGANer
def enhance_image(input_image_path, output_path):
    img = Image.open(input_image_path).convert("RGB")
    enhanced_img, _ = real_esrgan_model.enhance(img)
    enhanced_img.save(output_path)
    print(f"Enhanced image saved at {output_path}")
    return enhanced_img

# Example: Generate and enhance image
prompt = "A futuristic cityscape at night"
generated_image_path = os.path.join(GENERATED_IMAGES_PATH, "generated_image.jpg")
enhanced_image_path = os.path.join(ENHANCED_IMAGES_PATH, "enhanced_image.jpg")
comparison_image_path = os.path.join(COMPARISON_IMAGES_PATH, "comparison_image.jpg")

# Step 1: Generate image from text
generated_image = generate_image(prompt, generated_image_path)

# Step 2: Enhance the generated image using RealESRGANer
enhanced_image = enhance_image(generated_image_path, enhanced_image_path)

# Optional: Save comparison image (side by side)
comparison_image = Image.new('RGB', (generated_image.width * 2, generated_image.height))
comparison_image.paste(generated_image, (0, 0))
comparison_image.paste(enhanced_image, (generated_image.width, 0))
comparison_image.save(comparison_image_path)

print("Process complete!")
