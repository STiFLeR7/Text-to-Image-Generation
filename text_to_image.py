import os
import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from realesrgan import RealESRGAN
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from torchmetrics.image.fid import FrechetInceptionDistance

# Directories
DATASET_PATH = 'D:/Flickr8k-Dataset/Flicker8k_Dataset'
TEXT_PATH = 'D:/Flickr8k-Dataset/Flickr8k_text'
GENERATED_IMAGES_PATH = './generated_images'
ENHANCED_IMAGES_PATH = './enhanced_images'

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
    os.makedirs(ENHANCED_IMAGES_PATH, exist_ok=True)

def load_stable_diffusion(model_name="stabilityai/stable-diffusion-2-1"):
    """Load Stable Diffusion pipeline from Hugging Face."""
    print(f"Loading Stable Diffusion model {model_name} from Hugging Face")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype="float16"
    ).to("cuda")
    return pipe

def load_real_esrgan(enhancer_checkpoint):
    """Load Real-ESRGAN model for image enhancement."""
    print(f"Loading Real-ESRGAN model from {enhancer_checkpoint}")
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    enhancer = RealESRGAN("cuda", model=model, scale=4)
    enhancer.load_weights(enhancer_checkpoint)
    return enhancer

def generate_images(pipe, captions, num_images=5):
    """Generate images from text captions."""
    for i, caption in enumerate(captions):
        print(f"Generating images for caption {i+1}/{len(captions)}: '{caption}'")
        for j in range(num_images):
            image = pipe(caption).images[0]
            image.save(os.path.join(GENERATED_IMAGES_PATH, f"image_{i}_{j}.png"))

def enhance_images(enhancer):
    """Enhance images using RealESRGAN."""
    for img_file in os.listdir(GENERATED_IMAGES_PATH):
        img_path = os.path.join(GENERATED_IMAGES_PATH, img_file)
        enhanced_img_path = os.path.join(ENHANCED_IMAGES_PATH, img_file)
        img = Image.open(img_path)
        enhanced_img = enhancer.enhance(img)
        enhanced_img.save(enhanced_img_path)

def calculate_fid(generated_path, real_path):
    """Calculate FID score."""
    fid = FrechetInceptionDistance()
    for img_file in os.listdir(generated_path):
        img_path = os.path.join(generated_path, img_file)
        img = Image.open(img_path).convert("RGB")
        fid.update(img, real=False)

    for img_file in os.listdir(real_path):
        img_path = os.path.join(real_path, img_file)
        img = Image.open(img_path).convert("RGB")
        fid.update(img, real=True)

    return fid.compute()

def main(args):
    ensure_directories()

    # Load models manually
    pipe = load_stable_diffusion()  # Automatically loads from Hugging Face
    enhancer = load_real_esrgan(args.enhancer_checkpoint)

    # Load captions
    with open(os.path.join(TEXT_PATH, 'Flickr8k.token.txt'), 'r') as f:
        captions = [line.strip().split('\t')[1] for line in f.readlines()]

    # Generate images
    generate_images(pipe, captions[:args.num_captions])

    # Enhance images
    enhance_images(enhancer)

    # Calculate FID score
    fid_score = calculate_fid(GENERATED_IMAGES_PATH, DATASET_PATH)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Image Generation Project")
    parser.add_argument("--enhancer_checkpoint", type=str, default="./checkpoints/RealESRGAN_x4plus.pth", 
                        help="Path to the RealESRGAN checkpoint.")
    parser.add_argument("--num_captions", type=int, default=10, 
                        help="Number of captions to process for image generation.")
    args = parser.parse_args()

    main(args)
