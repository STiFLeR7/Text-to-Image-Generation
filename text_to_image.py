import os
import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import rgb_to_grayscale

# Directories
DATASET_PATH = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
TEXT_PATH = "D:/Flickr8k-Dataset/Flickr8k_text"
GENERATED_IMAGES_PATH = "./generated_images"
ENHANCED_IMAGES_PATH = "./enhanced_images"

def ensure_directories():
    os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
    os.makedirs(ENHANCED_IMAGES_PATH, exist_ok=True)

def load_stable_diffusion(model_name="stabilityai/stable-diffusion-2-1"):
    print(f"Loading Stable Diffusion model {model_name} from Hugging Face")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")
    return pipe

def enhance_image(image):
    print("Enhancing image...")
    return rgb_to_grayscale(image)

def generate_images(pipe, captions, num_images=5):
    for i, caption in enumerate(captions):
        print(f"Generating images for caption {i+1}/{len(captions)}: '{caption}'")
        for j in range(num_images):
            image = pipe(caption).images[0]
            image.save(os.path.join(GENERATED_IMAGES_PATH, f"image_{i}_{j}.png"))

def enhance_images():
    for img_file in os.listdir(GENERATED_IMAGES_PATH):
        img_path = os.path.join(GENERATED_IMAGES_PATH, img_file)
        enhanced_img_path = os.path.join(ENHANCED_IMAGES_PATH, img_file)
        img = Image.open(img_path).convert("RGB")
        enhanced_img = enhance_image(img)
        enhanced_img.save(enhanced_img_path)

def calculate_fid(generated_path, real_path):
    fid = FrechetInceptionDistance()
    for img_file in os.listdir(generated_path):
        img_path = os.path.join(generated_path, img_file)
        img = Image.open(img_path).convert("RGB")
        fid.update(torch.tensor(img).permute(2, 0, 1).unsqueeze(0), real=False)

    for img_file in os.listdir(real_path):
        img_path = os.path.join(real_path, img_file)
        img = Image.open(img_path).convert("RGB")
        fid.update(torch.tensor(img).permute(2, 0, 1).unsqueeze(0), real=True)

    return fid.compute()

def main(args):
    ensure_directories()

    # Load models manually
    pipe = load_stable_diffusion()  # Automatically loads from Hugging Face

    # Load captions
    with open(os.path.join(TEXT_PATH, "Flickr8k.token.txt"), "r") as f:
        captions = [line.strip().split("\t")[1] for line in f.readlines()]

    # Generate images
    generate_images(pipe, captions[: args.num_captions])

    # Enhance images
    enhance_images()

    # Calculate FID score
    fid_score = calculate_fid(GENERATED_IMAGES_PATH, DATASET_PATH)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Image Generation Project")
    parser.add_argument(
        "--num_captions",
        type=int,
        default=10,
        help="Number of captions to process for image generation.",
    )
    args = parser.parse_args()

    main(args)
