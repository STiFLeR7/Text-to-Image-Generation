import os
import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from torchvision import models
from torchvision import transforms
from scipy.linalg import sqrtm
import numpy as np
from tqdm import tqdm

# Directories
DATASET_PATH = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
TEXT_PATH = "D:/Flickr8k-Dataset/Flickr8k_text"
GENERATED_IMAGES_PATH = "./generated_images"
ENHANCED_IMAGES_PATH = "./enhanced_images"

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
    os.makedirs(ENHANCED_IMAGES_PATH, exist_ok=True)

def load_stable_diffusion(model_name="stabilityai/stable-diffusion-2-1"):
    """Load Stable Diffusion pipeline from Hugging Face."""
    print(f"Loading Stable Diffusion model {model_name} from Hugging Face")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")
    return pipe

def enhance_image(image):
    """Dummy enhancement process using grayscale conversion."""
    print("Enhancing image...")
    return image.convert("L")

def generate_images(pipe, captions, num_images=5):
    """Generate images from text captions."""
    for i, caption in enumerate(captions):
        print(f"Generating images for caption {i+1}/{len(captions)}: '{caption}'")
        for j in range(num_images):
            image = pipe(caption).images[0]
            image.save(os.path.join(GENERATED_IMAGES_PATH, f"image_{i}_{j}.png"))

def enhance_images():
    """Enhance images using a simple process."""
    for img_file in os.listdir(GENERATED_IMAGES_PATH):
        img_path = os.path.join(GENERATED_IMAGES_PATH, img_file)
        enhanced_img_path = os.path.join(ENHANCED_IMAGES_PATH, img_file)
        img = Image.open(img_path).convert("RGB")
        enhanced_img = enhance_image(img)
        enhanced_img.save(enhanced_img_path)

def extract_features(image_paths, model, device="cpu"):
    """Extract features using InceptionV3."""
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features = []
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            output = model(img)
            features.append(output.cpu().numpy())

    return np.concatenate(features, axis=0)

def calculate_fid(real_features, generated_features):
    """Calculate the FID score."""
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

def get_device():
    """Get the device (CPU or GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_fid_score():
    """Main FID calculation function."""
    device = get_device()
    print(f"Using device: {device}")

    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
    model.eval()

    real_image_paths = [
        os.path.join(ENHANCED_IMAGES_PATH, fname)
        for fname in os.listdir(ENHANCED_IMAGES_PATH)
        if fname.endswith(".png")
    ]
    generated_image_paths = [
        os.path.join(GENERATED_IMAGES_PATH, fname)
        for fname in os.listdir(GENERATED_IMAGES_PATH)
        if fname.endswith(".png")
    ]

    print(f"Extracting features from real images...")
    real_features = extract_features(real_image_paths, model, device=device)

    print(f"Extracting features from generated images...")
    generated_features = extract_features(generated_image_paths, model, device=device)

    fid_score = calculate_fid(real_features, generated_features)
    print(f"FID Score: {fid_score}")
    return fid_score

def main(args):
    ensure_directories()

    pipe = load_stable_diffusion()

    with open(os.path.join(TEXT_PATH, "Flickr8k.token.txt"), "r") as f:
        captions = [line.strip().split("\t")[1] for line in f.readlines()]

    generate_images(pipe, captions[: args.num_captions])

    enhance_images()

    fid_score = calculate_fid_score()
    print(f"Final FID Score: {fid_score}")

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
