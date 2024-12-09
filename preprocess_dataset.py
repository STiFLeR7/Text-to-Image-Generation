import os
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Paths
DATASET_DIR = "path_to_dataset/images"  # Update with your dataset's image folder
ANNOTATIONS_FILE = "path_to_dataset/captions.json"  # Update with your captions JSON file
OUTPUT_DIR = "preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resize and normalize images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_images():
    print("Preprocessing images...")
    for image_file in tqdm(os.listdir(DATASET_DIR)):
        try:
            img_path = os.path.join(DATASET_DIR, image_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            output_path = os.path.join(OUTPUT_DIR, f"processed_{image_file}")
            torch.save(img_tensor, output_path)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def preprocess_captions():
    print("Preprocessing captions...")
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)
    processed_captions = {item['image_id']: item['caption'] for item in annotations['annotations']}
    with open(os.path.join(OUTPUT_DIR, "captions.json"), "w") as f:
        json.dump(processed_captions, f)

if __name__ == "__main__":
    preprocess_images()
    preprocess_captions()
    print("Preprocessing complete.")
