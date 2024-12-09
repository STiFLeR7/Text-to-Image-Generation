import os
import json
from tqdm import tqdm
from PIL import Image

# Update these paths
DATASET_DIR = "D:/COCO-Dataset/train2014/train2014"
ANNOTATIONS_FILE = "D:/COCO-Dataset/annotations_trainval2014/annotations/captions_train2014.json"
OUTPUT_DIR = "preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file for errors
ERROR_LOG = "error_log.txt"

# Function to preprocess images
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((256, 256))
        return img_resized
    except Exception as e:
        raise RuntimeError(f"Error processing {image_path}: {e}")

# Load annotations
with open(ANNOTATIONS_FILE, 'r') as f:
    annotations = json.load(f)

# Map image IDs to file names
image_id_to_file = {img["id"]: img["file_name"] for img in annotations["images"]}

# Process dataset
processed_count = 0
total_count = len(annotations["images"])
error_count = 0

print(f"Processing {total_count} images...")
with open(ERROR_LOG, "w") as error_log:
    for annotation in tqdm(annotations["images"], desc="Processing images"):
        image_id = annotation["id"]
        image_file = image_id_to_file.get(image_id)
        image_path = os.path.join(DATASET_DIR, image_file)
        output_path = os.path.join(OUTPUT_DIR, f"processed_{image_file}")

        # Skip already processed files
        if os.path.exists(output_path):
            processed_count += 1
            continue

        try:
            processed_image = preprocess_image(image_path)
            processed_image.save(output_path)
            processed_count += 1
        except Exception as e:
            error_count += 1
            error_log.write(f"{image_file}: {e}\n")
            print(f"Error processing {image_file}: {e}")

print(f"Processing complete. Successfully processed {processed_count} out of {total_count} images.")
print(f"Errors encountered with {error_count} images. Check {ERROR_LOG} for details.")
