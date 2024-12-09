import os
import json
from tqdm import tqdm

DATASET_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"  # Path to the dataset images
ANNOTATIONS_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"  # Path to the captions file
OUTPUT_FILE = "D:/Text-to-Image-Generation/processed_captions.json"  # Output path for processed captions

def preprocess_flickr8k(dataset_dir, annotations_file, output_file):
    # Load captions from annotations file
    captions = {}
    with open(annotations_file, "r") as f:
        for line in tqdm(f.readlines(), desc="Processing captions"):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            image_id, caption = parts
            # Remove the #<number> suffix from the image ID
            image_id = image_id.split("#")[0]
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)

    # Check if all images exist
    valid_captions = {}
    for image_id, caption_list in captions.items():
        image_path = os.path.join(dataset_dir, image_id)
        if os.path.exists(image_path):
            valid_captions[image_id] = caption_list

    # Save processed captions to a JSON file
    with open(output_file, "w") as f:
        json.dump(valid_captions, f, indent=4)

    print(f"Processed {len(valid_captions)} images with captions.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    preprocess_flickr8k(DATASET_DIR, ANNOTATIONS_FILE, OUTPUT_FILE)
