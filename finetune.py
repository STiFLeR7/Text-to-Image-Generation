import os
import torch

# Define paths
image_folder = "D:/Flickr8k-Dataset/Images"  # Path to the folder containing images
dataset_file = "D:/Flickr8k-Dataset/captions.txt"  # Path to captions.txt

# Check if paths exist
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Image folder not found: {image_folder}")

if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

# Process captions.txt
def process_dataset(image_folder, dataset_file):
    dataset = []
    with open(dataset_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            # Split the line into image filename and caption using the dot as a separator
            parts = line.strip().split(".", 1)  # Split at the first dot
            if len(parts) != 2:
                print(f"[Warning] Skipping invalid line {line_num}: {line.strip()}")
                continue

            image_filename, caption = parts
            image_filename = image_filename.strip() + ".jpg"  # Ensure the filename ends with ".jpg"
            image_path = os.path.join(image_folder, image_filename)

            # Check if the image exists
            if not os.path.exists(image_path):
                print(f"[Warning] Image not found: {image_filename}, skipping...")
                continue

            # Add valid image-caption pair to the dataset
            dataset.append((image_path, caption.strip()))

    return dataset

# Load dataset
dataset = process_dataset(image_folder, dataset_file)

if len(dataset) == 0:
    raise ValueError("No valid image-caption pairs found. Check your dataset.")

print(f"Loaded {len(dataset)} image-caption pairs.")

# Fine-tuning model logic here
# Placeholder: Replace with actual fine-tuning code for your ML model
def finetune_model(dataset):
    # Example processing loop for dataset
    for image_path, caption in dataset:
        # Debugging: Print each pair
        print(f"Processing Image: {image_path}, Caption: {caption}")

        # Your fine-tuning code would go here
        pass

# Start fine-tuning
print("Starting fine-tuning...")
finetune_model(dataset)
print("Fine-tuning complete!")
