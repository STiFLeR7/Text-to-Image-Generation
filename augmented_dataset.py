import os
import random
from PIL import Image
from torchvision import transforms
import shutil

# Define paths
DATASET_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
ANNOTATIONS_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
AUGMENTED_DIR = "D:/Text-to-Image-Generation/Flicker8k_Augmented"

# Create directory for augmented images
if not os.path.exists(AUGMENTED_DIR):
    os.makedirs(AUGMENTED_DIR)

# Define augmentation transformations
augmentations = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
])

# Helper function to save augmented images
def augment_and_save(image_path, image_name):
    try:
        image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
        for i in range(5):  # Generate 5 augmentations per image
            augmented_image = augmentations(image)
            augmented_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.jpg"
            augmented_path = os.path.join(AUGMENTED_DIR, augmented_name)
            augmented_image.save(augmented_path)
    except Exception as e:
        print(f"Error augmenting {image_name}: {e}")

# Process dataset
print("Starting dataset augmentation...")
for image_name in os.listdir(DATASET_DIR):
    image_path = os.path.join(DATASET_DIR, image_name)
    if os.path.isfile(image_path):
        augment_and_save(image_path, image_name)
print("Dataset augmentation complete!")

# Update annotations file
print("Updating annotations for augmented dataset...")
augmented_annotations = []
with open(ANNOTATIONS_FILE, 'r') as f:
    for line in f:
        original_image, caption = line.strip().split('\t')
        original_image_name = original_image.split('#')[0]
        for i in range(5):  # Generate new annotations for augmented images
            augmented_name = f"{os.path.splitext(original_image_name)[0]}_aug_{i}.jpg"
            augmented_annotations.append(f"{augmented_name}#{i}\t{caption}\n")

augmented_annotations_path = os.path.join(AUGMENTED_DIR, "Flickr8k_augmented.token")
with open(augmented_annotations_path, 'w') as f:
    f.writelines(augmented_annotations)
print(f"Annotations saved to {augmented_annotations_path}")

print("Dataset augmentation and annotation update complete.")
