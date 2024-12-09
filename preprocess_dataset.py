import os
from PIL import Image

DATASET_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"  # Path to Flickr8k images
OUTPUT_DIR = "D:/Text-to-Image-Generation/preprocessed_data"  # Path for processed images

def preprocess_images(dataset_dir, output_dir, target_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    processed_count = 0
    for image_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        try:
            with Image.open(image_path) as img:
                img = img.resize(target_size)
                img.save(output_path, format='JPEG')
                processed_count += 1
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    print(f"Processing complete. Processed {processed_count} images.")

if __name__ == "__main__":
    preprocess_images(DATASET_DIR, OUTPUT_DIR)
