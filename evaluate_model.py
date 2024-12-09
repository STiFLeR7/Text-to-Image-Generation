from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image
import torch

# Load the trained model and tokenizer
model_path = "D:/Text-to-Image-Generation/model_checkpoints/model_epoch_3.pt"  # Replace with your model checkpoint path
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load checkpoint
model.load_state_dict(torch.load(model_path))
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test image
image_path = "path_to_test_image.jpg"  # Replace with the path to your test image
image = Image.open(image_path).convert("RGB")

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt").to(device)

# Generate caption
with torch.no_grad():
    output_ids = model.generate(inputs["pixel_values"], max_length=50, num_beams=4, early_stopping=True)

# Decode the generated caption
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Caption: ", caption)
