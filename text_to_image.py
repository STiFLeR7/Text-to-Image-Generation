import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

pipe=StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to("cuda")

def generate_image_from_text(prompt: str):
   with torch.no_grad():
      image = pipe(prompt).images[0]
      return image 
def save_and_display_image(image: Image.Image, filename: str):
   image.save(filename)

   plt.imshow(image)
   plt.axis("off")
   plt.show()

prompt = "A futuristic city at sunset with flying cars."
generated_image = generate_image_from_text(prompt)
save_and_display_image(generated_image, "generated_image.png")

def enhance_image(image_path: str):
   img = cv2.imread(image_path)

   enhanced_image = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
   
   enhanced_pil_img = Image.fromarray(enhanced_image)

   return enhanced_pil_img

enhanced_image = enhance_image("generated_image.png")
save_and_display_image(enhanced_image, "enhanced_image.png")

def compare_multiple_outputs(prompt: str, num_samples: int = 5):
   images = []
   for i in range(num_samples):
      img = generate_image_from_text(prompt)
      images.append(img)

   fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
   for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis("off")
plt.show()

compare_multiple_outputs("A beautiful forest landscape at dawn", num_samples=5)
   