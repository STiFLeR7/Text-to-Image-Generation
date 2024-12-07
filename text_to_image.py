from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os
from realesrgan import RealESRGAN
from PIL import Image

# Initialize the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original", torch_dtype=torch.float16)
pipe.to("cuda")

# Initialize RealESRGAN for image upscaling
esrgan_model = RealESRGAN.from_pretrained('RealESRGAN_x4')
esrgan_model = esrgan_model.to('cuda')

# Function to generate an image from a text prompt
def generate_image_from_prompt(prompt):
    # Generate image using the Stable Diffusion model
    image = pipe(prompt).images[0]
    
    # Enhance image using ESRGAN (optional)
    upscaled_image = esrgan_model.predict(image)

    # Save the image
    sanitized_prompt = prompt.replace(" ", "_")[:50]  # Avoid long filenames
    filename = f"{sanitized_prompt}.png"
    upscaled_image.save(filename)

    # Display the upscaled image
    plt.imshow(upscaled_image)
    plt.axis('off')
    plt.title(f"Generated Image for: {prompt}")
    plt.show()

    return filename

# List of diverse text prompts to test
prompts = [
    "A close-up of a red apple on a white plate.",
    "A futuristic city skyline at sunset with flying cars.",
    "A serene landscape with mountains and a lake during sunrise.",
    "A vintage car parked in front of an old brick building.",
    "A cat wearing a spacesuit on the moon.",
    "A portrait of a woman in the style of Picasso.",
    "A chaotic swirl of colors.",
    "A dragon flying over a medieval castle.",
    "A beautiful beach at sunset with waves crashing."
]

# Generate and save images for each prompt, then display them
for prompt in prompts:
    print(f"Generating image for: {prompt}")
    generated_image = generate_image_from_prompt(prompt)
    print(f"Saved generated image: {generated_image}")
