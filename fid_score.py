from pytorch_fid import fid_score
import os

generated_images_path = ("./generated_images")
real_images_path = ("./real_images")

fid_value = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], batch_size=50, device='cuda', dims=2048)
print(f"FID Score: {fid_value}")\


#Prepare the Dataset: If you don't have a dataset of real images, you can use a publicly available dataset (e.g., CIFAR-10, CelebA) or any other dataset relevant to your project.
#Download the dataset and extract the images to a folder named real_images in the same directory as the script.\
#Generate Images: Run the script to generate images based on the text prompts provided in the prompts list. The generated images will be saved in the generated_images folder.
#Calculate FID Score: After generating the images, run the script to calculate the FID score between the real images and the generated images. The FID score provides a quantitative measure of the similarity between the distributions of real and generated images.\
#Analyze Results: Based on the FID score, you can evaluate the quality of the generated images. Lower FID scores indicate better image quality and higher similarity to real images. You can further fine-tune the models or experiment with different prompts to improve the image generation process.\