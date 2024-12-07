
# **Text-to-Image Generation with Stable Diffusion and ESRGAN**

This project implements a Text-to-Image generation pipeline using Stable Diffusion and Real-ESRGAN for high-resolution image enhancement. The pipeline generates images based on descriptive text prompts and then enhances the resolution using ESRGAN. The generated images are evaluated using Inception Score (IS) to assess the quality and diversity of the output.


## **Key Features**
**Text-to-Image Generation**: Generates images based on text prompts using Stable Diffusion.

**Resolution Enhancement**: Enhances the resolution of generated images using Real-ESRGAN.

**Multiple Prompt Support**: Handles diverse prompts such as "A futuristic city skyline" and "A cat wearing a spacesuit".

**Inception Score (IS)**: Evaluates the quality and diversity of the generated images using the Inception Score metric.
## **Technologies Used**

**Stable Diffusion**: A powerful text-to-image generation model that generates realistic images from text descriptions.

**Real-ESRGAN**: A state-of-the-art image upscaling model that enhances the resolution of images while preserving quality.

**TensorFlow**: Used for the InceptionV3 model to compute the Inception Score (IS), which measures the quality of generated images.

**Matplotlib**: Used to display generated images inline for visual inspection.
## **Setup and Installation**

1. Install Dependencies

    You need to install the necessary libraries and dependencies for this project.

```bash
pip install torch torchvision
pip install diffusers
pip install realesrgan
pip install tensorflow
pip install matplotlib
pip install pytorch-fid
```

2. Clone the Repository

    Clone the repository to get started with the project:

```bash
git clone https://github.com/STiFLeR7/text-to-image-generation.git
cd text-to-image-generation
```



## **Usage**

1. **Generate Images from Text Prompts**

    To generate images, run the ```text_to_image.py``` script. It will generate images based on a list of predefined prompts and enhance their resolution using Real-ESRGAN.

```python text_to_image.py```

The script will:
    Generate images for multiple diverse prompts.

    Enhance the generated images using Real-ESRGAN.

    Save the generated images with filenames based on the prompt.

    Display the images using Matplotlib for visual inspection.

2. **Evaluate Image Quality Using Inception Score (IS)**

    The Inception Score (IS) is calculated for each generated image to evaluate its quality and diversity. The ```inception_score``` function uses the InceptionV3 model from TensorFlow.

    To calculate the Inception Score for an image:
    ```
    from inception_score import inception_score

    img_path = 'path_to_your_generated_image.png'
    print(f"Inception Score for the image: {inception_score(img_path)}")
    ```

## **Project Flow**

1. Text-to-Image Generation:

    The script generates images from a list of prompts using Stable Diffusion.
    The images are then passed through **Real-ESRGAN** for resolution  enhancement.
    
2. Saving and Displaying Images:

    Generated images are saved with filenames based on the prompts.
    Each image is displayed using **Matplotlib** for visual inspection.

3. Image Evaluation:

    The **Inception Score (IS)** is calculated for each image to measure its quality and diversity.
## **Future Work**

**Fine-Tuning**: Fine-tune the Stable Diffusion model on a specific dataset for improved results.

**Improved Prompt Engineering**: Experiment with more sophisticated and creative prompts to get better and more diverse outputs.

**Image Quality Enhancement**: Explore additional image enhancement techniques to further improve the resolution and clarity.

**Additional Evaluation Metrics**: Implement more evaluation metrics like FID (Fréchet Inception Distance) for comparing generated images with real ones.
## **Contributing**

Feel free to fork the repository and contribute by submitting issues or pull requests. Your contributions to improving this project are welcome!

