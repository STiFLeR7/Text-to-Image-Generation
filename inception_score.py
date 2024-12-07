from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
import numpy as np

def inception_score(img_path):
    # Load the InceptionV3 model pre-trained on ImageNet
    model = InceptionV3()

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get predictions for the image
    preds = model.predict(img_array)

    # Calculate the Inception Score (IS)
    score = np.exp(np.mean(np.sum(preds * np.log(preds), axis=1)))
    return score

# Example usage for a generated image:
img_path = 'path_to_your_generated_image.png'
print(f"Inception Score for the image: {inception_score(img_path)}")
