import numpy as np
import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Extract features using the pre-trained model
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features

# Example usage
img_path = 'example.jpg'
features = extract_features(img_path)
print(features.shape)  # Output shape should be (1, 7, 7, 512)