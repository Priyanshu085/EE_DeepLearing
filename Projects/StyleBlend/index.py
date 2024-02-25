import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras.preprocessing import image as kp_image
from keras import Model
from keras import layers
import time

# Load VGG19 model pretrained on ImageNet dataset
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = kp_image.load_img(path_to_img)
    img = kp_image.img_to_array(img)

    img = tf.image.resize(img, (max_dim,max_dim))
    img = img/255.0
    return img

# Function to display the image
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

# Function to preprocess image for VGG
def preprocess_img(image):
    image = tf.cast(image, dtype=tf.float32)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image

# Function to deprocess image for display
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x