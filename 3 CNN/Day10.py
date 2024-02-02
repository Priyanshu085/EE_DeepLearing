import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Create an instance of the ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5]
)

# Load and preprocess an example image
img = tf.keras.preprocessing.image.load_img('path_to_image.jpg')
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

# Generate augmented images
augmented_images = datagen.flow(img_array, batch_size=1)

# Visualize the original and augmented images
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img_array[0].astype('uint8'))
axs[0].title.set_text('Original Image')

for i in range(1):
    augmented_image = augmented_images.next()[0].astype('uint8')
    axs[1].imshow(augmented_image)
    axs[1].title.set_text('Augmented Image')

plt.show()