import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images - 127.5) / 127.5  # Normalize pixel values to be between -1 and 1
train_images = np.expand_dims(train_images, axis=-1)

# Set up the GAN model
latent_dim = 100

# Generator model
generator = models.Sequential([
    layers.Dense(7 * 7 * 128, input_dim=latent_dim),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'),
    layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', activation='tanh')
])

# Discriminator model
discriminator = models.Sequential([
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# Combined GAN model (stacking generator and discriminator)
discriminator.trainable = False  # Freeze discriminator during generator training
gan = models.Sequential([generator, discriminator])

# Compile models
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Function to train the GAN
def train_gan(epochs=1, batch_size=128):
    batch_count = train_images.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):
            # Generate random noise as input to the generator
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

            # Generate fake images using the generator
            generated_images = generator.predict(noise)

            # Get a random batch of real images from the dataset
            real_images = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]

            # Create labels for the generated and real images
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))

            # Train the discriminator on real and fake images
            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

            # Train the generator to fool the discriminator
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            labels_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, labels_gan)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}, [D loss: {0.5 * np.add(d_loss_real, d_loss_fake)[0]}, acc.: {0.5 * np.add(d_loss_real, d_loss_fake)[1]}] [G loss: {g_loss}]")

# Train the GAN
train_gan(epochs=30, batch_size=128)

# Generate and plot images
def generate_and_plot_images(generator, epoch, latent_dim):
    noise = np.random.normal(0, 1, size=[16, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale pixel values to be between 0 and 1

    plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.show()

# Generate images at the end of training
generate_and_plot_images(generator, epochs-1, latent_dim)
