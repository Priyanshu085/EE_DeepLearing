import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the dataset (MNIST for simplicity)
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to the range [-1, 1]

# Set up the DCGAN model
latent_dim = 100

generator = models.Sequential()
generator.add(layers.Dense(7 * 7 * 256, input_shape=(latent_dim,), use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Reshape((7, 7, 256)))
generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

discriminator = models.Sequential()
discriminator.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
discriminator.add(layers.LeakyReLU())

discriminator.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
discriminator.add(layers.BatchNormalization())
discriminator.add(layers.LeakyReLU())

discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(1, activation='sigmoid'))

# Compile the discriminator (note: using binary crossentropy for a GAN)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Compile the full GAN model (generator + discriminator)
discriminator.trainable = False  # Freeze the discriminator during combined model training

gan = models.Sequential([generator, discriminator])
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy')

# Training loop
batch_size = 64
epochs = 20

for epoch in range(epochs):
    for _ in range(train_images.shape[0] // batch_size):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate fake images with the generator
        generated_images = generator.predict(noise)

        # Get a random batch of real images
        idx = np.random.randint(0, train_images.shape[0], batch_size)
        real_images = train_images[idx]

        # Labels for generated and real data
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Train the discriminator on real and fake data separately
        d_loss_real = discriminator.train_on_batch(real_images, valid)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake)

        # Combine losses for the discriminator
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator to fool the discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))

        # Update the generator weights only in the combined model
        g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress
    print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss[0]}, D Accuracy: {100 * d_loss[1]}, G Loss: {g_loss}")

    # Periodically save generated images
    if (epoch + 1) % 5 == 0:
        generated_images = generator.predict(np.random.normal(0, 1, (16, latent_dim)))

        # Rescale generated images to [0, 1]
        generated_images = 0.5 * generated_images + 0.5

        # Plot generated images
        fig, axs = plt.subplots(4, 4)
        cnt = 0
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
