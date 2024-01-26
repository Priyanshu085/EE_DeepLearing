import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images.astype('float32') - 127.5) / 127.5  # Normalize to [-1, 1]
train_images = train_images[..., tf.newaxis]

# Build the generator model
generator = models.Sequential([
    layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Reshape((7, 7, 256)),
    layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])

# Build the discriminator model
discriminator = models.Sequential([
    layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
    layers.LeakyReLU(),
    layers.Dropout(0.3),

    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.LeakyReLU(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(1)
])

# Compile the models
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False

gan = models.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
def train_gan(epochs=50, batch_size=128):
    for epoch in range(epochs):
        for _ in range(train_images.shape[0] // batch_size):
            noise = tf.random.normal([batch_size, 100])

            generated_images = generator(noise)

            real_images = train_images[np.random.choice(train_images.shape[0], batch_size, replace=False)]

            labels_real = tf.ones((batch_size, 1))
            labels_fake = tf.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = tf.random.normal([batch_size, 100])
            labels_gan = tf.ones((batch_size, 1))

            g_loss = gan.train_on_batch(noise, labels_gan)

        print(f"Epoch {epoch}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

# Train the GAN
train_gan(epochs=50, batch_size=128)

# Generate synthetic images
noise = tf.random.normal([16, 100])
generated_images = generator(noise)

# Visualize the generated images
plt.figure(figsize=(4, 4))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((generated_images[i, :, :, 0] + 1) / 2, cmap='gray')
    plt.axis('off')
plt.show()
