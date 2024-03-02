import numpy as np
import tensorflow as tf
from keras import layers, models
import os
import matplotlib.pyplot as plt

# Load and preprocess data
# Assuming you have already collected and preprocessed the dataset

# Define generator model
def build_generator(latent_dim):
    generator_input = Input(shape=(latent_dim,))
    x = Dense(128 * 8 * 8)(generator_input)
    x = Reshape((8, 8, 128))(x)
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(512, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(3, kernel_size=3, activation='tanh', padding='same')(x)
    generator = Model(generator_input, x)
    return generator

# Define discriminator model
def build_discriminator(input_shape):
    discriminator_input = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(discriminator_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, x)
    return discriminator

# Define GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return gan

# Define training function
def train_gan(generator, discriminator, gan, iterations, batch_size, latent_dim, anime_images):
    for iteration in range(iterations):
        # Train discriminator
        idx = np.random.randint(0, anime_images.shape[0], batch_size)
        real_images = anime_images[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        X = np.concatenate([real_images, fake_images])
        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 0.9  # Label smoothing
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gen)
        
        # Print progress
        print(f'Iteration {iteration + 1}/{iterations} [D loss: {d_loss:.4f}, G loss: {g_loss:.4f}]')

# Define hyperparameters
latent_dim = 100
iterations = 10000
batch_size = 128

# Build and compile discriminator
discriminator = build_discriminator(anime_images.shape[1:])
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

# Build generator
generator = build_generator(latent_dim)

# Build and compile GAN
gan = build_gan(generator, discriminator)

# Train GAN
train_gan(generator, discriminator, gan, iterations, batch_size, latent_dim, anime_images)