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