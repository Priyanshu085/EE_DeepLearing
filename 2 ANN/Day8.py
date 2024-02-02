import tensorflow as tf
from keras import layers, models

# Generate synthetic data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0  # Normalize pixel values to be between 0 and 1
x_train = x_train.reshape((x_train.shape[0], -1))  # Flatten images

# Build a simple neural network
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using backpropagation
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Print model summary
model.summary()
