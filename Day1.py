import tensorflow as tf
import keras
from keras import layers

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train_flattened= X_train.reshape(len(X_train), 28*28)
X_test_flattened= X_test.reshape(len(X_test), 28*28)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(
    optimizer= 'adam',
    loss= 'sparse_categorical_crossentropy',
    metrics= ['accuracy']
)

model.fit(X_train_flattened, y_train, epochs=5)