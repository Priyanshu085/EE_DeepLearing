import tensorflow as tf
import keras
from keras import layers

#Learning Loss Function
# Path: Day3.py

y_true = tf.random.normal(shape=(1000, 1))
y_pred = tf.random.normal(shape=(1000, 1))

# Loss Function

# Using MSE in TensorFlow
mse_loss = tf.losses.MeanSquaredError()(y_true, y_pred)
print(mse_loss.numpy())

# Binary Cross Entropy 
bce_loss = tf.losses.BinaryCrossentropy()(y_true, y_pred)
print(bce_loss.numpy())

# Categorical Cross Entropy
cce_loss = tf.losses.CategoricalCrossentropy()(y_true, y_pred)
print(cce_loss.numpy())

# Sparse Categorical Cross Entropy
sp_cce_loss = tf.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
print(sp_cce_loss.numpy())
  
# Optimsation Algorithms

# Using SGD optimizer in TensorFlow
sgd_optimizer = tf.optimizers.SGD(learning_rate=0.01)
print(sgd_optimizer)

# Using Adam optimizer in TensorFlow
adam_optimizer = tf.optimizers.Adam(learning_rate=0.01)
print(adam_optimizer)

# Using RMSProp optimizer in TensorFlow
rmsprop_optimizer = tf.optimizers.RMSprop(learning_rate=0.01)
print(rmsprop_optimizer)

# Using Adagrad optimizer in TensorFlow
adagrad_optimizer = tf.optimizers.Adagrad(learning_rate=0.01)
print(adagrad_optimizer)
