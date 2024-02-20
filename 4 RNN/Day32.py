import tensorflow as tf
from keras.layers import Layer

class SoftAttention(Layer):
    def __init__(self):
        super(SoftAttention, self).__init__()

    def build(self, input_shape):
        # Initialize weights for computing attention scores
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(SoftAttention, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        scores = tf.matmul(inputs, self.W)
        scores = tf.squeeze(scores, axis=-1)

        # Apply softmax to obtain attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Compute context vector as the weighted sum of inputs
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)

        return context_vector, attention_weights

# Example usage
input_dim = 64
sequence_length = 10

# Sample input sequence
inputs = tf.random.normal(shape=(1, sequence_length, input_dim))

# Apply soft attention mechanism
soft_attention = SoftAttention()
context_vector, attention_weights = soft_attention(inputs)

print("Context vector shape:", context_vector.shape)
print("Attention weights shape:", attention_weights.shape)
