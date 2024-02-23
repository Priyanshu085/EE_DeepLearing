import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, input_sequence_length, target_sequence_length, dropout_rate=0.1):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.dropout_rate = dropout_rate
        
    def transformer_encoder_layer(self, inputs, mask, training):
        attention_output = self.multi_head_attention(inputs, mask)
        attention_output = Dropout(self.dropout_rate)(attention_output, training=training)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        ffn_output = self.feed_forward_network(attention_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output, training=training)
        encoder_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        return encoder_output
    
    def multi_head_attention(self, inputs, mask):
        d_k = self.d_model // self.num_heads
        queries = Dense(self.d_model)(inputs)
        keys = Dense(self.d_model)(inputs)
        values = Dense(self.d_model)(inputs)
        
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        
        scaled_attention_logits = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, values)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (tf.shape(output)[0], -1, self.d_model))
        return output
    
    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def feed_forward_network(self, inputs):
        ff_hidden_layer = Dense(self.dff, activation='relu')(inputs)
        ff_output = Dense(self.d_model)(ff_hidden_layer)
        return ff_output
    
    def build_model(self):
        inputs = Input(shape=(self.input_sequence_length,))
        mask = tf.linalg.band_part(tf.ones((self.input_sequence_length, self.input_sequence_length)), -1, 0)
        
        encoder_output = self.transformer_encoder_layer(inputs, mask, training=True)
        outputs = Dense(self.target_vocab_size, activation='softmax')(encoder_output)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
