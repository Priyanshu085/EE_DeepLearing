import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# Generate random input sequences
input_seqs = ['apple', 'banana', 'cherry', 'date']
target_seqs = ['elppa', 'ananab', 'yrrehc', 'etad']

# Define tokenizer dictionaries
input_tokenizer = {char: i+1 for i, char in enumerate(set(''.join(input_seqs)))}
target_tokenizer = {char: i+1 for i, char in enumerate(set(''.join(target_seqs)))}

# Convert input and target sequences to integer sequences
X = [[input_tokenizer[char] for char in seq] for seq in input_seqs]
y = [[target_tokenizer[char] for char in seq] for seq in target_seqs]

# Pad sequences to the same length
max_seq_length = max(max(len(seq) for seq in X), max(len(seq) for seq in y))
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_seq_length, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_seq_length, padding='post')

# Define the seq2seq model
input_seq = Input(shape=(max_seq_length,))
encoder = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder(input_seq)
encoder_states = [state_h, state_c]

decoder_input_seq = Input(shape=(max_seq_length,))
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_input_seq, initial_state=encoder_states)
decoder_dense = Dense(len(target_tokenizer)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile the model
model = Model([input_seq, decoder_input_seq], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit([X, X], y, epochs=100, verbose=2)

# Inference mode (sampling)
encoder_model = Model(input_seq, encoder_states)

decoder_state_input_h = Input(shape=(64,))
decoder_state_input_c = Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_input_seq, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_input_seq] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Reverse input sequence to get output
def reverse_sequence(seq):
    return ''.join([char for char, _ in sorted([(char, index) for char, index in target_tokenizer.items() if index in seq], key=lambda x: x[1])])

# Decode sequences
def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, len(target_tokenizer)+1))
    # Populate the first character of target sequence with the start character
    target_seq[0, 0, target_tokenizer['\t']] = 1.

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_sequence([sampled_token_index])
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character
        if sampled_char == '\n' or len(decoded_sentence) > max_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, len(target_tokenizer)+1))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Test the model
for seq_index in range(len(X)):
    input_seq = X[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input Sequence:', ''.join([char for char, index in sorted([(char, index) for char, index in input_tokenizer.items() if index in input_seq[0]], key=lambda x: x[1])]))
    print('Decoded Sentence:', decoded_sentence)
