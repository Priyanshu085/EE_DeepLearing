import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load IMDb movie review dataset
vocab_size = 10000  # Keep only the top 10,000 most frequently occurring words
max_len = 200  # Maximum length of each review
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
model.add(LSTM(units=64))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict sentiment for new reviews
def predict_sentiment(review_text):
    # Preprocess the input text
    review_seq = imdb.get_word_index()[word.lower()] + 3  # Convert words to indices
    review_seq = pad_sequences([review_seq], maxlen=max_len)  # Pad sequence
    # Predict sentiment (positive or negative)
    prediction = model.predict(review_seq)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment