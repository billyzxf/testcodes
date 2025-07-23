import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
sentences = [
    'I love machine learning',
    'Deep learning is a subset of machine learning',
    'Natural language processing is a part of AI',
    'AI is transforming the world'
]

# Tokenize the sentences
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')

# Define the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=padded_sequences.shape[1]),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Dummy labels for demonstration
labels = [1, 0, 1, 0]

# Train the model
model.fit(padded_sequences, labels, epochs=10)