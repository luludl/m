import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
import tensorflow as tf

data = ["the cat eats fish"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

words = tokenizer.word_index
vocab_size = len(words) + 1

seq = tokenizer.texts_to_sequences(data)[0]

X = []
Y = []

for i in range(1, len(seq)-1):
    X.append([seq[i-1], seq[i+1]])
    Y.append(seq[i])

X = np.array(X)
Y = np.array(Y)

model = Sequential([
    Embedding(vocab_size, 10),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, epochs=100, verbose=0)

test = np.array([[words['the'], words['cat']]])

prediction = model.predict(test, verbose=0)

index = np.argmax(prediction)
predicted_word = tokenizer.index_word[index]

print("Predicted word:", predicted_word)
