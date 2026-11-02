
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l1

df = pd.read_csv("./datasets/MNIST/mnist_784_csv.csv")

print(df.head())

X = df.iloc[:, 1:].values

X = X.astype("float32") / 255.0

X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu", activity_regularizer=l1(0.0001)),
    Dense(784, activation="sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X_train, X_train, epochs=10,
          batch_size=256, validation_split=0.1)

reconstructed = model.predict(X_test[:5], verbose=0)

plt.figure(figsize=(10, 4))

for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title("Original")

    plt.subplot(2, 5, i+6)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title("Reconstructed")

plt.tight_layout()
plt.show()
