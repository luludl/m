import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/MNIST/mnist_784_csv.csv")

X = df.drop('class', axis=1).values if 'class' in df.columns else df.values

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

X_train = X_train.astype('float32').reshape(-1, 28, 28) / 255.
X_test = X_test.astype('float32').reshape(-1, 28, 28) / 255.

# 3. Add noise
X_train_noisy = np.clip(X_train + 0.5 * np.random.normal(size=X_train.shape), 0, 1)
X_test_noisy = np.clip(X_test + 0.5 * np.random.normal(size=X_test.shape), 0, 1)

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train_noisy, X_train, epochs=5, batch_size=256,
          validation_data=(X_test_noisy, X_test))

denoised = model.predict(X_test_noisy[:5])

plt.figure(figsize=(10, 5))

for i in range(5):
    for j, img in enumerate([X_test[i], X_test_noisy[i], denoised[i]]):
        plt.subplot(3, 5, j * 5 + i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(['Original', 'Noisy', 'Denoised'][j])

plt.tight_layout()
plt.show()
