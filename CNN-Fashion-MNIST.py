

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

train = pd.read_csv("datasets/Fashion-MNIST/fashion-mnist_train.csv")
test = pd.read_csv("datasets/Fashion-MNIST/fashion-mnist_test.csv")

label_col = "label" if "label" in train.columns else train.columns[0]

y_train = train[label_col].values
X_train = train.drop(columns=[label_col]).values

y_test = test[label_col].values
X_test = test.drop(columns=[label_col]).values

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, batch_size=32,
          validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

prediction = model.predict(X_test[:1], verbose=0)
print("Predicted Class:", np.argmax(prediction))
print("Actual Class:", y_test[0])
