
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("datasets/MNIST/mnist_784_csv.csv")

if "label" in df.columns:
    label_col = "label"
else:
    label_col = df.columns[0]

y = df[label_col].values
X = df.drop(columns=[label_col]).values

X = X.astype("float32") / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(128, activation='relu',input_shape=(784,)),
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
print("Predicted Digit:", np.argmax(prediction))
print("Actual Digit:", y_test[0])
