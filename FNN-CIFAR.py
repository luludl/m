import pandas as pd
import numpy as np
import tensorflow as tf

train = pd.read_csv("datasets/cifar10/train.csv")
test = pd.read_csv("datasets/cifar10/test.csv")

label_col = "label" if "label" in train.columns else train.columns[0]

y_train = train[label_col].values
X_train = train.drop(columns=[label_col]).values

y_test = test[label_col].values
X_test = test.drop(columns=[label_col]).values


X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 3072)
X_test = X_test.reshape(-1, 3072)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3072,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, batch_size=64,
          validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# Predict image
prediction = model.predict(X_test[:1], verbose=0)

classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("Predicted Class:", classes[np.argmax(prediction)])
print("Actual Class:", classes[int(y_test[0][0])])
