import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("===== TENSORFLOW =====")

a = tf.constant([1, 2, 3, 4]) # -> tensors are multidimensional arrays, similar to NumPy arrays
b = tf.constant([5, 6, 7, 8])

print("Tensor A:", a.numpy())
print("Tensor B:", b.numpy())
print("Addition:", (a + b).numpy())
print("Multiplication:", (a * b).numpy())

model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(
    tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=tf.float32),
    tf.constant([[1], [0]], dtype=tf.float32),
    epochs=5,
    verbose=0
)

print("TensorFlow Prediction:",
      model.predict(tf.constant([[1, 2, 3, 4]], dtype=tf.float32), verbose=0))

import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid

print("\n===== PYTORCH =====")

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

print("Tensor X:", x)
print("Tensor Y:", y)
print("Addition:", x + y)
print("Multiplication:", x * y)

model2 = Sequential(
    Linear(4, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid()
)

output = model2(torch.tensor([[1., 2., 3., 4.]]))

print("PyTorch Prediction:", output.detach().numpy())
