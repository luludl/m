
import numpy as np

X = np.array([[1], [2], [3], [4]])

# Initialize weights
np.random.seed(42)
Wx = np.random.randn(4, 1) * 0.1
Wh = np.random.randn(4, 4) * 0.1
Wy = np.random.randn(1, 4) * 0.1

h = np.zeros((4, 1))

for x in X:
    h = np.tanh(np.dot(Wx, x.reshape(1, 1)) + np.dot(Wh, h))
    print("Hidden State:", h.flatten())

y = np.dot(Wy, h)

print("\nFinal Hidden State:", h.flatten())
print("RNN Output:", y.flatten())
