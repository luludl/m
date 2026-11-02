import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

df = pd.read_csv('./datasets/creditcard.csv')

df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df = df.drop('Time', axis=1)

normal = df[df.Class == 0].drop('Class', axis=1).values
fraud = df[df.Class == 1].drop('Class', axis=1).values

X_train, X_val = train_test_split(normal, test_size=0.2, random_state=42)

n = X_train.shape[1]

model = Sequential([
    Input(shape=(n,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(n, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=5, batch_size=64,
          validation_data=(X_val, X_val), verbose=1)

normal_mse = np.mean((X_val - model.predict(X_val, verbose=0))**2, axis=1)
fraud_mse = np.mean((fraud - model.predict(fraud, verbose=0))**2, axis=1)

threshold = np.percentile(normal_mse, 95)
detected = np.sum(fraud_mse > threshold)

print("Normal Mean MSE:", np.mean(normal_mse))
print("Fraud Mean MSE:", np.mean(fraud_mse))
print("Threshold:", threshold)
print("Fraud Detection Recall:", round(detected / len(fraud_mse) * 100, 2), "%")

