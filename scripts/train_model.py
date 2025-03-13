import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
df = pd.read_csv("dataset/gesture_dataset.csv")  

# Debugging: Print dataset shape
print("Dataset shape:", df.shape)

# Ensure correct number of features (42)
expected_features = 42  
if df.shape[1] - 1 != expected_features:
    raise ValueError(f"Dataset has {df.shape[1] - 1} features, but expected {expected_features}")

# Separate features (X) and labels (y)
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values  

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),  
    Dense(32, activation='relu'),  
    Dense(len(np.unique(y)), activation='softmax')  
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save Model
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")

# Save Label Encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("🎉 Model training complete! Saved as gesture_model.h5")
