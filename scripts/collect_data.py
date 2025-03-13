import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Ensure dataset folder exists
dataset_path = "dataset"
csv_path = os.path.join(dataset_path, "gesture_dataset.csv")

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)  # Create dataset folder if missing

# Create CSV with correct headers if it doesn’t exist
if not os.path.exists(csv_path):
    columns = ["x" + str(i) for i in range(42)] + ["label"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, index=False)
    print("✅ Created gesture_dataset.csv!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

data = []
labels = []

gesture_name = input("👉 Enter gesture name: ")  # Ask user for label

print(f"📸 Collecting data for '{gesture_name}', press 'Q' to stop...")

while len(data) < 500:  # Collect 500 samples
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])  # Only x, y

            if len(keypoints) == 42:  # Ensure correct number of keypoints
                data.append(keypoints)
                labels.append(gesture_name)

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data
df = pd.read_csv(csv_path)
new_data = pd.DataFrame(data, columns=["x" + str(i) for i in range(42)])
new_data["label"] = labels
df = pd.concat([df, new_data], ignore_index=True)
df.to_csv(csv_path, index=False)

print(f"✅ Data saved successfully in {csv_path}!")
