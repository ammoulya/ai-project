import tensorflow as tf
import numpy as np
import pickle
import pyttsx3
import cv2
import mediapipe as mp

# Load Model & Label Encoder
model = tf.keras.models.load_model("models/gesture_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Convert label numbers to text
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

engine = pyttsx3.init()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            if len(keypoints) == 42:  # Ensure correct shape
                keypoints = np.array(keypoints).reshape(1, 42)  
                prediction = model.predict(keypoints)
                gesture_label = np.argmax(prediction)
                gesture_text = label_map[gesture_label]

                print("Recognized Gesture:", gesture_text)

                engine.say(gesture_text)
                engine.runAndWait()

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
