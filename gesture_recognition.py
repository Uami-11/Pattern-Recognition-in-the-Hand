
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import pygame
import os
import time

# Load trained model
saved_data = joblib.load("models/gesture_model.pkl")
model = saved_data["model"]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Store recent predictions
prediction_history = deque(maxlen=10)

def extract_features(hand_landmarks):
    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ], dtype=np.float32)

    # Normalize coordinates relative to wrist
    points = points - points[0]

    # Normalize based on hand size
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale < 1e-6:
        return None

    points = points / scale

    return points.flatten().reshape(1, -1)


# Initialize sound system
pygame.mixer.init()

SOUND_FILES = {
    "OPEN_PALM": "sounds/open_palm.wav",
    "FIST": "sounds/fist.wav",
    "THUMBS_UP": "sounds/thumbs_up.wav",
    "PEACE": "sounds/peace.wav",
    "POINTING": "sounds/pointing.wav",
    "THREE_FINGERS": "sounds/three_fingers.wav"
}

sounds = {}

for gesture_name, file_path in SOUND_FILES.items():
    if os.path.exists(file_path):
        sounds[gesture_name] = pygame.mixer.Sound(file_path)
        print(f"Loaded sound: {gesture_name}")
    else:
        print(f"Sound file not found: {file_path}")

# Sound trigger variables
last_gesture = None
last_sound_time = 0
sound_cooldown = 1.0

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture = "NO HAND DETECTED"
    confidence = 0.0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = extract_features(hand_landmarks)

            if features is not None:

                # Predict gesture
                probabilities = model.predict_proba(features)[0]
                best_index = np.argmax(probabilities)

                predicted_gesture = model.classes_[best_index]
                confidence = probabilities[best_index]

                # Accept predictions with sufficient confidence
                if confidence >= 0.65:
                    prediction_history.append(predicted_gesture)
                else:
                    prediction_history.append("UNKNOWN")

                # Smooth predictions
                if prediction_history:
                    gesture = Counter(
                        prediction_history
                    ).most_common(1)[0][0]

                # Play sound when a new stable gesture is recognized
                current_time = time.time()

                if gesture != last_gesture:

                    if (
                        gesture in sounds
                        and current_time - last_sound_time >= sound_cooldown
                    ):
                        sounds[gesture].play()
                        last_sound_time = current_time

                    last_gesture = gesture

    else:
        prediction_history.clear()
        gesture = "NO HAND DETECTED"
        last_gesture = None

    # Display gesture
    cv2.rectangle(
        frame,
        (10, 10),
        (450, 110),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Hand Gesture Recognition System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close resources
cap.release()
hands.close()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("Application closed.")
