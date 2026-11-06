
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

DATASET_PATH = "dataset/gestures.csv"
os.makedirs("dataset", exist_ok=True)

GESTURES = [
    "OPEN_PALM",
    "FIST",
    "THUMBS_UP",
    "PEACE",
    "POINTING",
    "THREE_FINGERS"
]

SAMPLES_PER_GESTURE = 300


def extract_features(hand_landmarks):
    """
    Convert 21 landmarks into 63 normalized features.
    """

    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ], dtype=np.float32)

    # Use wrist as the origin
    points = points - points[0]

    # Normalize using hand size in x/y dimensions
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale < 1e-6:
        return None

    points = points / scale

    return points.flatten()


# Load existing dataset if available
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
    data = df.values.tolist()
else:
    data = []

print("\nAvailable gestures:")

for i, gesture in enumerate(GESTURES):
    print(f"{i + 1}. {gesture}")

choice = int(input("\nSelect gesture number: "))

if choice < 1 or choice > len(GESTURES):
    raise ValueError("Invalid gesture selection")

selected_gesture = GESTURES[choice - 1]

print(f"\nCollecting: {selected_gesture}")
print("Position your hand in front of the camera.")
print("Press SPACE to start collection.")
print("Press Q to quit.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

collected = 0
recording = False
last_capture = 0

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if recording and collected < SAMPLES_PER_GESTURE:

                current_time = time.time()

                # Capture approximately 10 samples per second
                if current_time - last_capture >= 0.1:

                    features = extract_features(hand_landmarks)

                    if features is not None:
                        row = features.tolist() + [selected_gesture]
                        data.append(row)
                        collected += 1
                        last_capture = current_time

    status = "RECORDING" if recording else "READY"

    cv2.putText(
        frame,
        f"Gesture: {selected_gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {collected}/{SAMPLES_PER_GESTURE}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Gesture Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        recording = True

    elif key == ord('q'):
        break

    if collected >= SAMPLES_PER_GESTURE:
        recording = False
        print(f"\nCompleted {selected_gesture}!")
        break


cap.release()
hands.close()
cv2.destroyAllWindows()

# Save collected dataset
if data:
    columns = (
        [f"feature_{i}" for i in range(63)]
        + ["label"]
    )

    new_df = pd.DataFrame(data, columns=columns)

    if os.path.exists(DATASET_PATH):
        new_df.to_csv(
            DATASET_PATH,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_df.to_csv(DATASET_PATH, index=False)

print(f"Saved {collected} samples for {selected_gesture}.")
