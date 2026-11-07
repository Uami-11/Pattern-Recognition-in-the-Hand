
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Dataset configuration
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

SAMPLES_PER_GESTURE = 200

# Select gesture
print("\nAvailable gestures:")

for i, gesture in enumerate(GESTURES):
    print(f"{i + 1}. {gesture}")

choice = int(input("\nSelect gesture number: "))

if choice < 1 or choice > len(GESTURES):
    raise ValueError("Invalid gesture selection")

selected_gesture = GESTURES[choice - 1]

print(f"\nSelected gesture: {selected_gesture}")
print("Show your hand clearly in front of the camera.")
print("Press SPACE to start collection.")
print("Press Q to quit.")

# Extract and normalize landmarks
def extract_features(hand_landmarks):

    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ], dtype=np.float32)

    # Wrist as origin
    points = points - points[0]

    # Normalize hand size
    scale = np.max(
        np.linalg.norm(points[:, :2], axis=1)
    )

    if scale < 1e-6:
        return None

    points = points / scale

    return points.flatten()


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

new_samples = []
recording = False
last_capture = 0

try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        hand_detected = False

        if results.multi_hand_landmarks:

            hand_detected = True

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                if recording and len(new_samples) < SAMPLES_PER_GESTURE:

                    current_time = time.time()

                    if current_time - last_capture >= 0.1:

                        features = extract_features(hand_landmarks)

                        if features is not None:

                            row = features.tolist() + [
                                selected_gesture
                            ]

                            new_samples.append(row)

                            last_capture = current_time

        # Display information
        status = "RECORDING" if recording else "READY"

        detection = (
            "HAND DETECTED"
            if hand_detected
            else "NO HAND DETECTED"
        )

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
            detection,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if hand_detected else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {len(new_samples)}/{SAMPLES_PER_GESTURE}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if hand_detected:
                recording = True
                print("Recording started...")
            else:
                print("Show your hand before starting.")

        elif key == ord('q'):
            break

        if len(new_samples) >= SAMPLES_PER_GESTURE:
            print("\nCollection completed!")
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()


# Save only newly collected samples
if new_samples:

    columns = (
        [f"feature_{i}" for i in range(63)]
        + ["label"]
    )

    new_df = pd.DataFrame(
        new_samples,
        columns=columns
    )

    file_has_data = (
        os.path.exists(DATASET_PATH)
        and os.path.getsize(DATASET_PATH) > 0
    )

    new_df.to_csv(
        DATASET_PATH,
        mode="a" if file_has_data else "w",
        header=not file_has_data,
        index=False
    )

    print(f"\nSuccessfully saved {len(new_samples)} samples!")
    print(f"Gesture: {selected_gesture}")
    print(f"File: {DATASET_PATH}")

else:
    print("\nNo samples collected.")
    print("Make sure your hand is visible to the webcam.")
