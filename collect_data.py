
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Configuration
DATASET_PATH = "dataset/gestures.csv"
SAMPLES_PER_GESTURE = 200

GESTURES = [
    "OPEN_PALM",
    "FIST",
    "THUMBS_UP",
    "PEACE",
    "POINTING",
    "THREE_FINGERS"
]

os.makedirs("dataset", exist_ok=True)

# Select gesture
print("\nSelect a gesture:")

for i, gesture in enumerate(GESTURES, 1):
    print(f"{i}. {gesture}")

choice = int(input("Enter gesture number: "))

if not 1 <= choice <= len(GESTURES):
    raise ValueError("Invalid gesture number")

selected_gesture = GESTURES[choice - 1]

print(f"\nSelected: {selected_gesture}")
print("Click on the webcam window.")
print("Press SPACE to start collecting.")
print("Press Q to stop.")

# Feature extraction
def extract_features(hand_landmarks):

    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ], dtype=np.float32)

    # Normalize relative to wrist
    points -= points[0]

    scale = np.max(
        np.linalg.norm(points[:, :2], axis=1)
    )

    if scale < 1e-6:
        return None

    points /= scale

    return points.flatten()


# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

samples = []
recording = False
last_capture = 0

try:
    while True:

        success, frame = cap.read()

        if not success:
            print("Camera frame could not be read.")
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

                # Collect samples
                if recording and len(samples) < SAMPLES_PER_GESTURE:

                    now = time.time()

                    if now - last_capture >= 0.1:

                        features = extract_features(hand_landmarks)

                        if features is not None:

                            row = features.tolist() + [
                                selected_gesture
                            ]

                            samples.append(row)
                            last_capture = now

                            # Print every 10 samples
                            if len(samples) % 10 == 0:
                                print(
                                    f"Collected: {len(samples)}/{SAMPLES_PER_GESTURE}"
                                )

        # Display status
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
            f"Samples: {len(samples)}/{SAMPLES_PER_GESTURE}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Gesture Data Collection", frame)

        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACEBAR

            if not recording:
                recording = True
                print("\nRECORDING STARTED!")

        elif key == ord('q'):
            print("\nStopping collection...")
            break

        if len(samples) >= SAMPLES_PER_GESTURE:
            print("\nCollection completed!")
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()


# Save collected data
if samples:

    columns = (
        [f"feature_{i}" for i in range(63)]
        + ["label"]
    )

    new_df = pd.DataFrame(samples, columns=columns)

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

    print("\nSuccessfully saved samples!")
    print("Gesture:", selected_gesture)
    print("Samples:", len(samples))
    print("Dataset:", DATASET_PATH)

else:
    print("\nNo samples collected.")
    print("Check webcam detection and keyboard input.")
